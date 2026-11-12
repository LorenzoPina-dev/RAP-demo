"""External API client powered by LiteLLM for multi-provider LLM access."""

from __future__ import annotations

import math
import os
import re
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from rap_demo.llm.base import BaseLLM, LLMConnectionError, LLMError, LLMTimeoutError
from rap_demo.llm.rate_limiter import TokenRateLimiter, resolve_model_rate_limits


# Reasoning models need room for their chain of thought before the final Yes/No.
# gpt-oss needs ~128 tokens; qwen3.5 needs ~400 before it emits its answer.
_SELF_EVAL_RETRY_MAX_TOKENS = 512
# Budget used when the model is asked to propose the next action so that the
# logprob of the generated action tokens can be read back.
_ACTION_SCORING_MAX_TOKENS = 512
# LiteLLM treats the OpenAI-compatible Ollama endpoints as OpenAI and refuses to
# call them without credentials, even though Ollama ignores the key entirely.
_OLLAMA_API_KEY = "ollama"


def _token_probabilities_are_usable(
    probabilities: Mapping[str, float],
    content: str | None,
) -> bool:
    """Return True when a Yes/No request produced an answer or a real distribution."""
    if any(value > 0.0 for value in probabilities.values()):
        return True
    # Reasoning models may spend the whole budget thinking: text such as
    # "Thinking" or a truncated rationale is not a usable final answer, so the
    # caller retries with a larger budget.
    return _answer_token_probabilities(content, ("Yes", "No")) is not None


def _load_dotenv_file() -> None:
    """Load environment variables from the nearest .env file without overriding shell vars."""
    from dotenv import find_dotenv, load_dotenv

    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=False)


def _ensure_litellm_compat() -> None:
    """Configure offline cost-map defaults and Python 3.15 stdlib shims for litellm."""
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    if sys.version_info >= (3, 15):
        import re

        for legacy_name, replacement_attr in (
            ("sre_constants", "_constants"),
            ("sre_parse", "_parser"),
            ("sre_compile", "_compiler"),
        ):
            if legacy_name not in sys.modules and hasattr(re, replacement_attr):
                sys.modules[legacy_name] = getattr(re, replacement_attr)


_ensure_litellm_compat()


def _load_litellm() -> Any:
    """Lazily import litellm after applying compatibility shims."""
    _ensure_litellm_compat()
    import litellm

    litellm.suppress_debug_info = True
    return litellm


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    """Read an attribute or mapping key uniformly from object or dict responses."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_first_choice(response: Any) -> Any:
    """Extract choices[0] from an object-style or dict-style response."""
    choices = _get_field(response, "choices")
    if not choices or not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
        return None
    return choices[0]


def _extract_thinking_blocks_text(message: Any) -> str | None:
    """Extract concatenated text from LiteLLM/Anthropic/Gemini thinking_blocks if present."""
    blocks = _get_field(message, "thinking_blocks")
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
        return None
    parts: list[str] = []
    for block in blocks:
        text = _get_field(block, "thinking") or _get_field(block, "text")
        if text:
            parts.append(str(text))
    return "\n".join(parts) if parts else None


def _extract_message_content(choice: Any) -> str | None:
    """Extract choices[0].message.content, falling back to reasoning_content or thinking_blocks."""
    message = _get_field(choice, "message")
    content = _get_field(message, "content")
    if content is not None:
        text = str(content)
        if text:
            return text

    reasoning = _get_field(message, "reasoning_content")
    if reasoning is not None and str(reasoning).strip():
        return str(reasoning)

    thinking_text = _extract_thinking_blocks_text(message)
    if thinking_text and thinking_text.strip():
        return thinking_text

    if content is not None:
        return str(content)
    return None


def _is_unsupported_param_error(
    exc: Exception,
    markers: tuple[str, ...] = ("unsupported", "logprobs", "top_logprobs", "not supported"),
) -> bool:
    """Return True when exc indicates the provider rejected an optional parameter."""
    litellm = sys.modules.get("litellm")
    if litellm is None:
        return False
    unsupported_cls = getattr(litellm, "UnsupportedParamsError", None)
    if unsupported_cls is not None and isinstance(exc, unsupported_cls):
        return True
    bad_request_cls = getattr(litellm, "BadRequestError", None)
    if bad_request_cls is not None and isinstance(exc, bad_request_cls):
        msg = str(exc).lower()
        return any(marker in msg for marker in markers)
    return False


def _extract_delta_segments(delta: Any) -> list[tuple[str, str]]:
    """Extract (kind, text) segments from an object-style or dict-style stream delta."""
    segments: list[tuple[str, str]] = []

    reasoning = _get_field(delta, "reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        segments.append(("reasoning", reasoning))

    thinking_blocks = _get_field(delta, "thinking_blocks")
    if isinstance(thinking_blocks, Sequence) and not isinstance(thinking_blocks, (str, bytes)):
        for block in thinking_blocks:
            text = _get_field(block, "thinking") or _get_field(block, "text")
            if isinstance(text, str) and text:
                segments.append(("reasoning", text))

    content = _get_field(delta, "content")
    if isinstance(content, str) and content:
        segments.append(("content", content))

    return segments


def _is_chunk_stream(response: Any) -> bool:
    """Return True if response is an iterable stream of chunks rather than a single response."""
    if isinstance(response, (Mapping, str, bytes)):
        return False
    return isinstance(response, (Iterator, list, tuple)) or hasattr(response, "__next__")


def _is_retryable_error(exc: Exception) -> bool:
    """Return True when exc is a transient rate-limit or 5xx server error."""
    litellm = sys.modules.get("litellm")
    if litellm is None:
        return False
    for attr in ("RateLimitError", "InternalServerError", "ServiceUnavailableError"):
        exc_cls = getattr(litellm, attr, None)
        if exc_cls is not None and isinstance(exc, exc_cls):
            return True
    return False


def _map_exception(exc: Exception) -> LLMError:
    """Translate standard library and litellm exceptions into the LLMError hierarchy."""
    if isinstance(exc, LLMError):
        return exc
    if isinstance(exc, TimeoutError):
        return LLMTimeoutError(f"LiteLLM request timed out: {exc}")
    if isinstance(exc, ConnectionError):
        return LLMConnectionError(f"LiteLLM connection failed: {exc}")

    litellm = sys.modules.get("litellm")
    if litellm is not None:
        # Check Timeout before APIConnectionError because litellm.Timeout inherits APIConnectionError.
        timeout_cls = getattr(litellm, "Timeout", None)
        if timeout_cls is not None and isinstance(exc, timeout_cls):
            return LLMTimeoutError(f"LiteLLM request timed out: {exc}")

        conn_cls = getattr(litellm, "APIConnectionError", None)
        if conn_cls is not None and isinstance(exc, conn_cls):
            return LLMConnectionError(f"LiteLLM connection failed: {exc}")

    return LLMError(f"LiteLLM error: {exc}")


def _extract_top_logprob_candidates(logprobs: Any) -> dict[str, float]:
    """Extract {token: logprob} candidates from chat or legacy top_logprobs structures."""
    if logprobs is None:
        return {}

    content_lps = _get_field(logprobs, "content")
    if isinstance(content_lps, Sequence) and len(content_lps) > 0:
        first_top = _get_field(content_lps[0], "top_logprobs")
        if isinstance(first_top, Sequence) and len(first_top) > 0:
            candidates: dict[str, float] = {}
            for item in first_top:
                tok = _get_field(item, "token")
                lp = _get_field(item, "logprob")
                if tok is not None and lp is not None:
                    candidates[str(tok)] = float(lp)
            return candidates

    legacy_top = _get_field(logprobs, "top_logprobs")
    if isinstance(legacy_top, Sequence) and len(legacy_top) > 0:
        first_map = legacy_top[0]
        if isinstance(first_map, Mapping):
            return {
                str(tok): float(lp)
                for tok, lp in first_map.items()
                if lp is not None
            }

    return {}


def _extract_logprob_entries(logprobs: Any) -> list[Any]:
    """Return per-token logprob entries from chat-style or legacy logprobs payloads."""
    content_lps = _get_field(logprobs, "content")
    if isinstance(content_lps, Sequence) and not isinstance(content_lps, (str, bytes)):
        return list(content_lps)
    return []


def _extract_last_answer_candidates(
    logprobs: Any,
    tokens: Sequence[str],
) -> dict[str, float]:
    """Return the candidate distribution at the last Yes/No token in the response.

    Reasoning models emit their chain of thought first and only then the final
    answer. Looking at the first generated token (the old behaviour) therefore
    reads a channel marker such as ``<|channel|>`` instead of the answer. This
    walks every generated token and keeps the *last* one that matches a candidate.
    """
    wanted = {token.strip().casefold() for token in tokens if token.strip()}
    if not wanted:
        return {}

    for entry in reversed(_extract_logprob_entries(logprobs)):
        answer_token = _get_field(entry, "token")
        if answer_token is None or str(answer_token).strip().casefold() not in wanted:
            continue

        candidates: dict[str, float] = {}
        top = _get_field(entry, "top_logprobs")
        if isinstance(top, Sequence) and not isinstance(top, (str, bytes)):
            for item in top:
                tok = _get_field(item, "token")
                lp = _get_field(item, "logprob")
                if tok is not None and lp is not None:
                    candidates[str(tok)] = float(lp)

        answer_logprob = _get_field(entry, "logprob")
        if answer_logprob is not None:
            candidates.setdefault(str(answer_token), float(answer_logprob))
        if candidates:
            return candidates

    return {}


def _match_token_logprob(candidates: Mapping[str, float], token: str) -> float | None:
    """Find logprob for token using exact, leading-space, then stripped matching."""
    if token in candidates:
        return candidates[token]
    spaced = f" {token}"
    if spaced in candidates:
        return candidates[spaced]
    token_stripped = token.strip()
    for candidate_tok, lp in candidates.items():
        if candidate_tok.strip() == token_stripped:
            return lp
    return None


def _answer_token_probabilities(content: str | None, tokens: Sequence[str]) -> dict[str, float] | None:
    """Convert an explicit Yes/No answer in the text to a one-hot fallback.

    Uses the last standalone occurrence so a final answer wins over a mention
    that appeared earlier in the response.
    """
    if not content:
        return None

    lowered = content.casefold()
    best_position = -1
    best_answer: str | None = None
    for token in tokens:
        needle = token.strip().casefold()
        if not needle:
            continue
        for match in re.finditer(rf"(?<!\w){re.escape(needle)}(?!\w)", lowered):
            if match.start() > best_position:
                best_position = match.start()
                best_answer = needle

    if best_answer is None:
        return None
    return {
        token: float(token.strip().casefold() == best_answer)
        for token in tokens
    }


class LiteLLMClient(BaseLLM):
    """LLM backend client using LiteLLM for 100+ external API providers."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: float = 30.0,
        completion_fn: Callable[..., Any] | None = None,
        tokens_per_minute: int | None = None,
        requests_per_minute: int | None = None,
        max_retries: int = 3,
        rate_limiter: TokenRateLimiter | None = None,
        sleeper: Callable[[float], None] | None = None,
        config_path: str | Path | None = None,
        on_token: Callable[[str, str], None] | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        """Initialize LiteLLM client.

        Args:
            model: LiteLLM model identifier (e.g. 'gpt-4o-mini', 'anthropic/claude-3-5-sonnet-20241022').
            api_key: Optional API key overriding provider environment variable.
            api_base: Optional custom API base URL (e.g. proxy or self-hosted endpoint).
            timeout: Request timeout in seconds.
            completion_fn: Optional completion callable override for dependency injection/testing.
            tokens_per_minute: Optional TPM rate limit (falls back to env / models.json).
            requests_per_minute: Optional RPM rate limit (falls back to env / models.json).
            max_retries: Maximum retries on litellm.RateLimitError (default: 3).
            rate_limiter: Optional preconfigured TokenRateLimiter instance.
            sleeper: Optional sleep callable used for rate-limit retries and limiter waits.
            config_path: Optional path to models.json configuration file.

        Raises:
            ValueError: If model is empty/whitespace-only or rate-limit settings are invalid.
        """
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"Invalid model identifier: {model!r}. Expected non-empty string.")
        if max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {max_retries}.")

        _load_dotenv_file()

        self.model: str = model.strip()
        effective_tpm, effective_rpm = resolve_model_rate_limits(
            self.model,
            tokens_per_minute=tokens_per_minute,
            requests_per_minute=requests_per_minute,
            config_path=config_path,
        )

        self.api_key: str | None = api_key
        self.api_base: str | None = api_base
        self.timeout: float = timeout
        self.max_retries: int = max_retries
        self._sleeper: Callable[[float], None] = sleeper if sleeper is not None else time.sleep
        self._completion_fn: Callable[..., Any] | None = completion_fn
        self._call_count: int = 0
        self.on_token: Callable[[str, str], None] | None = on_token
        self.reasoning_effort: str | None = reasoning_effort

        if rate_limiter is not None:
            self.rate_limiter: TokenRateLimiter | None = rate_limiter
        elif effective_tpm is not None or effective_rpm is not None:
            self.rate_limiter = TokenRateLimiter(
                tokens_per_minute=effective_tpm,
                requests_per_minute=effective_rpm,
                sleeper=sleeper,
            )
        else:
            self.rate_limiter = None

    @property
    def call_count(self) -> int:
        """Return the total number of completion requests issued by this client."""
        return self._call_count

    def _build_completion_kwargs(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Construct keyword arguments for litellm.completion, omitting None optionals."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": self.timeout,
        }
        if stop is not None:
            kwargs["stop"] = stop
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.api_base is not None:
            kwargs["api_base"] = self.api_base
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        kwargs.update(extra)
        return kwargs

    def _estimate_request_tokens(self, kwargs: Mapping[str, Any]) -> int:
        """Estimate token usage from completion request messages and max_tokens."""
        messages = kwargs.get("messages") or []
        prompt_text = " ".join(
            str(_get_field(message, "content", "")) for message in messages
        )
        if not prompt_text and kwargs.get("prompt") is not None:
            prompt_text = str(kwargs["prompt"])
        max_tokens = int(kwargs.get("max_tokens") or 0)
        return TokenRateLimiter.estimate_tokens(prompt_text, max_tokens=max_tokens)

    def _record_usage(self, estimated_tokens: int, response: Any) -> None:
        """Reconcile rate limiter budget using response.usage.total_tokens if present."""
        if self.rate_limiter is None:
            return
        usage = _get_field(response, "usage")
        total_tokens = _get_field(usage, "total_tokens")
        if total_tokens is not None:
            self.rate_limiter.record_actual_tokens(estimated_tokens, int(total_tokens))

    def _ollama_openai_config(self) -> tuple[str, str] | None:
        """Return the native OpenAI-compatible Ollama model and API base when applicable."""
        if not self.model.lower().startswith("ollama/"):
            return None
        model = f"openai/{self.model.split('/', 1)[1]}"
        api_base = (self.api_base or "http://localhost:11434/v1").rstrip("/")
        if not api_base.lower().endswith("/v1"):
            api_base = f"{api_base}/v1"
        return model, api_base

    def _invoke_completion(self, kwargs: dict[str, Any], *, allow_retries: bool = True) -> Any:
        """Invoke completion with rate-limiting, retry on transient errors, and usage metering."""
        fn = self._completion_fn if self._completion_fn is not None else _load_litellm().completion
        estimated_tokens = self._estimate_request_tokens(kwargs)
        max_attempts = (self.max_retries + 1) if allow_retries else 1

        for attempt in range(max_attempts):
            if self.rate_limiter is not None:
                waited = self.rate_limiter.acquire(estimated_tokens)
                if waited > 0.1:
                    print(
                        f"[RateLimiter] Waited {waited:.1f}s before calling {self.model}",
                        file=sys.stderr,
                        flush=True,
                    )
            self._call_count += 1
            try:
                response = fn(**kwargs)
            except Exception as exc:
                if self.rate_limiter is not None:
                    self.rate_limiter.record_actual_tokens(estimated_tokens, 0)
                if allow_retries and _is_retryable_error(exc) and attempt < self.max_retries:
                    delay = 0.5 * (2**attempt)
                    print(
                        f"[LiteLLM] Transient error on {self.model} (attempt {attempt + 1}/{self.max_retries}), retrying in {delay:.1f}s: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    self._sleeper(delay)
                    continue
                raise

            self._record_usage(estimated_tokens, response)
            return response

        raise LLMError("LiteLLM completion failed after exhausting retries")

    def _open_stream(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str] | None,
    ) -> tuple[Any, dict[str, Any]]:
        """Open a streaming completion request, falling back if stream_options is unsupported."""
        kwargs = self._build_completion_kwargs(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            stream=True,
            stream_options={"include_usage": True},
        )
        try:
            return self._invoke_completion(kwargs), kwargs
        except Exception as exc:
            if not _is_unsupported_param_error(
                exc,
                markers=("unsupported", "not supported", "stream_options", "unknown parameter"),
            ):
                raise _map_exception(exc) from exc
            kwargs_no_opts = self._build_completion_kwargs(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                stream=True,
            )
            try:
                return self._invoke_completion(kwargs_no_opts), kwargs_no_opts
            except Exception as retry_exc:
                raise _map_exception(retry_exc) from retry_exc

    def _generate_non_streaming(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str] | None,
        allow_retries: bool = True,
    ) -> str:
        """Execute non-streaming completion request and extract text content."""
        kwargs = self._build_completion_kwargs(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        try:
            response = self._invoke_completion(kwargs, allow_retries=allow_retries)
        except Exception as exc:
            raise _map_exception(exc) from exc

        choice = _extract_first_choice(response)
        if choice is None:
            raise LLMError("Malformed LLM response: missing or empty choices")

        content = _extract_message_content(choice)
        if content is None:
            raise LLMError("Malformed LLM response: missing message content")

        return content

    def _generate_streaming(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str] | None,
    ) -> str:
        """Stream completion chunks, retrying transient errors with non-streaming fallback."""
        callback = self.on_token

        for attempt in range(self.max_retries + 1):
            raw_stream, kwargs = self._open_stream(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
            if not _is_chunk_stream(raw_stream):
                choice = _extract_first_choice(raw_stream)
                if choice is None:
                    raise LLMError("Malformed LLM response: missing or empty choices")
                content = _extract_message_content(choice)
                if content is None:
                    raise LLMError("Malformed LLM response: missing message content")
                if content and callback is not None:
                    callback("content", content)
                return content

            estimated_tokens = self._estimate_request_tokens(kwargs)
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            saw_content_field = False

            try:
                for chunk in raw_stream:
                    self._record_usage(estimated_tokens, chunk)
                    choice = _extract_first_choice(chunk)
                    if choice is None:
                        continue
                    delta = _get_field(choice, "delta")
                    if delta is None:
                        continue
                    if _get_field(delta, "content") is not None:
                        saw_content_field = True
                    for kind, text in _extract_delta_segments(delta):
                        if kind == "reasoning":
                            reasoning_parts.append(text)
                        else:
                            content_parts.append(text)
                        if callback is not None:
                            callback(kind, text)
            except Exception as exc:
                if self.rate_limiter is not None:
                    self.rate_limiter.record_actual_tokens(estimated_tokens, 0)
                if _is_retryable_error(exc) and not content_parts and not reasoning_parts:
                    if attempt < self.max_retries:
                        delay = 0.5 * (2**attempt)
                        print(
                            f"[LiteLLM] Mid-stream transient error on {self.model} (attempt {attempt + 1}/{self.max_retries}), retrying in {delay:.1f}s: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                        self._sleeper(delay)
                        continue
                    print(
                        f"[LiteLLM] Streaming failed on {self.model}; falling back to non-streaming mode.",
                        file=sys.stderr,
                        flush=True,
                    )
                    fallback_text = self._generate_non_streaming(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=stop,
                        allow_retries=False,
                    )
                    if fallback_text and callback is not None:
                        callback("content", fallback_text)
                    return fallback_text
                raise _map_exception(exc) from exc

            if content_parts:
                return "".join(content_parts)
            if reasoning_parts:
                return "".join(reasoning_parts)
            if saw_content_field:
                return ""
            raise LLMError("Malformed LLM streaming response: empty stream with no content")

        raise LLMError("LiteLLM streaming completion failed after exhausting retries")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        """Generate text completion from a prompt via LiteLLM.

        Args:
            prompt: Text prompt input to LLM.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            stop: Optional list of stop sequences.

        Returns:
            The generated text string.

        Raises:
            LLMTimeoutError: If the request times out.
            LLMConnectionError: If the provider cannot be reached.
            LLMError: If the API returns an error or malformed response.
        """
        if self.on_token is not None:
            return self._generate_streaming(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )

        return self._generate_non_streaming(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )

    def get_logprobs(self, prompt: str, target: str) -> float:
        """Score a continuation when the provider can return prompt-token logprobs."""
        ollama = self._ollama_openai_config()
        if ollama is None:
            return 0.0
        model, api_base = ollama
        kwargs: dict[str, Any] = {
            "model": model,
            "prompt": f"{prompt}{target}",
            "max_tokens": 0,
            "temperature": 0.0,
            "logprobs": 1,
            "echo": True,
            "timeout": self.timeout,
            "api_base": api_base,
            "api_key": self.api_key or _OLLAMA_API_KEY,
        }
        try:
            response = self._invoke_completion(kwargs)
        except Exception as exc:
            if _is_unsupported_param_error(exc):
                return 0.0
            raise _map_exception(exc) from exc

        choice = _extract_first_choice(response)
        logprobs = _get_field(choice, "logprobs")
        if logprobs is None:
            return 0.0

        token_lps = _get_field(logprobs, "token_logprobs")
        offsets = _get_field(logprobs, "text_offset")
        if not (
            isinstance(token_lps, Sequence)
            and not isinstance(token_lps, (str, bytes))
            and isinstance(offsets, Sequence)
            and not isinstance(offsets, (str, bytes))
            and len(token_lps) == len(offsets)
        ):
            return 0.0
        target_logprobs = [
            float(lp)
            for offset, lp in zip(offsets, token_lps)
            if offset is not None and int(offset) >= len(prompt) and lp is not None
        ]
        return float(sum(target_logprobs)) if target_logprobs else 0.0

    def get_token_probability(self, prompt: str, token: str) -> float:
        """Return one candidate probability using the shared multi-token request."""
        return self.get_token_probabilities(prompt, [token]).get(token, 0.0)

    def get_token_probabilities(self, prompt: str, tokens: Sequence[str]) -> dict[str, float]:
        """Return candidate probabilities from a single completion request."""
        probabilities, _ = self.get_token_probabilities_and_response(prompt, tokens)
        return probabilities

    def get_token_probabilities_and_response(
        self,
        prompt: str,
        tokens: Sequence[str],
    ) -> tuple[dict[str, float], str | None]:
        """Extract the probability of a candidate token following prompt.

        Returns the generated text as well, so R3 can log the model's answer from the
        same request instead of issuing a separate generation call.

        Some models are "reasoners" (for example gpt-oss via Ollama): with a
        one-token budget they only emit a channel marker such as ``<|channel|>``
        and never reach the Yes/No answer. When that happens the request is retried
        with a larger budget so the answer and its logprob can be read.
        """
        probabilities, content = self._score_candidate_tokens(prompt, tokens, max_tokens=1)

        if _token_probabilities_are_usable(probabilities, content):
            return probabilities, content

        retry_probabilities, retry_content = self._score_candidate_tokens(
            prompt,
            tokens,
            max_tokens=_SELF_EVAL_RETRY_MAX_TOKENS,
        )
        if _token_probabilities_are_usable(retry_probabilities, retry_content):
            return retry_probabilities, retry_content
        if (retry_content or "").strip() and not (content or "").strip():
            return retry_probabilities, retry_content
        return probabilities, content

    def _score_candidate_tokens(
        self,
        prompt: str,
        tokens: Sequence[str],
        *,
        max_tokens: int,
    ) -> tuple[dict[str, float], str | None]:
        """Issue one logprob request and map it to the requested candidate tokens."""
        kwargs = self._build_completion_kwargs(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            logprobs=True,
            top_logprobs=10,
        )
        ollama = self._ollama_openai_config()
        if ollama is not None:
            kwargs["model"], kwargs["api_base"] = ollama
            kwargs["api_key"] = self.api_key or _OLLAMA_API_KEY
        try:
            response = self._invoke_completion(kwargs)
        except Exception as exc:
            if not _is_unsupported_param_error(exc):
                raise _map_exception(exc) from exc

            fallback_kwargs = self._build_completion_kwargs(
                prompt,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            try:
                response = self._invoke_completion(fallback_kwargs)
            except Exception as fallback_exc:
                if _is_unsupported_param_error(fallback_exc):
                    return {token: 0.0 for token in tokens}, None
                raise _map_exception(fallback_exc) from fallback_exc

        choice = _extract_first_choice(response)
        if choice is None:
            return {token: 0.0 for token in tokens}, None

        content = _extract_message_content(choice)
        logprobs = _get_field(choice, "logprobs")
        candidates = _extract_last_answer_candidates(logprobs, tokens)
        if not candidates:
            candidates = _extract_top_logprob_candidates(logprobs)
        if candidates:
            matched = {
                token: _match_token_logprob(candidates, token)
                for token in tokens
            }
            if any(logprob is not None for logprob in matched.values()):
                return (
                    {
                        token: float(max(0.0, min(1.0, math.exp(logprob))))
                        if logprob is not None
                        else 0.0
                        for token, logprob in matched.items()
                    },
                    content,
                )

        answer_probabilities = _answer_token_probabilities(content, tokens)
        if answer_probabilities is not None:
            return answer_probabilities, content
        return {token: 0.0 for token in tokens}, content
