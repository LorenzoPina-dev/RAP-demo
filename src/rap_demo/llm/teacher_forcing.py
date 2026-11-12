"""Teacher-forced scoring of candidate continuations.

The RAP paper defines R1 (the "likelihood of the action" reward, Section 4.1) as
the log probability of the action continuation given the current state, and the
reference implementation computes it with a single forward pass per candidate
(``LLaMA.get_ll`` in the official code).

Chat APIs only report the logprob of the tokens they *generate*, so an
arbitrary continuation has to be scored token by token: the continuation is
forced into the assistant turn and the distribution of the next token is read
from ``top_logprobs``. Candidates usually share prefixes, so the requests are
grouped per distinct prefix and issued in parallel.

This module keeps that logic provider-agnostic (the request function is
injected) and adds an Ollama implementation, because Ollama's OpenAI-compatible
``/v1/completions`` endpoint silently drops the ``echo`` and ``logprobs``
parameters, and its native ``/api/generate`` endpoint cannot force a chat
continuation.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

# One request returns this many alternatives for the next token.
DEFAULT_TOP_LOGPROBS = 20
# Number of parallel requests used to score the candidate actions.
DEFAULT_MAX_WORKERS = 8

RequestFn = Callable[[str], Sequence[tuple[str, float]]]


@dataclass
class _Candidate:
    """A candidate continuation and the prefix already scored for it."""

    action: str
    prefix: str
    remaining: str
    total: float = 0.0
    tokens: list[str] = field(default_factory=list)
    exact: bool = True


def score_continuations(
    actions: Sequence[str],
    request: RequestFn,
    *,
    seed: str = "",
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, float]:
    """Score every candidate continuation with teacher forcing.

    Args:
        actions: Candidate continuations to score.
        request: Callable returning the ``(token, logprob)`` alternatives for
            the token that follows the given prefix.
        seed: Context the continuations are appended to. It is sent to the
            backend but is not part of the scored text.
        max_workers: Parallel requests used per scoring round.

    Returns:
        Mapping of candidate continuation to its summed log probability.
    """
    pending = [
        _Candidate(action=action, prefix=seed, remaining=action)
        for action in actions
        if action
    ]
    scores: dict[str, float] = {}

    while pending:
        prefixes = sorted({candidate.prefix for candidate in pending})
        if len(prefixes) == 1:
            alternatives = {prefixes[0]: list(request(prefixes[0]))}
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                results = list(pool.map(request, prefixes))
            alternatives = dict(zip(prefixes, results))

        remaining_candidates: list[_Candidate] = []
        for candidate in pending:
            options = alternatives.get(candidate.prefix, [])
            token, logprob = _match_next_token(candidate.remaining, options)
            if token is None:
                # The candidate's next token is not among the reported
                # alternatives, so the exact probability is unknown. Charge the
                # least likely alternative and stop: the text no longer matches,
                # so continuing would not make progress.
                if not options:
                    scores[candidate.action] = candidate.total
                    continue
                _, logprob = min(options, key=lambda option: option[1])
                candidate.total += logprob
                candidate.exact = False
                scores[candidate.action] = candidate.total
                continue
            candidate.total += logprob
            candidate.tokens.append(token)
            candidate.prefix += token
            candidate.remaining = candidate.remaining[len(token) :]
            if candidate.remaining:
                remaining_candidates.append(candidate)
            else:
                scores[candidate.action] = candidate.total
        pending = remaining_candidates

    for action in actions:
        scores.setdefault(action, 0.0)
    return scores


def _match_next_token(
    remaining: str,
    options: Sequence[tuple[str, float]],
) -> tuple[str | None, float]:
    """Return the alternative token that continues ``remaining``, if any."""
    best: tuple[str | None, float] = (None, 0.0)
    for token, logprob in options:
        if not token or not remaining.startswith(token):
            continue
        if best[0] is None or len(token) > len(best[0]):
            best = (token, logprob)
    return best


class OllamaChatScorer:
    """Scores candidate continuations with Ollama's OpenAI-compatible chat API.

    The continuation is forced into the assistant turn, so the model continues
    it instead of starting a new answer. The logprob of the next token is read
    from ``top_logprobs``, which Ollama's chat endpoint does report.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = 300.0,
        top_logprobs: int = DEFAULT_TOP_LOGPROBS,
        max_workers: int = DEFAULT_MAX_WORKERS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.top_logprobs = top_logprobs
        self.max_workers = max_workers

    @classmethod
    def from_model_name(
        cls,
        model: str,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> "OllamaChatScorer | None":
        """Build a scorer for an ``ollama/<model>`` name, or None if not Ollama."""
        if not model.lower().startswith("ollama/"):
            return None
        url = (base_url or "http://localhost:11434").rstrip("/")
        if url.endswith("/v1"):
            url = url[: -len("/v1")]
        return cls(base_url=url, model=model.split("/", 1)[1], **kwargs)

    def next_token_alternatives(
        self,
        user_prompt: str,
        assistant_prefix: str,
    ) -> list[tuple[str, float]]:
        """Return the alternatives that follow ``assistant_prefix``.

        Args:
            user_prompt: The prompt the continuation is scored against.
            assistant_prefix: The continuation forced into the assistant turn.

        Returns:
            The ``(token, logprob)`` alternatives for the next token.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_prefix},
            ],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            "stream": False,
        }
        data = self._post(payload)
        choices = data.get("choices") or []
        if not choices:
            return []
        logprobs = choices[0].get("logprobs") or {}
        entries = logprobs.get("content") or []
        if not entries:
            return []
        alternatives = entries[0].get("top_logprobs") or []
        return [
            (str(option.get("token", "")), float(option.get("logprob", 0.0)))
            for option in alternatives
            if option.get("token")
        ]

    def score(
        self,
        prompt: str,
        actions: Sequence[str],
        *,
        assistant_prefix: str = "",
    ) -> dict[str, float]:
        """Score every candidate continuation against ``prompt``.

        Args:
            prompt: The prompt the continuations are scored against.
            actions: Candidate continuations.
            assistant_prefix: Context the continuations are appended to inside
                the assistant turn. It is not part of the scored text.

        Returns:
            Mapping of candidate continuation to summed log probability.
        """

        def request(prefix: str) -> list[tuple[str, float]]:
            return self.next_token_alternatives(prompt, prefix)

        return score_continuations(
            actions,
            request,
            seed=assistant_prefix,
            max_workers=self.max_workers,
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer ollama",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:  # pragma: no cover - network failure
            raise OSError(f"Ollama request failed: {exc}") from exc
