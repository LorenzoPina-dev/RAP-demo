"""HTTP client for local llama-server OpenAI-compatible API."""

from __future__ import annotations

import json
import math
import re
import subprocess
from collections.abc import Callable
from typing import Any
import httpx
from rap_demo.llm.base import BaseLLM, LLMConnectionError, LLMError, LLMTimeoutError


def _raise_for_http_error(url: str, response: httpx.Response) -> None:
    """Raise LLMError with parsed server error details for 4xx/5xx responses."""
    error_detail = ""
    try:
        err_json = response.json()
        if isinstance(err_json, dict):
            err_obj = err_json.get("error")
            if isinstance(err_obj, dict):
                error_detail = str(err_obj.get("message", ""))
            elif isinstance(err_obj, str):
                error_detail = err_obj
    except Exception:
        error_detail = response.text
    detail_str = f": {error_detail}" if error_detail else ""
    raise LLMError(f"HTTP {response.status_code} error from {url}{detail_str}")


def _map_httpx_error(url: str, exc: httpx.HTTPError) -> LLMError:
    """Translate httpx exceptions into the LLMError hierarchy."""
    if isinstance(exc, httpx.TimeoutException):
        return LLMTimeoutError(f"Request to {url} timed out: {exc}")
    if isinstance(exc, (httpx.ConnectError, httpx.NetworkError)):
        return LLMConnectionError(f"Failed to connect to {url}: {exc}")
    return LLMError(f"HTTP error communicating with {url}: {exc}")


class LlamaServerClient(BaseLLM):
    """Client connecting to llama-server OpenAI-compatible completions endpoints."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 30.0,
        model: str | None = None,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        server_command: list[str] | None = None,
        on_token: Callable[[str, str], None] | None = None,
    ) -> None:
        """Initialize llama-server client.

        Args:
            base_url: Base URL where llama-server is listening (default: http://localhost:8080).
            timeout: HTTP request timeout in seconds.
            model: Optional model identifier string for completions payload.
            api_key: Optional Bearer API key token.
            client: Optional preconfigured httpx.Client (e.g. for testing with mock transport).
            server_command: Optional command list to auto-launch llama-server as a subprocess.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.model = model
        self.api_key = api_key
        self._server_command = server_command
        self._server_process: subprocess.Popen[bytes] | None = None
        self.on_token: Callable[[str, str], None] | None = on_token

        self._headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

        if client is not None:
            self._client = client
        else:
            self._client = httpx.Client(timeout=timeout)

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Perform HTTP POST request with error mapping to LLM exceptions.

        Args:
            endpoint: URL path segment (e.g. '/v1/completions').
            payload: JSON request body.

        Returns:
            Decoded JSON dictionary response.

        Raises:
            LLMTimeoutError: If the request times out.
            LLMConnectionError: If network connection to the server fails.
            LLMError: If HTTP error occurs or response is malformed.
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = self._client.post(
                url,
                json=payload,
                headers=self._headers,
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise _map_httpx_error(url, exc) from exc

        if response.status_code >= 400:
            _raise_for_http_error(url, response)

        try:
            data = response.json()
        except Exception as exc:
            raise LLMError(f"Failed to decode JSON response from {url}: {exc}") from exc

        if not isinstance(data, dict):
            raise LLMError(f"Expected JSON object response, got {type(data).__name__}")

        return data

    def is_healthy(self) -> bool:
        """Check if llama-server instance is reachable and responding.

        Returns:
            True if /health returns HTTP 200, False otherwise.
        """
        url = f"{self.base_url}/health"
        try:
            response = self._client.get(url, timeout=self.timeout)
            return response.status_code == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        """Generate text completion from a prompt via /v1/completions.

        Args:
            prompt: Text prompt input to LLM.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            stop: Optional list of stop sequences.
            reasoning_effort: Optional per-request reasoning effort. The
                llama-server endpoint has no such knob, so it is ignored.

        Returns:
            The generated text string.

        Raises:
            LLMError: If choices list is empty or response is malformed.
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.model:
            payload["model"] = self.model
        if stop is not None:
            payload["stop"] = stop

        if self.on_token is not None:
            return self._generate_streaming(payload)

        data = self._post("/v1/completions", payload)
        choices = data.get("choices")
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            raise LLMError("Malformed LLM response: missing or empty choices")

        choice = choices[0]
        if not isinstance(choice, dict) or "text" not in choice:
            raise LLMError("Malformed LLM response: choice missing 'text' field")

        return str(choice["text"])

    def _generate_streaming(self, payload: dict[str, Any]) -> str:
        """Stream SSE completion chunks from /v1/completions and invoke self.on_token."""
        url = f"{self.base_url}/v1/completions"
        stream_payload = {**payload, "stream": True}
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        callback = self.on_token

        try:
            with self._client.stream(
                "POST",
                url,
                json=stream_payload,
                headers=self._headers,
                timeout=self.timeout,
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    _raise_for_http_error(url, response)

                for raw_line in response.iter_lines():
                    line = raw_line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices") if isinstance(chunk, dict) else None
                    if not choices or not isinstance(choices, list):
                        continue
                    choice = choices[0]
                    if not isinstance(choice, dict):
                        continue

                    delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
                    reasoning = choice.get("reasoning_content") or delta.get("reasoning_content")
                    if isinstance(reasoning, str) and reasoning:
                        reasoning_parts.append(reasoning)
                        if callback is not None:
                            callback("reasoning", reasoning)

                    text = choice.get("text")
                    if text is None:
                        text = delta.get("content")
                    if isinstance(text, str) and text:
                        content_parts.append(text)
                        if callback is not None:
                            callback("content", text)
        except LLMError:
            raise
        except httpx.HTTPError as exc:
            raise _map_httpx_error(url, exc) from exc
        except Exception as exc:
            raise LLMError(f"Failed to parse streaming response from {url}: {exc}") from exc

        if content_parts:
            return "".join(content_parts)
        return "".join(reasoning_parts)

    def get_logprobs(self, prompt: str, target: str) -> float:
        """Calculate total log-probability of target string given prompt.

        Args:
            prompt: Context or prefix prompt.
            target: Target continuation string.

        Returns:
            Cumulative log probability sum.

        Raises:
            LLMError: If response is missing logprob data.
        """
        payload: dict[str, Any] = {
            "prompt": f"{prompt}{target}",
            "max_tokens": 0,
            "temperature": 0.0,
            "logprobs": 1,
            "echo": True,
        }
        if self.model:
            payload["model"] = self.model

        data = self._post("/v1/completions", payload)
        choices = data.get("choices")
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            raise LLMError("Malformed LLM response: missing or empty choices")

        choice = choices[0]
        if not isinstance(choice, dict):
            raise LLMError("Malformed LLM response: choice is not a dict")

        logprobs_data = choice.get("logprobs")
        if not isinstance(logprobs_data, dict):
            return 0.0
        token_logprobs = logprobs_data.get("token_logprobs")
        offsets = logprobs_data.get("text_offset")
        if (
            not isinstance(token_logprobs, list)
            or not isinstance(offsets, list)
            or len(token_logprobs) != len(offsets)
        ):
            return 0.0
        target_logprobs = [
            float(lp)
            for offset, lp in zip(offsets, token_logprobs)
            if offset is not None and int(offset) >= len(prompt) and lp is not None
        ]
        return float(sum(target_logprobs)) if target_logprobs else 0.0

    def get_token_probability(self, prompt: str, token: str) -> float:
        """Return one candidate probability using the shared multi-token request."""
        return self.get_token_probabilities(prompt, [token]).get(token, 0.0)

    def get_token_probabilities(self, prompt: str, tokens: list[str]) -> dict[str, float]:
        """Return candidate probabilities from a single completion request."""
        probabilities, _ = self.get_token_probabilities_and_response(prompt, tokens)
        return probabilities

    def get_token_probabilities_and_response(
        self,
        prompt: str,
        tokens: list[str],
    ) -> tuple[dict[str, float], str | None]:
        """Extract probabilities for candidate tokens from one top_logprobs response.

        Args:
            prompt: Context or prefix prompt.
            tokens: Candidate token strings (e.g. 'Yes' and 'No').

        Returns:
            Extracted probability converted via exp and clamped to [0.0, 1.0].
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 10,
        }
        if self.model:
            payload["model"] = self.model

        data = self._post("/v1/completions", payload)
        choices = data.get("choices")
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            raise LLMError("Malformed LLM response: missing or empty choices")

        choice = choices[0]
        if not isinstance(choice, dict):
            raise LLMError("Malformed LLM response: choice is not a dict")

        response_text = choice.get("text")
        response_text = str(response_text) if response_text is not None else None
        logprobs_data = choice.get("logprobs")
        top_logprobs_list = (
            logprobs_data.get("top_logprobs")
            if isinstance(logprobs_data, dict)
            else None
        )
        first_top_logprobs = (
            top_logprobs_list[0]
            if isinstance(top_logprobs_list, list) and top_logprobs_list
            else None
        )
        if not isinstance(first_top_logprobs, dict):
            return self._answer_probabilities_or_zero(response_text, tokens), response_text

        probabilities: dict[str, float] = {}
        matched_any = False
        for token in tokens:
            target_lp: float | None = None
            if token in first_top_logprobs:
                target_lp = first_top_logprobs[token]
            elif f" {token}" in first_top_logprobs:
                target_lp = first_top_logprobs[f" {token}"]
            else:
                token_stripped = token.strip()
                for candidate, logprob in first_top_logprobs.items():
                    if candidate.strip() == token_stripped:
                        target_lp = logprob
                        break
            matched_any = matched_any or target_lp is not None
            probabilities[token] = (
                float(max(0.0, min(1.0, math.exp(target_lp))))
                if target_lp is not None
                else 0.0
            )
        if not matched_any:
            return self._answer_probabilities_or_zero(response_text, tokens), response_text
        return probabilities, response_text

    @staticmethod
    def _answer_probabilities_or_zero(
        response: str | None,
        tokens: list[str],
    ) -> dict[str, float]:
        if response:
            match = re.match(r"\s*([a-z]+)\b", response, flags=re.IGNORECASE)
            if match is not None:
                answer = match.group(1).casefold()
                if answer in {token.casefold() for token in tokens}:
                    return {token: float(token.casefold() == answer) for token in tokens}
        return {token: 0.0 for token in tokens}

    def start_server(self) -> None:
        """Launch local llama-server subprocess if server_command was specified."""
        if self._server_command and self._server_process is None:
            self._server_process = subprocess.Popen(
                self._server_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def stop_server(self) -> None:
        """Terminate auto-launched llama-server subprocess if running."""
        if self._server_process is not None:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
            self._server_process = None

    def close(self) -> None:
        """Close underlying HTTP client session and terminate any auto-launched server."""
        self._client.close()
        self.stop_server()

    def __enter__(self) -> LlamaServerClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
