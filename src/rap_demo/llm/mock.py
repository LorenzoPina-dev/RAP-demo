"""Mock LLM backend for deterministic testing and simulated reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from collections.abc import Sequence
from typing import Any, Pattern
from rap_demo.llm.base import BaseLLM, LLMError


@dataclass
class LLMCall:
    """Record of an invocation of MockLLM."""

    method: str
    prompt: str
    args: dict[str, Any] = field(default_factory=dict)


class MockLLM(BaseLLM):
    """Mock LLM implementation with configurable canned responses, probabilities, and call tracing."""

    def __init__(
        self,
        default_response: str = "",
        responses: dict[str, str] | None = None,
        response_rules: list[tuple[str | Pattern[str], str]] | None = None,
        logprobs: dict[tuple[str, str], float] | dict[str, float] | None = None,
        token_probs: dict[tuple[str, str], float] | dict[str, float] | None = None,
        default_token_prob: float = 0.5,
        default_logprob: float = 0.0,
        raise_on_unmatched: bool = False,
    ) -> None:
        """Initialize MockLLM with deterministic canned outputs and probabilities.

        Args:
            default_response: Fallback text generation when no rules/exact match trigger.
            responses: Exact prompt-to-response dictionary mapping.
            response_rules: List of (prefix/pattern, response) rules matched in order.
            logprobs: Mapping of (prompt, target) or target to log-probability float.
            token_probs: Mapping of (prompt, token) or token to probability float.
            default_token_prob: Default token probability if not in mapping.
            default_logprob: Default logprob if not in mapping.
            raise_on_unmatched: If True, raise LLMError when generation cannot match prompt.
        """
        self.default_response = default_response
        self.responses: dict[str, str] = dict(responses) if responses is not None else {}
        self.response_rules: list[tuple[str | Pattern[str], str]] = (
            list(response_rules) if response_rules is not None else []
        )
        self.logprobs: dict[tuple[str, str], float] | dict[str, float] = (
            dict(logprobs) if logprobs is not None else {}
        )
        self.token_probs: dict[tuple[str, str], float] | dict[str, float] = (
            dict(token_probs) if token_probs is not None else {}
        )
        self.default_token_prob = default_token_prob
        self.default_logprob = default_logprob
        self.raise_on_unmatched = raise_on_unmatched

        self._calls: list[LLMCall] = []

    @property
    def calls(self) -> list[LLMCall]:
        """List of all calls made to this mock instance."""
        return list(self._calls)

    @property
    def call_count(self) -> int:
        """Total number of method calls recorded."""
        return len(self._calls)

    def reset(self) -> None:
        """Clear recorded call history."""
        self._calls.clear()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        """Generate deterministic text matching prompt rules.

        Args:
            prompt: Text prompt input to LLM.
            max_tokens: Maximum tokens to generate (recorded in call trace).
            temperature: Sampling temperature (recorded in call trace).
            stop: Optional list of stop strings where generation halts.

        Returns:
            Matched text response truncated at earliest stop token.

        Raises:
            LLMError: If prompt is unmatched and raise_on_unmatched is True.
        """
        self._calls.append(
            LLMCall(
                method="generate",
                prompt=prompt,
                args={
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": stop,
                },
            )
        )

        text: str | None = None

        # 1. Exact match
        if prompt in self.responses:
            text = self.responses[prompt]

        # 2. Rule matching
        if text is None:
            for pattern, response in self.response_rules:
                if isinstance(pattern, re.Pattern):
                    if pattern.search(prompt):
                        text = response
                        break
                elif isinstance(pattern, str):
                    if prompt.startswith(pattern):
                        text = response
                        break
                    try:
                        if re.search(pattern, prompt):
                            text = response
                            break
                    except re.error:
                        pass

        # 3. Fallback or raise
        if text is None:
            if self.raise_on_unmatched:
                raise LLMError(f"No response matching prompt: {prompt!r}")
            text = self.default_response

        # 4. Stop token truncation
        if stop:
            earliest_idx: int | None = None
            for s in stop:
                if s and s in text:
                    idx = text.find(s)
                    if earliest_idx is None or idx < earliest_idx:
                        earliest_idx = idx
            if earliest_idx is not None:
                text = text[:earliest_idx]

        return text

    def get_logprobs(self, prompt: str, target: str) -> float:
        """Return configured log probability for (prompt, target) or target.

        Args:
            prompt: Context or prefix prompt.
            target: Target continuation string.

        Returns:
            Configured logprob or default_logprob.
        """
        self._calls.append(
            LLMCall(
                method="get_logprobs",
                prompt=prompt,
                args={"target": target},
            )
        )

        if (prompt, target) in self.logprobs:
            return float(self.logprobs[(prompt, target)])
        if target in self.logprobs:
            return float(self.logprobs[target])
        return float(self.default_logprob)

    def score_continuations(
        self,
        prompt: str,
        targets: Sequence[str],
        *,
        assistant_prefix: str = "",
    ) -> dict[str, float]:
        """Return configured log probabilities for every continuation.

        Args:
            prompt: The prompt the continuations are scored against.
            targets: Candidate continuations.
            assistant_prefix: Context the continuations are appended to inside
                the assistant turn. It is not part of the scored text.

        Returns:
            Mapping of continuation to its configured log probability.
        """
        self._calls.append(
            LLMCall(
                method="score_continuations",
                prompt=prompt,
                args={
                    "targets": list(targets),
                    "assistant_prefix": assistant_prefix,
                },
            )
        )
        scores: dict[str, float] = {}
        for target in targets:
            if (prompt, target) in self.logprobs:
                scores[target] = float(self.logprobs[(prompt, target)])
            elif target in self.logprobs:
                scores[target] = float(self.logprobs[target])
            else:
                scores[target] = float(self.default_logprob)
        return scores

    def get_token_probability(self, prompt: str, token: str) -> float:
        """Return configured probability in [0.0, 1.0] for (prompt, token) or token.

        Args:
            prompt: Context or prefix prompt.
            token: Candidate token string (e.g. 'Yes' or 'No').

        Returns:
            Configured probability clamped between 0.0 and 1.0.
        """
        self._calls.append(
            LLMCall(
                method="get_token_probability",
                prompt=prompt,
                args={"token": token},
            )
        )

        prob: float
        if (prompt, token) in self.token_probs:
            prob = float(self.token_probs[(prompt, token)])
        elif token in self.token_probs:
            prob = float(self.token_probs[token])
        else:
            prob = float(self.default_token_prob)

        return float(max(0.0, min(1.0, prob)))

    def get_token_probabilities(self, prompt: str, tokens: list[str]) -> dict[str, float]:
        """Return configured token probabilities with one recorded mock call."""
        self._calls.append(
            LLMCall(
                method="get_token_probabilities",
                prompt=prompt,
                args={"tokens": list(tokens)},
            )
        )
        probabilities: dict[str, float] = {}
        for token in tokens:
            if (prompt, token) in self.token_probs:
                prob = float(self.token_probs[(prompt, token)])
            elif token in self.token_probs:
                prob = float(self.token_probs[token])
            else:
                prob = float(self.default_token_prob)
            probabilities[token] = float(max(0.0, min(1.0, prob)))
        return probabilities
