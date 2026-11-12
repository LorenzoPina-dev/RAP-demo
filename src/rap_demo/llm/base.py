"""Base LLM interface definition and exceptions for RAP reasoning."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


class LLMError(Exception):
    """Base exception for all LLM backend errors."""


class LLMConnectionError(LLMError):
    """Raised when LLM backend server cannot be reached."""


class LLMTimeoutError(LLMError):
    """Raised when LLM backend request exceeds configured timeout."""


@runtime_checkable
class ContinuationScorer(Protocol):
    """Optional capability: score several continuations in one batch.

    The RAP paper defines R1 as the log probability of the action continuation
    given the current state. Backends that can report token-level logprobs
    implement this protocol so the whole action space can be scored with one
    teacher-forcing pass per token instead of one request per action.
    """

    def score_continuations(
        self,
        prompt: str,
        targets: Sequence[str],
        *,
        assistant_prefix: str = "",
    ) -> dict[str, float] | None:
        """Score every continuation of ``prompt``.

        Args:
            prompt: The prompt the continuations are scored against.
            targets: Candidate continuations to score.
            assistant_prefix: Context the continuations are appended to inside
                the assistant turn. It is not part of the scored text.

        Returns:
            Mapping of continuation to its summed log probability, or None when
            the backend cannot score arbitrary continuations.
        """
        ...


@runtime_checkable
class BaseLLM(Protocol):
    """Protocol defining the core interface for LLM backends in RAP."""

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        """Generate text completion from a prompt.

        Args:
            prompt: Text prompt input to LLM.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 for greedy decoding).
            stop: Optional list of stop strings where generation halts.

        Returns:
            The generated text string.
        """
        ...

    def get_logprobs(self, prompt: str, target: str) -> float:
        """Calculate total log-probability of target string given prompt.

        Used in RAP for scoring actions and transitions (log P(target | prompt)).

        Args:
            prompt: Context or prefix prompt.
            target: Target continuation string.

        Returns:
            Cumulative log probability (typically <= 0.0).
        """
        ...

    def get_token_probability(self, prompt: str, token: str) -> float:
        """Extract the probability (0.0 to 1.0) of a single candidate token following prompt.

        Used in RAP for self-evaluation (e.g., P("Yes" | prompt) vs P("No" | prompt)).

        Args:
            prompt: Context or prefix prompt.
            token: Candidate token string (e.g. "Yes" or "No").

        Returns:
            Normalized probability between 0.0 and 1.0.
        """
        ...
