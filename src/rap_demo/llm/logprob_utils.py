"""Helpers to read log probabilities of generated text out of provider responses.

Chat providers report the logprob of every token they generate. That is enough
to recover the log probability of a specific action string, as long as the
string can be located inside the generated token stream.
"""

from __future__ import annotations

import re
from collections.abc import Sequence


def normalize_for_match(text: str) -> str:
    """Normalize text so formatting differences do not break token matching."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def sum_token_logprobs(
    tokens: Sequence[tuple[str, float]],
    target: str,
) -> float | None:
    """Sum the logprob of the generated tokens that spell out ``target``.

    Args:
        tokens: Generated tokens with their logprob, in generation order.
        target: Text to locate inside the generated continuation.

    Returns:
        The summed logprob of the matching tokens, or None when the target does
        not appear as a contiguous run of generated tokens.
    """
    normalized_target = normalize_for_match(target)
    if not normalized_target:
        return None

    # Search from the end: a reasoning transcript may mention the same action
    # before the model states it as its final answer.
    for start in range(len(tokens) - 1, -1, -1):
        accumulated = ""
        for end in range(start, len(tokens)):
            accumulated += tokens[end][0]
            normalized = normalize_for_match(accumulated)
            if normalized == normalized_target:
                return float(sum(logprob for _, logprob in tokens[start : end + 1]))
            if len(normalized) > len(normalized_target):
                break
    return None
