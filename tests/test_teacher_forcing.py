"""Tests for teacher-forced scoring of candidate continuations.

The RAP paper defines R1 as the log probability of the action continuation
given the state. These tests cover the generic token-walk used to recover that
probability from providers that only report the distribution of the *next*
token.
"""

from __future__ import annotations

from typing import Any

import pytest

from rap_demo.llm.teacher_forcing import OllamaChatScorer, score_continuations


class TestScoreContinuations:
    """Behaviour of the generic teacher-forcing walker."""

    def test_scores_a_single_candidate(self) -> None:
        """The summed logprob of the matched tokens is returned."""
        alternatives = {
            "": [("pick", -1.0)],
            "pick": [(" up", -2.0)],
            "pick up": [(" the red block", -3.0)],
        }

        def request(prefix: str) -> list[tuple[str, float]]:
            return alternatives.get(prefix, [])

        scores = score_continuations(["pick up the red block"], request)

        assert scores == {"pick up the red block": pytest.approx(-6.0)}

    def test_the_seed_is_not_part_of_the_scored_text(self) -> None:
        """The context the continuation is appended to is not scored."""
        alternatives = {
            "[PLAN]\n": [("pick", -1.0)],
            "[PLAN]\npick": [(" up", -2.0)],
        }

        def request(prefix: str) -> list[tuple[str, float]]:
            return alternatives.get(prefix, [])

        scores = score_continuations(
            ["pick up"], request, seed="[PLAN]\n"
        )

        assert scores == {"pick up": pytest.approx(-3.0)}

    def test_shares_prefixes_between_candidates(self) -> None:
        """Candidates sharing a prefix are scored with one request per prefix."""
        calls: list[str] = []

        def request(prefix: str) -> list[tuple[str, float]]:
            calls.append(prefix)
            table = {
                "": [("pick", -1.0)],
                "pick": [(" up", -0.5)],
                "pick up": [(" the", -0.25)],
                "pick up the": [(" red", -0.75), (" blue", -1.75)],
                "pick up the red": [(" block", -0.1)],
                "pick up the blue": [(" block", -0.2)],
            }
            return table.get(prefix, [])

        scores = score_continuations(
            ["pick up the red block", "pick up the blue block"], request
        )

        assert scores["pick up the red block"] == pytest.approx(-2.6)
        assert scores["pick up the blue block"] == pytest.approx(-3.7)
        # The shared prefix "pick up the" is requested only once.
        assert calls.count("pick up the") == 1

    def test_falls_back_to_least_likely_alternative(self) -> None:
        """A candidate outside the reported alternatives is scored inexactly."""

        def request(prefix: str) -> list[tuple[str, float]]:
            if prefix == "":
                return [("stack", -0.5), ("pick", -3.0)]
            return []

        scores = score_continuations(["pick up the red block"], request)

        assert scores["pick up the red block"] == pytest.approx(-3.0)

    def test_terminates_when_the_candidate_never_matches(self) -> None:
        """A backend that never returns a matching token does not hang."""

        def request(prefix: str) -> list[tuple[str, float]]:
            return [("zzz", -1.0)]

        scores = score_continuations(["pick up the red block"], request)

        assert scores["pick up the red block"] < 0.0

    def test_ignores_empty_candidates(self) -> None:
        """Empty continuations are not sent to the backend."""

        def request(prefix: str) -> list[tuple[str, float]]:
            raise AssertionError("the backend must not be called")

        scores = score_continuations([""], request)

        assert scores == {"": 0.0}

    def test_requests_are_issued_in_parallel(self) -> None:
        """Distinct prefixes are scored concurrently."""
        seen: list[str] = []

        def request(prefix: str) -> list[tuple[str, float]]:
            seen.append(prefix)
            return [("x", -1.0)]

        score_continuations(["xa", "xb"], request, max_workers=4)

        assert set(seen) == {"", "x"}


class TestOllamaChatScorer:
    """Behaviour of the Ollama chat scorer."""

    def test_from_model_name_ignores_other_providers(self) -> None:
        assert OllamaChatScorer.from_model_name("gpt-4o-mini") is None
        assert OllamaChatScorer.from_model_name("openai/gpt-4o-mini") is None

    def test_from_model_name_keeps_the_ollama_prefix_out_of_the_model(self) -> None:
        scorer = OllamaChatScorer.from_model_name("ollama/qwen3.5:9b")

        assert scorer is not None
        assert scorer.model == "qwen3.5:9b"
        assert scorer.base_url == "http://localhost:11434"

    def test_from_model_name_strips_the_openai_compatible_suffix(self) -> None:
        scorer = OllamaChatScorer.from_model_name(
            "ollama/gpt-oss:20b", base_url="http://localhost:11434/v1"
        )

        assert scorer is not None
        assert scorer.base_url == "http://localhost:11434"

    def test_next_token_alternatives_reads_the_chat_response(self) -> None:
        scorer = OllamaChatScorer(base_url="http://localhost:11434", model="m")
        captured: dict[str, Any] = {}

        def fake_post(payload: dict[str, Any]) -> dict[str, Any]:
            captured.update(payload)
            return {
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {
                                    "token": " pick",
                                    "logprob": -0.5,
                                    "top_logprobs": [
                                        {"token": " pick", "logprob": -0.5},
                                        {"token": " stack", "logprob": -1.5},
                                    ],
                                }
                            ]
                        }
                    }
                ]
            }

        scorer._post = fake_post  # type: ignore[method-assign]

        alternatives = scorer.next_token_alternatives("prompt", "[PLAN]\n")

        assert alternatives == [(" pick", -0.5), (" stack", -1.5)]
        assert captured["messages"] == [
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": "[PLAN]\n"},
        ]
        assert captured["logprobs"] is True
        assert captured["max_tokens"] == 1

    def test_next_token_alternatives_returns_empty_without_logprobs(self) -> None:
        scorer = OllamaChatScorer(base_url="http://localhost:11434", model="m")
        scorer._post = lambda payload: {"choices": []}  # type: ignore[method-assign]

        assert scorer.next_token_alternatives("prompt", "[PLAN]\n") == []

    def test_score_forces_the_continuation_into_the_assistant_turn(self) -> None:
        """Each request carries the candidate prefix in the assistant turn."""
        seen: list[str] = []

        def fake_post(payload: dict[str, Any]) -> dict[str, Any]:
            seen.append(payload["messages"][1]["content"])
            token = {"[PLAN]\n": "pick", "pick": " up"}.get(
                payload["messages"][1]["content"], " up"
            )
            return {
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {
                                    "token": token,
                                    "logprob": -1.0,
                                    "top_logprobs": [{"token": token, "logprob": -1.0}],
                                }
                            ]
                        }
                    }
                ]
            }

        scorer = OllamaChatScorer(base_url="http://localhost:11434", model="m")
        scorer._post = fake_post  # type: ignore[method-assign]

        scores = scorer.score("prompt", ["pick up"], assistant_prefix="[PLAN]\n")

        assert scores["pick up"] == pytest.approx(-2.0)
        assert seen[0] == "[PLAN]\n"
