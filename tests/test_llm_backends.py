"""Comprehensive unit and integration tests for Phase 3: LLM Backends & Local Model Integration.

Tests cover:
1. BaseLLM Protocol:
   - Protocol runtime checkability and contract verification.
   - Exception hierarchy (LLMError, LLMConnectionError, LLMTimeoutError).
2. MockLLM:
   - Configurable canned responses (exact match, prefix match, regex match, fallback).
   - Configurable logprobs for target strings and token probabilities.
   - Call history tracking (number of calls, received prompts, arguments, reset).
   - Edge cases: unmatched prompts, empty strings, unicode, stop sequences.
3. LlamaServerClient:
   - Client connecting to llama-server OpenAI-compatible API (/v1/completions, /health).
   - Mock transport for text generation with stop tokens and temperature.
   - Extraction of log probabilities from token_logprobs.
   - Extraction of token probability from top_logprobs distribution (Yes/No evaluation).
   - Connection failure, timeout, HTTP error, and malformed response handling.
   - Server health checking.
4. LLM Factory (get_llm):
   - Creation of MockLLM and LlamaServerClient.
   - Backend alias resolution and case-insensitivity.
   - Keyword argument forwarding.
   - Validation and informative error messages for unknown backends.
"""

from __future__ import annotations

import json
import math
import re
from types import SimpleNamespace
from typing import Any
import httpx
import pytest

from rap_demo.llm.base import (
    BaseLLM,
    LLMError,
    LLMConnectionError,
    LLMTimeoutError,
)
from rap_demo.llm.mock import MockLLM, LLMCall
from rap_demo.llm.llama_client import LlamaServerClient
from rap_demo.llm.litellm_client import LiteLLMClient
from rap_demo.llm.factory import get_llm


# ============================================================================
# 1. BaseLLM Protocol & Exception Hierarchy Tests
# ============================================================================


class TestBaseLLMProtocol:
    """Verifies the BaseLLM Protocol contract and exception classifications."""

    def test_protocol_runtime_checkable_with_valid_implementation(self) -> None:
        """A class implementing generate, get_logprobs, and get_token_probability satisfies BaseLLM."""

        class ValidLLM:
            def generate(
                self,
                prompt: str,
                max_tokens: int = 128,
                temperature: float = 0.0,
                stop: list[str] | None = None,
            ) -> str:
                return "generated"

            def get_logprobs(self, prompt: str, target: str) -> float:
                return -0.5

            def get_token_probability(self, prompt: str, token: str) -> float:
                return 0.8

        instance = ValidLLM()
        assert isinstance(instance, BaseLLM)

    def test_protocol_rejects_missing_generate(self) -> None:
        """A class missing generate() must not satisfy BaseLLM."""

        class MissingGenerate:
            def get_logprobs(self, prompt: str, target: str) -> float:
                return 0.0

            def get_token_probability(self, prompt: str, token: str) -> float:
                return 0.5

        assert not isinstance(MissingGenerate(), BaseLLM)

    def test_protocol_rejects_missing_get_logprobs(self) -> None:
        """A class missing get_logprobs() must not satisfy BaseLLM."""

        class MissingLogprobs:
            def generate(
                self,
                prompt: str,
                max_tokens: int = 128,
                temperature: float = 0.0,
                stop: list[str] | None = None,
            ) -> str:
                return ""

            def get_token_probability(self, prompt: str, token: str) -> float:
                return 0.5

        assert not isinstance(MissingLogprobs(), BaseLLM)

    def test_protocol_rejects_missing_get_token_probability(self) -> None:
        """A class missing get_token_probability() must not satisfy BaseLLM."""

        class MissingTokenProb:
            def generate(
                self,
                prompt: str,
                max_tokens: int = 128,
                temperature: float = 0.0,
                stop: list[str] | None = None,
            ) -> str:
                return ""

            def get_logprobs(self, prompt: str, target: str) -> float:
                return 0.0

        assert not isinstance(MissingTokenProb(), BaseLLM)

    def test_exception_hierarchy(self) -> None:
        """Verify custom exception inheritance relationships."""
        assert issubclass(LLMError, Exception)
        assert issubclass(LLMConnectionError, LLMError)
        assert issubclass(LLMTimeoutError, LLMError)

        # Catching LLMError catches both connection and timeout subclasses
        conn_err = LLMConnectionError("Failed to reach server")
        timeout_err = LLMTimeoutError("Request timed out after 30s")

        assert isinstance(conn_err, LLMError)
        assert isinstance(timeout_err, LLMError)


# ============================================================================
# 2. MockLLM Unit Tests
# ============================================================================


class TestMockLLM:
    """Verifies deterministic MockLLM capabilities, rule matching, and call tracking."""

    def test_mock_llm_satisfies_protocol(self) -> None:
        """MockLLM must satisfy BaseLLM protocol."""
        mock = MockLLM()
        assert isinstance(mock, BaseLLM)

    def test_generate_exact_match(self) -> None:
        """MockLLM returns canned response for exact prompt match."""
        mock = MockLLM(
            responses={
                "What is 2+2?": "4",
                "Name a color": "blue",
            },
            default_response="unknown",
        )
        assert mock.generate("What is 2+2?") == "4"
        assert mock.generate("Name a color") == "blue"

    def test_generate_default_response_fallback(self) -> None:
        """MockLLM falls back to default_response when prompt is unmatched."""
        mock = MockLLM(
            responses={"ping": "pong"},
            default_response="fallback response",
        )
        assert mock.generate("unregistered prompt") == "fallback response"

    def test_generate_prefix_rule_match(self) -> None:
        """MockLLM matches response rules using prefix strings."""
        mock = MockLLM(
            response_rules=[
                ("Calculate:", "Calculation result: 42"),
                ("Translate to French:", "Bonjour"),
            ],
            default_response="no match",
        )
        assert mock.generate("Calculate: 10 + 32") == "Calculation result: 42"
        assert mock.generate("Translate to French: Hello") == "Bonjour"
        assert mock.generate("Other prompt") == "no match"

    def test_generate_regex_rule_match(self) -> None:
        """MockLLM matches response rules using compiled regex or regex pattern strings."""
        mock = MockLLM(
            response_rules=[
                (re.compile(r"Step \d+:"), "Next step executed"),
                (r"^Is \w+ a fruit\?$", "Yes"),
            ],
            default_response="unmatched",
        )
        assert mock.generate("Step 1: compute initial state") == "Next step executed"
        assert mock.generate("Step 42: finalize trajectory") == "Next step executed"
        assert mock.generate("Is apple a fruit?") == "Yes"
        assert mock.generate("Is apple a fruit? extra") == "unmatched"

    def test_generate_priority_exact_before_rules_before_default(self) -> None:
        """Exact matches take precedence over rule matches, which take precedence over default."""
        mock = MockLLM(
            responses={"Step 1:": "Exact match for Step 1"},
            response_rules=[(r"Step \d+:", "Rule match for Step")],
            default_response="Default match",
        )
        assert mock.generate("Step 1:") == "Exact match for Step 1"
        assert mock.generate("Step 2:") == "Rule match for Step"
        assert mock.generate("Unrelated") == "Default match"

    def test_generate_raise_on_unmatched(self) -> None:
        """When raise_on_unmatched is True, unmatched prompts raise LLMError or KeyError."""
        mock = MockLLM(
            responses={"known": "result"},
            raise_on_unmatched=True,
        )
        assert mock.generate("known") == "result"
        with pytest.raises((LLMError, KeyError)):
            mock.generate("unknown prompt")

    def test_generate_stop_token_truncation(self) -> None:
        """Stop tokens truncate generated text before or at the stop sequence."""
        mock = MockLLM(
            responses={"prompt": "Line 1\nLine 2\nLine 3"},
        )
        result = mock.generate("prompt", stop=["\n"])
        assert result == "Line 1"

    def test_get_logprobs_exact_prompt_and_target(self) -> None:
        """MockLLM returns configured log probability for (prompt, target) pair."""
        mock = MockLLM(
            logprobs={
                ("Context A", "Action 1"): -0.25,
                ("Context A", "Action 2"): -1.80,
            },
            default_logprob=-0.5,
        )
        assert mock.get_logprobs("Context A", "Action 1") == -0.25
        assert mock.get_logprobs("Context A", "Action 2") == -1.80
        assert mock.get_logprobs("Context B", "Action 1") == -0.5

    def test_get_logprobs_target_only_mapping(self) -> None:
        """MockLLM supports target-only dictionary mapping regardless of prompt."""
        mock = MockLLM(
            logprobs={
                "good_action": -0.1,
                "bad_action": -4.5,
            },
            default_logprob=-1.0,
        )
        assert mock.get_logprobs("Any prompt", "good_action") == -0.1
        assert mock.get_logprobs("Another prompt", "bad_action") == -4.5
        assert mock.get_logprobs("Any prompt", "unknown_action") == -1.0

    def test_get_token_probability_pair_and_token_only(self) -> None:
        """MockLLM returns configured token probability for self-eval questions."""
        mock = MockLLM(
            token_probs={
                ("Is this solution valid?", "Yes"): 0.85,
                ("Is this solution valid?", "No"): 0.15,
                "True": 0.95,
                "False": 0.05,
            },
            default_token_prob=0.5,
        )
        assert mock.get_token_probability("Is this solution valid?", "Yes") == 0.85
        assert mock.get_token_probability("Is this solution valid?", "No") == 0.15
        assert mock.get_token_probability("Any prompt", "True") == 0.95
        assert mock.get_token_probability("Any prompt", "False") == 0.05
        assert mock.get_token_probability("Unmatched prompt", "Unknown") == 0.5

    def test_get_token_probability_bounds(self) -> None:
        """Token probabilities returned by MockLLM must be valid probabilities in [0.0, 1.0]."""
        mock = MockLLM(
            token_probs={"Yes": 0.9, "No": 0.1},
            default_token_prob=0.5,
        )
        prob_yes = mock.get_token_probability("prompt", "Yes")
        prob_no = mock.get_token_probability("prompt", "No")
        prob_default = mock.get_token_probability("prompt", "Maybe")

        for p in (prob_yes, prob_no, prob_default):
            assert 0.0 <= p <= 1.0

    def test_call_history_tracking(self) -> None:
        """MockLLM records all calls to generate, get_logprobs, and get_token_probability."""
        mock = MockLLM(default_response="ok", default_token_prob=0.7, default_logprob=-0.3)

        assert mock.call_count == 0
        assert mock.calls == []

        mock.generate("Prompt 1", max_tokens=64, temperature=0.7, stop=["\n"])
        mock.get_logprobs("Prompt 2", target="action_1")
        mock.get_token_probability("Prompt 3", token="Yes")

        assert mock.call_count == 3
        assert len(mock.calls) == 3

        call_0: LLMCall = mock.calls[0]
        assert call_0.method == "generate"
        assert call_0.prompt == "Prompt 1"
        assert call_0.args.get("max_tokens") == 64
        assert call_0.args.get("temperature") == 0.7
        assert call_0.args.get("stop") == ["\n"]

        call_1: LLMCall = mock.calls[1]
        assert call_1.method == "get_logprobs"
        assert call_1.prompt == "Prompt 2"
        assert call_1.args.get("target") == "action_1"

        call_2: LLMCall = mock.calls[2]
        assert call_2.method == "get_token_probability"
        assert call_2.prompt == "Prompt 3"
        assert call_2.args.get("token") == "Yes"

    def test_call_history_reset(self) -> None:
        """reset() clears all recorded calls."""
        mock = MockLLM(default_response="ok")
        mock.generate("test prompt")
        assert mock.call_count == 1

        mock.reset()
        assert mock.call_count == 0
        assert mock.calls == []

    @pytest.mark.parametrize(
        "edge_prompt",
        [
            "",  # Empty prompt
            "   \n\t  ",  # Whitespace only
            r"Special characters: !@#$%^&*()_+{}[]|\:;\"'<>,.?/~`",
            "Unicode: 𝒳 = ∑ 𝛼_i · β_j 🚀 ñ á é ö 中文",
            "A" * 5000,  # Long prompt
        ],
    )
    def test_mock_llm_handles_edge_case_prompts(self, edge_prompt: str) -> None:
        """MockLLM safely handles empty strings, long prompts, and unicode characters."""
        mock = MockLLM(default_response="handled")
        res = mock.generate(edge_prompt)
        assert res == "handled"
        assert mock.call_count == 1
        assert mock.calls[0].prompt == edge_prompt


# ============================================================================
# 3. LlamaServerClient Tests (Mocking HTTP Transport)
# ============================================================================


class TestLlamaServerClient:
    """Verifies LlamaServerClient communication with llama-server OpenAI API using mock HTTP transport."""

    def test_client_satisfies_protocol(self) -> None:
        """LlamaServerClient must satisfy BaseLLM protocol."""
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"status": "ok"}))
        client = LlamaServerClient(base_url="http://localhost:8080", client=httpx.Client(transport=transport))
        assert isinstance(client, BaseLLM)

    def test_default_initialization(self) -> None:
        """Client initializes with default base_url and timeout."""
        client = LlamaServerClient()
        assert client.base_url == "http://localhost:8080"
        assert client.timeout == 30.0

    def test_health_check_healthy(self) -> None:
        """is_healthy() returns True when server responds 200 with status ok."""

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/health"
            assert request.method == "GET"
            return httpx.Response(200, json={"status": "ok"})

        client = LlamaServerClient(
            base_url="http://testserver:8080",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        assert client.is_healthy() is True

    def test_health_check_unhealthy_status_code(self) -> None:
        """is_healthy() returns False when server responds with 503 loading model."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={"status": "loading model"})

        client = LlamaServerClient(
            base_url="http://testserver:8080",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        assert client.is_healthy() is False

    def test_health_check_connection_error(self) -> None:
        """is_healthy() returns False gracefully on connection failure instead of raising."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Failed to establish connection", request=request)

        client = LlamaServerClient(
            base_url="http://down-server:8080",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        assert client.is_healthy() is False

    def test_generate_successful_completion(self) -> None:
        """generate() sends proper payload to /v1/completions and parses choices[0].text."""
        captured_request: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/completions"
            assert request.method == "POST"
            captured_request["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "id": "cmpl-test",
                    "choices": [
                        {
                            "text": "Generated plan step 1",
                            "index": 0,
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        client = LlamaServerClient(
            base_url="http://testserver:8080",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        result = client.generate(
            prompt="Given initial state A, find action:",
            max_tokens=64,
            temperature=0.2,
            stop=["\n", "###"],
        )

        assert result == "Generated plan step 1"
        payload = captured_request["body"]
        assert payload["prompt"] == "Given initial state A, find action:"
        assert payload["max_tokens"] == 64
        assert math.isclose(payload["temperature"], 0.2)
        assert payload["stop"] == ["\n", "###"]

    def test_generate_default_arguments(self) -> None:
        """generate() uses max_tokens=128 and temperature=0.0 by default."""
        captured_payload: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_payload.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={"choices": [{"text": "default output"}]},
            )

        client = LlamaServerClient(
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        client.generate("test prompt")

        assert captured_payload["max_tokens"] == 128
        assert captured_payload["temperature"] == 0.0
        assert captured_payload.get("stop") is None

    def test_get_logprobs_extraction_from_token_logprobs(self) -> None:
        """get_logprobs() computes cumulative log probability of target string tokens."""
        captured_payload: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_payload.update(json.loads(request.content))
            # Mock llama-server response with logprobs
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "text": " move right",
                            "logprobs": {
                                "tokens": [" move", " right"],
                                "token_logprobs": [-0.15, -0.35],
                                "text_offset": [
                                    len("State: at (0, 0)"),
                                    len("State: at (0, 0)") + 5,
                                ],
                            },
                        }
                    ],
                },
            )

        client = LlamaServerClient(
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        logprob = client.get_logprobs(prompt="State: at (0, 0)", target="move right")

        # Sum of token logprobs: -0.15 + -0.35 = -0.50
        assert math.isclose(logprob, -0.50, rel_tol=1e-5)
        assert "logprobs" in captured_payload
        assert captured_payload["logprobs"] >= 1

    def test_get_token_probability_extraction_from_top_logprobs(self) -> None:
        """get_token_probability() extracts logprob from top_logprobs and converts to probability."""

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            assert payload["max_tokens"] == 1
            assert payload["logprobs"] >= 1
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "text": "Yes",
                            "logprobs": {
                                "tokens": ["Yes"],
                                "token_logprobs": [-0.22314355],  # ln(0.8)
                                "top_logprobs": [
                                    {
                                        "Yes": -0.22314355,   # p = 0.80
                                        " No": -1.60943791,   # p = 0.20
                                        " Maybe": -4.605170,  # p = 0.01
                                    }
                                ],
                            },
                        }
                    ],
                },
            )

        client = LlamaServerClient(
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        # "Yes" probability: exp(-0.22314355) ≈ 0.80
        prob_yes = client.get_token_probability("Is this step correct?", "Yes")
        assert math.isclose(prob_yes, 0.80, rel_tol=1e-3)

        # "No" probability (with leading space handled): exp(-1.60943791) ≈ 0.20
        prob_no = client.get_token_probability("Is this step correct?", "No")
        assert math.isclose(prob_no, 0.20, rel_tol=1e-3)

    def test_get_token_probabilities_scores_yes_and_no_in_one_request(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "logprobs": {
                                "top_logprobs": [
                                    {" Yes": math.log(0.8), " No": math.log(0.2)}
                                ]
                            }
                        }
                    ]
                },
            )

        client = LlamaServerClient(
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        probabilities = client.get_token_probabilities("Is this promising?", ["Yes", "No"])

        assert probabilities["Yes"] == pytest.approx(0.8)
        assert probabilities["No"] == pytest.approx(0.2)
        assert calls == 1

    def test_get_token_probability_token_not_in_top_logprobs(self) -> None:
        """When candidate token is not in top_logprobs distribution, returns 0.0."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "text": "Alpha",
                            "logprobs": {
                                "tokens": ["Alpha"],
                                "token_logprobs": [-0.01],
                                "top_logprobs": [{"Alpha": -0.01, "Beta": -4.6}],
                            },
                        }
                    ],
                },
            )

        client = LlamaServerClient(
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        prob_missing = client.get_token_probability("Prompt", "Gamma")
        assert prob_missing == 0.0

    def test_network_connection_error_raises_llm_connection_error(self) -> None:
        """When connection fails, client raises LLMConnectionError with clear information."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused to 127.0.0.1:8080", request=request)

        client = LlamaServerClient(
            base_url="http://127.0.0.1:8080",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        with pytest.raises(LLMConnectionError) as exc_info:
            client.generate("test prompt")
        assert "127.0.0.1:8080" in str(exc_info.value) or "connection" in str(exc_info.value).lower()

    def test_network_timeout_raises_llm_timeout_error(self) -> None:
        """When request times out, client raises LLMTimeoutError."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("Request timed out", request=request)

        client = LlamaServerClient(
            timeout=5.0,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        with pytest.raises(LLMTimeoutError):
            client.generate("long running prompt")

    def test_http_error_response_raises_llm_error(self) -> None:
        """When server returns 4xx or 5xx, client raises LLMError with status code."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": {"message": "Internal CUDA memory error"}})

        client = LlamaServerClient(
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        with pytest.raises(LLMError) as exc_info:
            client.generate("test prompt")
        assert "500" in str(exc_info.value) or "CUDA" in str(exc_info.value)

    def test_malformed_empty_choices_raises_llm_error(self) -> None:
        """When response JSON has empty choices list, raises LLMError."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": []})

        client = LlamaServerClient(
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        with pytest.raises(LLMError):
            client.generate("prompt")

    def test_context_manager_and_close(self) -> None:
        """LlamaServerClient can be used as context manager and closed cleanly."""
        underlying_client = httpx.Client(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"status": "ok"}))
        )
        with LlamaServerClient(client=underlying_client) as client:
            assert client.is_healthy() is True

        assert underlying_client.is_closed


class TestLiteLLMOllamaLogprobs:
    """Verify native Ollama continuation scoring and paired self-evaluation probabilities."""

    def test_r1_scores_only_echoed_target_and_r3_uses_one_request(self) -> None:
        completion_calls: list[dict[str, Any]] = []
        context = "State: "

        def completion(**kwargs: Any) -> Any:
            completion_calls.append(kwargs)
            if "prompt" in kwargs:
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            logprobs=SimpleNamespace(
                                token_logprobs=[-0.1, -0.2, -0.3],
                                text_offset=[0, len(context), len(context) + 2],
                            )
                        )
                    ]
                )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Yes"),
                        logprobs=SimpleNamespace(
                            content=[
                                SimpleNamespace(
                                    top_logprobs=[
                                        SimpleNamespace(token="Yes", logprob=math.log(0.8)),
                                        SimpleNamespace(token=" No", logprob=math.log(0.2)),
                                    ]
                                )
                            ]
                        ),
                    )
                ]
            )

        client = LiteLLMClient(
            model="ollama/llama3.2",
            api_base="http://localhost:11434",
            completion_fn=completion,
        )

        assert client.get_logprobs(context, "red") == pytest.approx(-0.5)
        probabilities, answer = client.get_token_probabilities_and_response(
            "Is it promising?",
            ["Yes", "No"],
        )

        assert probabilities == pytest.approx({"Yes": 0.8, "No": 0.2})
        assert answer == "Yes"
        assert len(completion_calls) == 2
        assert completion_calls[0]["model"] == "openai/llama3.2"
        assert completion_calls[0]["echo"] is True
        assert completion_calls[0]["max_tokens"] == 0
        assert completion_calls[0]["api_key"] == "ollama"
        assert completion_calls[1]["top_logprobs"] == 10
        assert completion_calls[1]["model"] == "openai/llama3.2"
        assert completion_calls[1]["api_key"] == "ollama"

    def test_r3_uses_generated_answer_when_candidate_logprobs_are_missing(self) -> None:
        client = LiteLLMClient(
            model="ollama/llama3.2",
            completion_fn=lambda **_: SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="No, this action does not help."),
                        logprobs=SimpleNamespace(
                            content=[
                                SimpleNamespace(
                                    top_logprobs=[
                                        SimpleNamespace(token="Maybe", logprob=-0.1)
                                    ]
                                )
                            ]
                        ),
                    )
                ]
            ),
        )

        probabilities, answer = client.get_token_probabilities_and_response(
            "Is it promising?",
            ["Yes", "No"],
        )

        assert probabilities == {"Yes": 0.0, "No": 1.0}
        assert answer == "No, this action does not help."


# ============================================================================
# 4. LLM Factory (get_llm) Tests
# ============================================================================


class TestLLMFactory:
    """Verifies backend instantiation, alias resolution, and configuration via get_llm()."""

    def test_get_llm_mock_default(self) -> None:
        """get_llm() returns MockLLM by default or when backend='mock'."""
        llm_default = get_llm()
        assert isinstance(llm_default, MockLLM)

        llm_mock = get_llm("mock")
        assert isinstance(llm_mock, MockLLM)

    def test_get_llm_mock_with_parameters(self) -> None:
        """get_llm('mock', ...) passes kwargs to MockLLM constructor."""
        llm = get_llm(
            "mock",
            default_response="factory default",
            token_probs={"Yes": 0.95},
        )
        assert isinstance(llm, MockLLM)
        assert llm.default_response == "factory default"
        assert llm.get_token_probability("p", "Yes") == 0.95

    @pytest.mark.parametrize("backend_name", ["llama", "llama_server", "llama-server", "LLAMA_SERVER"])
    def test_get_llm_llama_server_aliases(self, backend_name: str) -> None:
        """get_llm resolves various aliases and casing for llama-server."""
        llm = get_llm(backend_name, base_url="http://custom:9090", timeout=15.0)
        assert isinstance(llm, LlamaServerClient)
        assert llm.base_url == "http://custom:9090"
        assert llm.timeout == 15.0

    def test_get_llm_case_insensitive(self) -> None:
        """get_llm is case-insensitive for all backends."""
        assert isinstance(get_llm("MOCK"), MockLLM)
        assert isinstance(get_llm("Mock"), MockLLM)
        assert isinstance(get_llm("Llama"), LlamaServerClient)

    def test_get_llm_unknown_backend_raises_value_error(self) -> None:
        """get_llm raises ValueError with informative message for unknown backends."""
        with pytest.raises(ValueError) as exc_info:
            get_llm("unsupported_backend_xyz")

        error_msg = str(exc_info.value).lower()
        assert "unsupported_backend_xyz" in error_msg
        assert "mock" in error_msg
        assert "llama" in error_msg

    @pytest.mark.parametrize("invalid_backend", ["", "   ", None])
    def test_get_llm_invalid_backend_values(self, invalid_backend: Any) -> None:
        """get_llm raises ValueError on empty or invalid backend identifiers."""
        with pytest.raises(ValueError):
            get_llm(invalid_backend)


# ============================================================================
# 5. Integration: Simulated RAP Reasoning with LLM Backend
# ============================================================================


class TestLLMReasoningIntegration:
    """Verifies end-to-end simulated interaction pattern between RAP components and LLM backend."""

    def test_simulated_rap_step_generation_and_evaluation(self) -> None:
        """Demonstrates RAP action proposal, transition scoring, and self-evaluation via LLM."""
        llm = MockLLM(
            responses={
                "Propose action for Blocksworld state [on(A, Table), on(B, A)]": "Action: unstack(B, A)",
            },
            logprobs={
                "Action: unstack(B, A)": -0.12,  # High likelihood action proposal
            },
            token_probs={
                ("Is action 'unstack(B, A)' valid in current state?", "Yes"): 0.95,
                ("Is action 'unstack(B, A)' valid in current state?", "No"): 0.05,
            },
        )

        # 1. Action Generation (Reasoning Agent)
        action_prompt = "Propose action for Blocksworld state [on(A, Table), on(B, A)]"
        proposed_action = llm.generate(action_prompt, max_tokens=32, temperature=0.0)
        assert proposed_action == "Action: unstack(B, A)"

        # 2. Prior Probability / Transition Logprob (World Model / Action Prior)
        logprob = llm.get_logprobs(action_prompt, target=proposed_action)
        assert logprob == -0.12

        # 3. Self-Evaluation (Reward Model / Value Estimator)
        eval_prompt = "Is action 'unstack(B, A)' valid in current state?"
        p_valid = llm.get_token_probability(eval_prompt, "Yes")
        p_invalid = llm.get_token_probability(eval_prompt, "No")
        assert p_valid > p_invalid
        assert math.isclose(p_valid + p_invalid, 1.0)

        # Verify call history
        assert llm.call_count == 4
        assert [c.method for c in llm.calls] == [
            "generate",
            "get_logprobs",
            "get_token_probability",
            "get_token_probability",
        ]
