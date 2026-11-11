"""Comprehensive TDD tests for LiteLLM external API backend integration.

Tests cover the complete contract BEFORE production implementation:
1. LiteLLMClient (rap_demo.llm.litellm_client & rap_demo.llm):
   - BaseLLM protocol compliance, constructor validation, call_count tracking.
   - generate(): parameter forwarding, object & dict responses, malformed responses,
     and exception mapping to LLMTimeoutError / LLMConnectionError / LLMError.
   - get_logprobs(): OpenAI chat logprobs.content and legacy token_logprobs summation,
     graceful 0.0 fallback on unsupported/missing logprobs, operational error mapping.
   - get_token_probability(): math.exp(logprob) clamped to [0.0, 1.0] across
     chat top_logprobs and legacy top_logprobs, exact/leading-space/stripped matching,
     content-matching fallback when top_logprobs unavailable, error mapping.
2. Factory integration (rap_demo.llm.factory.get_llm):
   - Alias and case-insensitive resolution ('litellm', 'LiteLLM', 'lite_llm', 'lite-llm').
   - Keyword argument forwarding.
3. CLI integration (rap_demo.cli):
   - build_parser() flags (--backend litellm, --model, --api-base) and defaults on run/ui/compare.
   - _create_llm() backward compatibility and litellm configuration.
   - End-to-end `main(["run", ...])` execution with mocked litellm.completion.
   - Error reporting and helpful tip (API key / --model, never llama-server) on LLMError.
4. Automatic `.env` loading (`python-dotenv`):
   - Loads API tokens from `.env` in CWD when instantiating LiteLLMClient or running CLI.
   - Preserves existing environment variables (`override=False`).
   - Gracefully handles missing `.env` files.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from rap_demo.cli import _create_llm, build_parser, main
from rap_demo.llm import LiteLLMClient as ExportedLiteLLMClient
from rap_demo.llm.base import (
    BaseLLM,
    LLMConnectionError,
    LLMError,
    LLMTimeoutError,
)
from rap_demo.llm.factory import get_llm
from rap_demo.llm.litellm_client import LiteLLMClient
from rap_demo.llm.mock import MockLLM


# ============================================================================
# Helpers & Fixtures
# ============================================================================


def _import_litellm() -> Any:
    """Import litellm lazily so collection doesn't mask missing-module TDD failures."""
    import litellm

    return litellm


def _make_obj_response(
    content: str | None = "ok",
    logprobs: Any = None,
) -> SimpleNamespace:
    """Build an OpenAI/LiteLLM object-style ModelResponse."""
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        logprobs=logprobs,
    )
    return SimpleNamespace(choices=[choice])


def _make_dict_response(
    content: str | None = "ok",
    logprobs: Any = None,
) -> dict[str, Any]:
    """Build a dict-style completion response."""
    choice: dict[str, Any] = {
        "message": {"content": content},
    }
    if logprobs is not None:
        choice["logprobs"] = logprobs
    return {"choices": [choice]}


def _raising_completion(exc: Exception) -> Any:
    """Return a completion_fn callable that always raises the given exception."""

    def _fn(**_: Any) -> Any:
        raise exc

    return _fn


# ============================================================================
# 1. LiteLLMClient Protocol, Constructor & Call Tracking
# ============================================================================


class TestLiteLLMClientInitAndProtocol:
    """Tests exports, BaseLLM protocol conformance, validation, and call_count."""

    def test_exported_from_rap_demo_llm_package(self) -> None:
        """LiteLLMClient is exported from rap_demo.llm and matches module class."""
        assert ExportedLiteLLMClient is LiteLLMClient

    def test_satisfies_base_llm_protocol(self) -> None:
        """LiteLLMClient satisfies runtime-checkable BaseLLM protocol."""
        client = LiteLLMClient()
        assert isinstance(client, BaseLLM)

    def test_default_constructor_attributes(self) -> None:
        """Default constructor sets expected defaults and call_count=0."""
        client = LiteLLMClient()
        assert client.model == "gpt-4o-mini"
        assert client.api_key is None
        assert client.api_base is None
        assert client.timeout == 30.0
        assert client.call_count == 0

    def test_custom_constructor_attributes(self) -> None:
        """Custom constructor arguments are stored cleanly."""
        fn = lambda **kwargs: _make_obj_response("hi")  # noqa: E731
        client = LiteLLMClient(
            model="anthropic/claude-3-5-sonnet-20241022",
            api_key="sk-ant-test",
            api_base="https://custom.api.local/v1",
            timeout=15.5,
            completion_fn=fn,
        )
        assert client.model == "anthropic/claude-3-5-sonnet-20241022"
        assert client.api_key == "sk-ant-test"
        assert client.api_base == "https://custom.api.local/v1"
        assert client.timeout == 15.5
        assert client.call_count == 0

    @pytest.mark.parametrize("bad_model", ["", "   ", "\t\n"])
    def test_rejects_empty_or_whitespace_model(self, bad_model: str) -> None:
        """Empty or whitespace-only model string raises ValueError."""
        with pytest.raises(ValueError):
            LiteLLMClient(model=bad_model)

    def test_call_count_increments_across_methods(self) -> None:
        """call_count starts at 0 and increments on generate, get_logprobs, and get_token_probability."""

        def fake_completion(**kwargs: Any) -> SimpleNamespace:
            return _make_obj_response(
                content="Yes",
                logprobs=SimpleNamespace(
                    content=[
                        SimpleNamespace(
                            token="Yes",
                            logprob=-0.2,
                            top_logprobs=[{"token": "Yes", "logprob": -0.2}],
                        )
                    ]
                ),
            )

        client = LiteLLMClient(
            model="ollama/llama3.2",
            completion_fn=fake_completion,
        )
        assert client.call_count == 0

        client.generate("hello")
        assert client.call_count == 1

        client.get_logprobs("hello", "Yes")
        assert client.call_count == 2

        client.get_token_probability("hello", "Yes")
        assert client.call_count == 3


# ============================================================================
# 2. LiteLLMClient.generate()
# ============================================================================


class TestLiteLLMClientGenerate:
    """Tests text generation, parameter forwarding, response parsing, and error mapping."""

    def test_generate_forwards_required_and_optional_parameters(self) -> None:
        """Forwards model, messages, max_tokens, temperature, timeout, stop, api_key, api_base."""
        captured_kwargs: list[dict[str, Any]] = []

        def fake_completion(**kwargs: Any) -> SimpleNamespace:
            captured_kwargs.append(kwargs)
            return _make_obj_response("generated text")

        client = LiteLLMClient(
            model="openai/gpt-4o",
            api_key="sk-secret",
            api_base="https://proxy.example.com/v1",
            timeout=45.0,
            completion_fn=fake_completion,
        )
        result = client.generate(
            "Stack red on blue",
            max_tokens=256,
            temperature=0.4,
            stop=["\n", "END"],
        )

        assert result == "generated text"
        assert len(captured_kwargs) == 1
        kw = captured_kwargs[0]
        assert kw["model"] == "openai/gpt-4o"
        assert kw["messages"] == [{"role": "user", "content": "Stack red on blue"}]
        assert kw["max_tokens"] == 256
        assert kw["temperature"] == 0.4
        assert kw["timeout"] == 45.0
        assert kw["stop"] == ["\n", "END"]
        assert kw["api_key"] == "sk-secret"
        assert kw["api_base"] == "https://proxy.example.com/v1"

    def test_generate_omits_optional_none_parameters(self) -> None:
        """When stop, api_key, and api_base are None, they are not passed with None values."""
        captured_kwargs: list[dict[str, Any]] = []

        def fake_completion(**kwargs: Any) -> SimpleNamespace:
            captured_kwargs.append(kwargs)
            return _make_obj_response("ok")

        client = LiteLLMClient(model="gpt-4o-mini", completion_fn=fake_completion)
        client.generate("prompt")

        kw = captured_kwargs[0]
        assert kw["model"] == "gpt-4o-mini"
        assert kw["messages"] == [{"role": "user", "content": "prompt"}]
        assert kw["max_tokens"] == 128
        assert kw["temperature"] == 0.0
        assert kw["timeout"] == 30.0
        assert "stop" not in kw
        assert "api_key" not in kw
        assert "api_base" not in kw

    def test_generate_uses_litellm_completion_by_default(self) -> None:
        """When completion_fn is None, delegates to litellm.completion."""
        litellm = _import_litellm()
        client = LiteLLMClient(model="gpt-4o-mini")

        with patch.object(
            litellm,
            "completion",
            return_value=_make_obj_response("from litellm.completion"),
        ) as mock_comp:
            out = client.generate("hello world")

        assert out == "from litellm.completion"
        mock_comp.assert_called_once()
        assert mock_comp.call_args.kwargs["model"] == "gpt-4o-mini"

    def test_generate_handles_dict_style_response(self) -> None:
        """Supports dict-style response['choices'][0]['message']['content']."""
        client = LiteLLMClient(
            completion_fn=lambda **_: _make_dict_response("dict content"),
        )
        assert client.generate("prompt") == "dict content"

    def test_generate_allows_empty_string_content(self) -> None:
        """Empty string content '' is valid text and does not raise LLMError."""
        client = LiteLLMClient(
            completion_fn=lambda **_: _make_obj_response(""),
        )
        assert client.generate("prompt") == ""

    @pytest.mark.parametrize(
        "bad_response",
        [
            SimpleNamespace(choices=[]),
            SimpleNamespace(choices=None),
            SimpleNamespace(),
            {"choices": []},
            {"choices": None},
            {},
            _make_obj_response(content=None),
            _make_dict_response(content=None),
            SimpleNamespace(choices=[SimpleNamespace(message=None)]),
            {"choices": [{"message": {}}]},
            {"choices": [{}]},
        ],
    )
    def test_generate_raises_llm_error_on_malformed_or_empty_response(
        self, bad_response: Any
    ) -> None:
        """Missing/empty choices or None content raises LLMError."""
        client = LiteLLMClient(completion_fn=lambda **_: bad_response)
        with pytest.raises(LLMError):
            client.generate("prompt")

    def test_generate_maps_builtin_timeout_error_to_llm_timeout_error(self) -> None:
        """Built-in TimeoutError maps to LLMTimeoutError."""

        def raise_timeout(**_: Any) -> Any:
            raise TimeoutError("socket timed out")

        client = LiteLLMClient(completion_fn=raise_timeout)
        with pytest.raises(LLMTimeoutError):
            client.generate("prompt")

    def test_generate_maps_litellm_timeout_to_llm_timeout_error(self) -> None:
        """litellm.Timeout maps to LLMTimeoutError."""
        litellm = _import_litellm()

        def raise_timeout(**_: Any) -> Any:
            raise litellm.Timeout(
                message="Request timed out",
                model="gpt-4o-mini",
                llm_provider="openai",
            )

        client = LiteLLMClient(completion_fn=raise_timeout)
        with pytest.raises(LLMTimeoutError):
            client.generate("prompt")

    def test_generate_maps_builtin_connection_error_to_llm_connection_error(self) -> None:
        """Built-in ConnectionError maps to LLMConnectionError."""

        def raise_conn(**_: Any) -> Any:
            raise ConnectionError("network unreachable")

        client = LiteLLMClient(completion_fn=raise_conn)
        with pytest.raises(LLMConnectionError):
            client.generate("prompt")

    def test_generate_maps_litellm_api_connection_error_to_llm_connection_error(self) -> None:
        """litellm.APIConnectionError maps to LLMConnectionError."""
        litellm = _import_litellm()

        def raise_conn(**_: Any) -> Any:
            raise litellm.APIConnectionError(
                message="Cannot connect",
                llm_provider="openai",
                model="gpt-4o-mini",
            )

        client = LiteLLMClient(completion_fn=raise_conn)
        with pytest.raises(LLMConnectionError):
            client.generate("prompt")

    @pytest.mark.parametrize(
        "exc_name",
        ["AuthenticationError", "RateLimitError", "BadRequestError"],
    )
    def test_generate_maps_litellm_api_errors_to_llm_error(self, exc_name: str) -> None:
        """AuthenticationError, RateLimitError, BadRequestError map to LLMError."""
        litellm = _import_litellm()
        exc_cls = getattr(litellm, exc_name)

        def raise_exc(**_: Any) -> Any:
            raise exc_cls(
                message=f"Simulated {exc_name}",
                model="gpt-4o-mini",
                llm_provider="openai",
            )

        client = LiteLLMClient(completion_fn=raise_exc)
        with pytest.raises(LLMError) as exc_info:
            client.generate("prompt")
        assert not isinstance(exc_info.value, (LLMTimeoutError, LLMConnectionError))

    def test_generate_maps_generic_exception_to_llm_error(self) -> None:
        """Unexpected generic exceptions map to LLMError."""

        def raise_generic(**_: Any) -> Any:
            raise RuntimeError("unexpected failure")

        client = LiteLLMClient(completion_fn=raise_generic)
        with pytest.raises(LLMError):
            client.generate("prompt")


# ============================================================================
# 3. LiteLLMClient.get_logprobs()
# ============================================================================


class TestLiteLLMClientGetLogprobs:
    """Tests target-only echo scoring for Ollama and safe uniform fallback elsewhere."""

    def test_non_ollama_provider_uses_uniform_fallback(self) -> None:
        called = False

        def unexpected_completion(**_: Any) -> Any:
            nonlocal called
            called = True
            return _make_obj_response()

        client = LiteLLMClient(model="openai/gpt-4o-mini", completion_fn=unexpected_completion)
        assert client.get_logprobs("prompt", "target") == 0.0
        assert not called

    def test_ollama_echo_scores_only_target_tokens(self) -> None:
        captured: dict[str, Any] = {}
        prompt = "state: "

        def completion(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        logprobs=SimpleNamespace(
                            token_logprobs=[-0.1, -0.2, -0.3],
                            text_offset=[0, len(prompt), len(prompt) + 2],
                        )
                    )
                ]
            )

        client = LiteLLMClient(
            model="ollama/llama3.2",
            api_base="http://localhost:11434",
            completion_fn=completion,
        )
        assert client.get_logprobs(prompt, "red") == pytest.approx(-0.5)
        assert captured["model"] == "openai/llama3.2"
        assert captured["api_base"] == "http://localhost:11434/v1"
        assert captured["prompt"] == "state: red"
        assert captured["echo"] is True
        assert captured["max_tokens"] == 0

    def test_ollama_echo_without_offsets_uses_uniform_fallback(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    logprobs=SimpleNamespace(token_logprobs=[-0.1, -0.2])
                )
            ]
        )
        client = LiteLLMClient(
            model="ollama/llama3.2",
            completion_fn=lambda **_: response,
        )
        assert client.get_logprobs("prompt", "target") == 0.0

    def test_get_logprobs_returns_zero_on_unsupported_params_error(self) -> None:
        """Gracefully returns 0.0 when provider raises litellm.UnsupportedParamsError."""
        litellm = _import_litellm()

        def raise_unsupported(**_: Any) -> Any:
            raise litellm.UnsupportedParamsError(
                message="logprobs are not supported for this model",
                llm_provider="anthropic",
                model="anthropic/claude-3-5-sonnet-20241022",
            )

        client = LiteLLMClient(
            model="ollama/llama3.2",
            completion_fn=raise_unsupported,
        )
        assert client.get_logprobs("prompt", "target") == 0.0

    def test_get_logprobs_returns_zero_on_unsupported_param_bad_request(self) -> None:
        """Gracefully returns 0.0 when BadRequestError indicates unsupported logprobs."""
        litellm = _import_litellm()

        def raise_bad_req(**_: Any) -> Any:
            raise litellm.BadRequestError(
                message="Unsupported parameter: logprobs",
                model="custom/model",
                llm_provider="custom",
            )

        client = LiteLLMClient(model="ollama/llama3.2", completion_fn=raise_bad_req)
        assert client.get_logprobs("prompt", "target") == 0.0

    def test_get_logprobs_raises_on_timeout_connection_and_auth_errors(self) -> None:
        """Operational failures still raise LLMTimeoutError / LLMConnectionError / LLMError."""
        litellm = _import_litellm()

        timeout_client = LiteLLMClient(
            model="ollama/llama3.2",
            completion_fn=_raising_completion(
                litellm.Timeout(message="timeout", model="gpt-4o-mini", llm_provider="openai")
            )
        )
        with pytest.raises(LLMTimeoutError):
            timeout_client.get_logprobs("p", "t")

        conn_client = LiteLLMClient(
            model="ollama/llama3.2",
            completion_fn=_raising_completion(
                litellm.APIConnectionError(
                    message="conn failed", llm_provider="openai", model="gpt-4o-mini"
                )
            )
        )
        with pytest.raises(LLMConnectionError):
            conn_client.get_logprobs("p", "t")

        auth_client = LiteLLMClient(
            model="ollama/llama3.2",
            completion_fn=_raising_completion(
                litellm.AuthenticationError(
                    message="bad key", llm_provider="openai", model="gpt-4o-mini"
                )
            )
        )
        with pytest.raises(LLMError):
            auth_client.get_logprobs("p", "t")


# ============================================================================
# 4. LiteLLMClient.get_token_probability()
# ============================================================================


class TestLiteLLMClientGetTokenProbability:
    """Tests token probability extraction, token matching, clamping, and fallback."""

    def test_chat_format_dict_top_logprobs_exact_match(self) -> None:
        """Computes math.exp(logprob) from choices[0].logprobs.content[0].top_logprobs dicts."""
        lp_yes = math.log(0.75)
        lp_no = math.log(0.25)
        resp = _make_obj_response(
            "Yes",
            logprobs=SimpleNamespace(
                content=[
                    SimpleNamespace(
                        token="Yes",
                        logprob=lp_yes,
                        top_logprobs=[
                            {"token": "Yes", "logprob": lp_yes},
                            {"token": "No", "logprob": lp_no},
                        ],
                    )
                ]
            ),
        )
        client = LiteLLMClient(completion_fn=lambda **_: resp)
        assert client.get_token_probability("Is valid?", "Yes") == pytest.approx(0.75)
        assert client.get_token_probability("Is valid?", "No") == pytest.approx(0.25)

    def test_chat_format_object_top_logprobs_and_dict_outer_response(self) -> None:
        """Supports object-style top_logprobs items and dict-style outer response."""
        lp_yes = math.log(0.8)
        resp_obj_items = _make_obj_response(
            "Yes",
            logprobs=SimpleNamespace(
                content=[
                    SimpleNamespace(
                        top_logprobs=[
                            SimpleNamespace(token="Yes", logprob=lp_yes),
                        ]
                    )
                ]
            ),
        )
        assert LiteLLMClient(completion_fn=lambda **_: resp_obj_items).get_token_probability(
            "p", "Yes"
        ) == pytest.approx(0.8)

        resp_dict_outer = _make_dict_response(
            "Yes",
            logprobs={
                "content": [
                    {
                        "token": "Yes",
                        "logprob": lp_yes,
                        "top_logprobs": [{"token": "Yes", "logprob": lp_yes}],
                    }
                ]
            },
        )
        assert LiteLLMClient(completion_fn=lambda **_: resp_dict_outer).get_token_probability(
            "p", "Yes"
        ) == pytest.approx(0.8)

    def test_legacy_top_logprobs_dict(self) -> None:
        """Supports legacy choices[0].logprobs.top_logprobs[0] dict[str, float]."""
        lp_yes = math.log(0.6)
        resp_obj = _make_obj_response(
            "Yes",
            logprobs=SimpleNamespace(top_logprobs=[{"Yes": lp_yes, "No": math.log(0.4)}]),
        )
        assert LiteLLMClient(completion_fn=lambda **_: resp_obj).get_token_probability(
            "p", "Yes"
        ) == pytest.approx(0.6)

        resp_dict = _make_dict_response(
            "Yes",
            logprobs={"top_logprobs": [{"Yes": lp_yes, "No": math.log(0.4)}]},
        )
        assert LiteLLMClient(completion_fn=lambda **_: resp_dict).get_token_probability(
            "p", "Yes"
        ) == pytest.approx(0.6)

    def test_token_matching_leading_space_and_stripped_and_missing(self) -> None:
        """Matches exact token, leading-space token, stripped match, or returns 0.0 if absent."""
        lp_space = math.log(0.65)
        lp_trailing = math.log(0.20)
        resp = _make_obj_response(
            " Yes",
            logprobs=SimpleNamespace(
                content=[
                    SimpleNamespace(
                        top_logprobs=[
                            {"token": " Yes", "logprob": lp_space},
                            {"token": "Maybe ", "logprob": lp_trailing},
                        ]
                    )
                ]
            ),
        )
        client = LiteLLMClient(completion_fn=lambda **_: resp)
        assert client.get_token_probability("p", "Yes") == pytest.approx(0.65)
        assert client.get_token_probability("p", "Maybe") == pytest.approx(0.20)
        assert client.get_token_probability("p", "Never") == 0.0

    def test_exact_match_takes_precedence_over_leading_space(self) -> None:
        """Exact token match is preferred over leading-space variant."""
        resp = _make_obj_response(
            "Yes",
            logprobs=SimpleNamespace(
                content=[
                    SimpleNamespace(
                        top_logprobs=[
                            {"token": " Yes", "logprob": math.log(0.1)},
                            {"token": "Yes", "logprob": math.log(0.9)},
                        ]
                    )
                ]
            ),
        )
        client = LiteLLMClient(completion_fn=lambda **_: resp)
        assert client.get_token_probability("p", "Yes") == pytest.approx(0.9)

    def test_probability_clamped_to_unit_interval(self) -> None:
        """Positive logprobs from floating-point quirks are clamped to 1.0."""
        resp = _make_obj_response(
            "Yes",
            logprobs=SimpleNamespace(
                content=[
                    SimpleNamespace(
                        top_logprobs=[{"token": "Yes", "logprob": 0.05}]
                    )
                ]
            ),
        )
        client = LiteLLMClient(completion_fn=lambda **_: resp)
        prob = client.get_token_probability("p", "Yes")
        assert prob == 1.0

    @pytest.mark.parametrize(
        "response",
        [
            _make_obj_response(" Yes\n", logprobs=None),
            _make_dict_response(" Yes ", logprobs=None),
            _make_obj_response("Yes", logprobs=SimpleNamespace(content=[])),
            _make_obj_response(
                "Yes",
                logprobs=SimpleNamespace(content=[SimpleNamespace(top_logprobs=None)]),
            ),
            _make_obj_response(
                "Yes",
                logprobs=SimpleNamespace(content=[SimpleNamespace(top_logprobs=[])]),
            ),
            _make_obj_response("Yes", logprobs=SimpleNamespace(top_logprobs=[])),
        ],
    )
    def test_fallback_when_top_logprobs_unavailable_matches_stripped_content(
        self, response: Any
    ) -> None:
        """Without top_logprobs, returns 1.0 when stripped content matches token.strip(), else 0.0."""
        client = LiteLLMClient(completion_fn=lambda **_: response)
        assert client.get_token_probability("Is valid?", "Yes") == 1.0
        assert client.get_token_probability("Is valid?", "No") == 0.0

    def test_fallback_when_content_is_none_and_no_top_logprobs(self) -> None:
        """If both top_logprobs and content are unavailable, returns 0.0 cleanly."""
        resp = _make_obj_response(content=None, logprobs=None)
        client = LiteLLMClient(completion_fn=lambda **_: resp)
        assert client.get_token_probability("Is valid?", "Yes") == 0.0

    def test_get_token_probability_handles_unsupported_params_gracefully(self) -> None:
        """Does not crash if provider rejects logprobs with UnsupportedParamsError."""
        litellm = _import_litellm()

        def fake_completion(**kwargs: Any) -> Any:
            if kwargs.get("logprobs") or kwargs.get("top_logprobs"):
                raise litellm.UnsupportedParamsError(
                    message="top_logprobs unsupported",
                    llm_provider="anthropic",
                    model="anthropic/claude-3-5-sonnet-20241022",
                )
            return _make_obj_response("Yes")

        client = LiteLLMClient(
            model="anthropic/claude-3-5-sonnet-20241022",
            completion_fn=fake_completion,
        )
        prob = client.get_token_probability("Is valid?", "Yes")
        assert 0.0 <= prob <= 1.0

    def test_get_token_probability_maps_connection_timeout_and_auth_errors(self) -> None:
        """Maps network, timeout, and auth errors to LLMConnectionError/LLMTimeoutError/LLMError."""
        litellm = _import_litellm()

        with pytest.raises(LLMTimeoutError):
            LiteLLMClient(
                completion_fn=_raising_completion(TimeoutError("t"))
            ).get_token_probability("p", "Yes")

        with pytest.raises(LLMConnectionError):
            LiteLLMClient(
                completion_fn=_raising_completion(ConnectionError("c"))
            ).get_token_probability("p", "Yes")

        with pytest.raises(LLMError):
            LiteLLMClient(
                completion_fn=_raising_completion(
                    litellm.AuthenticationError(
                        message="invalid key",
                        llm_provider="openai",
                        model="gpt-4o-mini",
                    )
                )
            ).get_token_probability("p", "Yes")


# ============================================================================
# 5. Factory Integration (rap_demo.llm.factory.get_llm)
# ============================================================================


class TestLiteLLMFactory:
    """Tests get_llm() resolution, aliases, and keyword forwarding for litellm."""

    def test_get_llm_litellm_with_model_and_api_key(self) -> None:
        """get_llm('litellm', ...) returns configured LiteLLMClient."""
        llm = get_llm(
            "litellm",
            model="anthropic/claude-3-5-sonnet-20241022",
            api_key="sk-test",
        )
        assert isinstance(llm, LiteLLMClient)
        assert isinstance(llm, BaseLLM)
        assert llm.model == "anthropic/claude-3-5-sonnet-20241022"
        assert llm.api_key == "sk-test"

    @pytest.mark.parametrize(
        "backend_name",
        ["litellm", "LiteLLM", "LITELLM", "lite_llm", "lite-llm", "LITE-LLM"],
    )
    def test_get_llm_litellm_case_insensitive_and_aliases(self, backend_name: str) -> None:
        """get_llm supports case-insensitive names and underscore/hyphen aliases."""
        llm = get_llm(backend_name)
        assert isinstance(llm, LiteLLMClient)

    def test_get_llm_forwards_all_litellm_kwargs(self) -> None:
        """get_llm forwards model, api_key, api_base, timeout, and completion_fn."""
        calls: list[dict[str, Any]] = []

        def custom_fn(**kwargs: Any) -> SimpleNamespace:
            calls.append(kwargs)
            return _make_obj_response("custom-factory")

        llm = get_llm(
            "litellm",
            model="openai/gpt-4o",
            api_key="sk-key",
            api_base="https://endpoint.test/v1",
            timeout=12.0,
            completion_fn=custom_fn,
        )
        assert isinstance(llm, LiteLLMClient)
        assert llm.model == "openai/gpt-4o"
        assert llm.api_key == "sk-key"
        assert llm.api_base == "https://endpoint.test/v1"
        assert llm.timeout == 12.0
        assert llm.generate("hi") == "custom-factory"
        assert len(calls) == 1


# ============================================================================
# 6. CLI Integration (rap_demo.cli)
# ============================================================================


class TestLiteLLMCLIIntegration:
    """Tests CLI parser options, _create_llm(), end-to-end run, and error tips."""

    @pytest.mark.parametrize("subcommand", ["run", "ui", "compare"])
    def test_parser_accepts_litellm_backend_model_and_api_base(
        self, subcommand: str
    ) -> None:
        """build_parser() supports --backend litellm, --model, and --api-base on all subcommands."""
        parser = build_parser()
        args = parser.parse_args(
            [
                subcommand,
                "--backend",
                "litellm",
                "--model",
                "anthropic/claude-3-5-sonnet-20241022",
                "--api-base",
                "https://custom.endpoint/v1",
            ]
        )
        assert args.backend == "litellm"
        assert args.model == "anthropic/claude-3-5-sonnet-20241022"
        assert args.api_base == "https://custom.endpoint/v1"

    @pytest.mark.parametrize("subcommand", ["run", "ui", "compare"])
    def test_parser_defaults_preserve_backward_compatibility(
        self, subcommand: str
    ) -> None:
        """Default backend remains 'mock' and model/api_base default to None."""
        parser = build_parser()
        args = parser.parse_args([subcommand])
        assert args.backend == "mock"
        assert args.model is None
        assert args.api_base is None

    def test_create_llm_litellm_with_model_and_api_base(self) -> None:
        """_create_llm('litellm', model=..., api_base=...) configures LiteLLMClient."""
        llm = _create_llm(
            "litellm",
            model="openai/gpt-4o",
            api_base="https://custom.endpoint/v1",
        )
        assert isinstance(llm, LiteLLMClient)
        assert llm.model == "openai/gpt-4o"
        assert llm.api_base == "https://custom.endpoint/v1"

    def test_create_llm_mock_single_positional_arg_backward_compatible(self) -> None:
        """_create_llm('mock') with one positional argument still returns configured MockLLM."""
        llm = _create_llm("mock")
        assert isinstance(llm, MockLLM)
        out = llm.generate("Pick up the red block.")
        assert "[CHANGE]" in out

    def test_main_run_with_litellm_mocked_completion_succeeds(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main(['run', '--backend', 'litellm', ...]) runs MCTS search and exits 0."""
        litellm = _import_litellm()

        def fake_completion(**kwargs: Any) -> SimpleNamespace:
            messages = kwargs.get("messages", [])
            prompt_text = messages[-1]["content"] if messages else ""

            # Reward self-evaluation prompt check
            if (
                kwargs.get("logprobs")
                or kwargs.get("max_tokens") == 1
                or "Is this" in prompt_text
                or "Yes" in prompt_text
            ):
                return _make_obj_response(
                    "Yes",
                    logprobs=SimpleNamespace(
                        content=[
                            SimpleNamespace(
                                token="Yes",
                                logprob=math.log(0.9),
                                top_logprobs=[
                                    {"token": "Yes", "logprob": math.log(0.9)},
                                    {"token": "No", "logprob": math.log(0.1)},
                                ],
                            )
                        ]
                    ),
                )

            # Hybrid mode action proposal JSON
            if "holding the red block" in prompt_text:
                proposal = {
                    "thought": "Stack red onto blue",
                    "action": "stack",
                    "block": "red",
                    "target": "blue",
                }
            else:
                proposal = {
                    "thought": "Pick up red from table",
                    "action": "pick_up",
                    "block": "red",
                    "target": None,
                }
            return _make_obj_response(json.dumps(proposal))

        with patch.object(litellm, "completion", side_effect=fake_completion) as mock_comp:
            exit_code = main(
                [
                    "run",
                    "--backend",
                    "litellm",
                    "--model",
                    "gpt-4o-mini",
                    "--iterations",
                    "2",
                ]
            )

        assert exit_code == 0
        assert mock_comp.call_count >= 1
        assert all(
            call.kwargs.get("model") == "gpt-4o-mini"
            for call in mock_comp.call_args_list
        )
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_main_run_litellm_llm_error_exits_1_with_helpful_tip_not_llama_server(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When LiteLLMClient raises LLMError, CLI exits 1 with API key/--model tip, not llama-server."""
        litellm = _import_litellm()

        def raise_auth_error(**_: Any) -> Any:
            raise litellm.AuthenticationError(
                message="Missing OPENAI_API_KEY",
                llm_provider="openai",
                model="gpt-4o-mini",
            )

        with patch.object(litellm, "completion", side_effect=raise_auth_error):
            exit_code = main(
                [
                    "run",
                    "--backend",
                    "litellm",
                    "--model",
                    "gpt-4o-mini",
                    "--iterations",
                    "2",
                ]
            )

        assert exit_code == 1
        captured = capsys.readouterr()
        err_text = captured.err.lower()
        assert "error" in err_text
        assert "api" in err_text or "key" in err_text or "--model" in err_text
        assert "llama-server" not in err_text


# ============================================================================
# 7. Automatic .env Loading (python-dotenv)
# ============================================================================


class TestLiteLLMDotenvLoading:
    """Tests automatic loading of API tokens from .env with override=False."""

    @pytest.fixture(autouse=True)
    def _clean_api_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ensure API key env vars are cleared for test and restored via monkeypatch."""
        for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            # Record original state (even if initially unset) before clearing
            monkeypatch.setenv(var, "")
            monkeypatch.delenv(var, raising=False)

    def test_litellm_client_loads_dotenv_from_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Instantiating LiteLLMClient automatically loads .env from the working directory."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "OPENAI_API_KEY=sk-from-dotenv\nANTHROPIC_API_KEY=sk-ant-dotenv\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)

        LiteLLMClient(model="gpt-4o-mini")

        assert os.environ.get("OPENAI_API_KEY") == "sk-from-dotenv"
        assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-dotenv"

    def test_litellm_client_dotenv_respects_existing_env_vars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Existing shell env vars take precedence over .env (override=False)."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "OPENAI_API_KEY=sk-from-dotenv\nANTHROPIC_API_KEY=sk-ant-dotenv\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("OPENAI_API_KEY", "sk-shell-override")
        monkeypatch.chdir(tmp_path)

        LiteLLMClient(model="gpt-4o-mini")

        assert os.environ.get("OPENAI_API_KEY") == "sk-shell-override"
        assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-dotenv"

    def test_cli_run_loads_dotenv_before_calling_litellm_completion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI `rap run --backend litellm` loads .env before invoking litellm.completion."""
        litellm = _import_litellm()
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=sk-from-dotenv\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        observed_keys: list[str | None] = []

        def fake_completion(**kwargs: Any) -> SimpleNamespace:
            observed_keys.append(os.environ.get("OPENAI_API_KEY"))
            if kwargs.get("logprobs") or kwargs.get("max_tokens") == 1:
                return _make_obj_response(
                    "Yes",
                    logprobs=SimpleNamespace(
                        content=[
                            SimpleNamespace(
                                token="Yes",
                                logprob=math.log(0.9),
                                top_logprobs=[
                                    {"token": "Yes", "logprob": math.log(0.9)},
                                    {"token": "No", "logprob": math.log(0.1)},
                                ],
                            )
                        ]
                    ),
                )
            return _make_obj_response(
                json.dumps(
                    {
                        "thought": "Pick up red from table",
                        "action": "pick_up",
                        "block": "red",
                        "target": None,
                    }
                )
            )

        with patch.object(litellm, "completion", side_effect=fake_completion):
            exit_code = main(
                [
                    "run",
                    "--backend",
                    "litellm",
                    "--model",
                    "gpt-4o-mini",
                    "--iterations",
                    "1",
                ]
            )

        assert exit_code == 0
        assert observed_keys
        assert all(key == "sk-from-dotenv" for key in observed_keys)

    def test_missing_dotenv_file_does_not_raise_on_client_or_cli(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing .env file in working directory causes no error for LiteLLMClient or CLI."""
        litellm = _import_litellm()
        monkeypatch.chdir(tmp_path)
        assert not (tmp_path / ".env").exists()

        client = LiteLLMClient(model="gpt-4o-mini")
        assert isinstance(client, BaseLLM)

        def fake_completion(**kwargs: Any) -> SimpleNamespace:
            if kwargs.get("logprobs") or kwargs.get("max_tokens") == 1:
                return _make_obj_response("Yes")
            return _make_obj_response(
                json.dumps(
                    {
                        "thought": "Pick up red",
                        "action": "pick_up",
                        "block": "red",
                        "target": None,
                    }
                )
            )

        with patch.object(litellm, "completion", side_effect=fake_completion):
            exit_code = main(
                [
                    "run",
                    "--backend",
                    "litellm",
                    "--model",
                    "gpt-4o-mini",
                    "--iterations",
                    "1",
                ]
            )

        assert exit_code == 0



class TestReasoningModelSupport:
    """Tests reasoning_content/thinking_blocks fallback and sufficient max_tokens for thinking models."""

    def test_generate_falls_back_to_reasoning_content_or_thinking_blocks(self) -> None:
        """When a reasoning model returns content=None with reasoning_content or thinking_blocks, extract it."""
        obj_resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        reasoning_content='{"thought": "think", "action": "pick_up", "block": "red", "target": null}',
                    )
                )
            ]
        )
        client_reasoning = LiteLLMClient(completion_fn=lambda **_: obj_resp)
        assert "pick_up" in client_reasoning.generate("prompt")

        blocks_resp = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "thinking_blocks": [
                            {"type": "thinking", "thinking": "Considering move to pick up red."}
                        ],
                    }
                }
            ]
        }
        client_blocks = LiteLLMClient(completion_fn=lambda **_: blocks_resp)
        assert client_blocks.generate("prompt") == "Considering move to pick up red."

    def test_blocksworld_hybrid_agent_requests_sufficient_max_tokens_for_reasoning_models(self) -> None:
        """BlocksworldAgent hybrid mode passes max_tokens >= 512 so thinking models don't truncate before JSON."""
        from rap_demo.domains.blocksworld.agent import BlocksworldAgent
        from rap_demo.domains.blocksworld.models import BlockState

        seen_max_tokens: list[int] = []

        def fake_completion(**kwargs: Any) -> SimpleNamespace:
            seen_max_tokens.append(int(kwargs["max_tokens"]))
            return _make_obj_response(
                json.dumps(
                    {
                        "thought": "Pick up red",
                        "action": "pick_up",
                        "block": "red",
                        "target": None,
                    }
                )
            )

        client = LiteLLMClient(model="gemini/gemma-4-26b-a4b-it", completion_fn=fake_completion)
        agent = BlocksworldAgent(llm=client, goal="the red block is on top of the blue block", mode="hybrid")
        state = BlockState(clear={"red", "blue"}, on_table={"red", "blue"}, on={}, holding=None)

        actions = agent.get_actions(state)
        assert len(actions) == 1
        assert seen_max_tokens and seen_max_tokens[0] >= 512
