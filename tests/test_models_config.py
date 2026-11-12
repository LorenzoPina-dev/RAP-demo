"""Comprehensive TDD tests for models.json configuration, rate-limit precedence, and CLI integration.

Tests cover contracts BEFORE production implementation:
1. Configuration loader & schema (`rap_demo.config` / `rap_demo.config.models`):
   - `get_models_config_path()` precedence (explicit -> RAP_MODELS_CONFIG -> Path('models.json')).
   - `ModelProfile` defaults and positive numeric validation (tpm, rpm, timeout, iterations; rejecting <=0 and bools).
   - `ModelsConfig.resolve_model()` matching exact alias, exact model ID, case-insensitive alias/ID, and short model name.
   - `load_models_config()` missing explicit vs missing default behavior, valid parsing, and ValueError on invalid JSON/schema.
2. Rate-limit precedence (`rap_demo.llm.rate_limiter` & `LiteLLMClient`):
   - Priority: explicit arg > per-model env vars / RAP_MODEL_LIMITS > models.json profile > global RAP_TPM/RAP_RPM.
   - `LiteLLMClient` reading profile limits from `RAP_MODELS_CONFIG` / `models.json`.
3. CLI integration (`rap_demo.cli`):
   - `--config` flag on `run`, `ui`, `compare`, and `models`.
   - `rap models` subcommand rendering Rich table (and empty config handling).
   - Backward compatibility (`parser.parse_args(["run"]).backend == "mock"`, `main(["run"])` uses `mock`).
   - Alias/model resolution via `--model` without explicit `--backend` on `run` and `ui`.
   - Explicit CLI flags (`--backend`, `--tpm`, `--rpm`, `--api-base`) overriding `models.json` profile values.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rap_demo.cli import _create_llm, _resolve_execution_llm_config, build_parser, main
from rap_demo.config import (
    ModelProfile,
    ModelsConfig,
    get_models_config_path,
    load_models_config,
)
from rap_demo.config.models import (
    ModelProfile as DirectModelProfile,
    ModelsConfig as DirectModelsConfig,
    get_models_config_path as direct_get_models_config_path,
    load_models_config as direct_load_models_config,
)
from rap_demo.llm.litellm_client import LiteLLMClient
from rap_demo.llm.mock import MockLLM
from rap_demo.llm.rate_limiter import resolve_model_rate_limits


# ============================================================================
# Helpers & Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def _clean_config_and_rate_env(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Clean RAP_MODELS_CONFIG and rate-limit env vars before and after each test."""
    known_vars = [
        "RAP_MODELS_CONFIG",
        "RAP_TPM",
        "RAP_RPM",
        "RAP_MODEL_LIMITS",
        "RAP_TPM_GEMINI_GEMMA_4_26B_A4B_IT",
        "RAP_RPM_GEMINI_GEMMA_4_26B_A4B_IT",
        "RAP_TPM_GEMMA_4_26B_A4B_IT",
        "RAP_RPM_GEMMA_4_26B_A4B_IT",
        "RAP_TPM_GEMMA4_26B",
        "RAP_RPM_GEMMA4_26B",
    ]
    existing_dynamic = [
        k
        for k in list(os.environ)
        if k.startswith(("RAP_TPM", "RAP_RPM")) or k in ("RAP_MODEL_LIMITS", "RAP_MODELS_CONFIG")
    ]
    for var in set(known_vars + existing_dynamic):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var, raising=False)
    yield
    for k in list(os.environ):
        if k.startswith(("RAP_TPM", "RAP_RPM")) or k in ("RAP_MODEL_LIMITS", "RAP_MODELS_CONFIG"):
            monkeypatch.delenv(k, raising=False)


def _import_litellm() -> Any:
    """Import litellm lazily after compatibility shims are loaded."""
    import litellm

    return litellm


def _sample_models_payload() -> dict[str, Any]:
    """Build a representative models.json payload with multiple profiles."""
    return {
        "default_model": "gemma4-26b",
        "defaults": {
            "mode": "hybrid",
            "iterations": 10,
            "timeout": 45.0,
        },
        "models": {
            "gemma4-26b": {
                "backend": "litellm",
                "model": "gemini/gemma-4-26b-a4b-it",
                "tpm": 15000,
                "rpm": 30,
                "timeout": 60.0,
                "mode": "hybrid",
                "iterations": 12,
            },
            "gpt4o-mini": {
                "backend": "litellm",
                "model": "openai/gpt-4o-mini",
                "api_base": "https://proxy.example.com/v1",
                "tpm": 200000,
                "rpm": 500,
            },
            "local-lfm": {
                "backend": "llama-server",
                "model": "LFM3-VL-2B",
                "api_base": "http://localhost:8080",
            },
        },
    }


def _write_models_json(path: Path, payload: dict[str, Any] | None = None) -> Path:
    data = payload if payload is not None else _sample_models_payload()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


# ============================================================================
# 1. Config Loader & Schema Tests (`rap_demo.config.models`)
# ============================================================================


class TestModelsConfigPathAndExports:
    """Tests package exports and get_models_config_path() resolution."""

    def test_config_package_exports_match_models_module(self) -> None:
        """rap_demo.config re-exports symbols from rap_demo.config.models."""
        assert get_models_config_path is direct_get_models_config_path
        assert ModelProfile is DirectModelProfile
        assert ModelsConfig is DirectModelsConfig
        assert load_models_config is direct_load_models_config

    def test_get_models_config_path_precedence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit config_path beats RAP_MODELS_CONFIG, which beats Path('models.json')."""
        assert get_models_config_path() == Path("models.json")

        monkeypatch.setenv("RAP_MODELS_CONFIG", "   ")
        assert get_models_config_path() == Path("models.json")

        env_cfg = tmp_path / "env_models.json"
        monkeypatch.setenv("RAP_MODELS_CONFIG", str(env_cfg))
        assert get_models_config_path() == env_cfg

        explicit_cfg = tmp_path / "explicit_models.json"
        assert get_models_config_path(explicit_cfg) == explicit_cfg
        assert get_models_config_path(str(explicit_cfg)) == explicit_cfg


class TestModelProfileAndModelsConfig:
    """Tests ModelProfile validation and ModelsConfig.resolve_model()."""

    def test_model_profile_defaults_and_custom_fields(self) -> None:
        """ModelProfile sets sensible defaults and stores all optional fields."""
        minimal = ModelProfile(alias="my-model")
        assert minimal.alias == "my-model"
        assert minimal.backend == "litellm"
        assert minimal.model is None
        assert minimal.api_base is None
        assert minimal.tpm is None
        assert minimal.rpm is None
        assert minimal.timeout is None
        assert minimal.mode is None
        assert minimal.iterations is None
        assert minimal.reasoning_effort is None

        full = ModelProfile(
            alias="gemma4-26b",
            backend="litellm",
            model="gemini/gemma-4-26b-a4b-it",
            api_base="https://generativelanguage.googleapis.com",
            tpm=15000,
            rpm=30,
            timeout=60.0,
            mode="hybrid",
            iterations=15,
            reasoning_effort="low",
        )
        assert full.tpm == 15000
        assert full.rpm == 30
        assert full.timeout == 60.0
        assert full.iterations == 15
        assert full.reasoning_effort == "low"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"tpm": 0},
            {"tpm": -100},
            {"tpm": True},
            {"tpm": False},
            {"rpm": 0},
            {"rpm": -5},
            {"rpm": True},
            {"timeout": 0},
            {"timeout": -1.5},
            {"timeout": True},
            {"iterations": 0},
            {"iterations": -3},
            {"iterations": False},
        ],
    )
    def test_model_profile_rejects_non_positive_or_bool_numeric_fields(
        self, kwargs: dict[str, Any]
    ) -> None:
        """ModelProfile rejects <= 0 and bool values for tpm, rpm, timeout, and iterations."""
        with pytest.raises(ValueError):
            ModelProfile(alias="bad-profile", **kwargs)

    def test_resolve_model_exact_alias_exact_id_case_insensitive_and_short_name(
        self, tmp_path: Path
    ) -> None:
        """ModelsConfig.resolve_model resolves exact alias, exact ID, case-insensitive, and short name."""
        cfg_path = _write_models_json(tmp_path / "models.json")
        cfg = load_models_config(cfg_path)

        # 1. Exact alias key
        by_alias = cfg.resolve_model("gemma4-26b")
        assert by_alias is not None
        assert by_alias.model == "gemini/gemma-4-26b-a4b-it"
        assert by_alias.tpm == 15000

        # 2. Exact profile.model
        by_model_id = cfg.resolve_model("gemini/gemma-4-26b-a4b-it")
        assert by_model_id is not None
        assert by_model_id.alias == "gemma4-26b"

        # 3. Case-insensitive alias & model ID
        assert cfg.resolve_model("GEMMA4-26B") is by_alias
        assert cfg.resolve_model("GEMINI/GEMMA-4-26B-A4B-IT") is by_model_id

        # 4. Short name after '/' (and case-insensitive short name)
        by_short = cfg.resolve_model("gemma-4-26b-a4b-it")
        assert by_short is not None
        assert by_short.alias == "gemma4-26b"
        assert cfg.resolve_model("GEMMA-4-26B-A4B-IT") is by_short

        # Unknown model -> None
        assert cfg.resolve_model("nonexistent-model") is None
        assert cfg.resolve_model("") is None


class TestLoadModelsConfig:
    """Tests load_models_config() file loading, defaults, and error handling."""

    def test_missing_default_models_json_returns_empty_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When default models.json does not exist, returns empty ModelsConfig() cleanly."""
        monkeypatch.chdir(tmp_path)
        cfg = load_models_config()
        assert isinstance(cfg, ModelsConfig)
        assert cfg.default_model is None
        assert cfg.defaults == {}
        assert cfg.models == {}

    def test_missing_explicit_config_path_raises_file_not_found(self, tmp_path: Path) -> None:
        """Explicit config_path pointing to missing file raises FileNotFoundError."""
        missing = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError):
            load_models_config(missing)

    def test_loads_valid_models_json_via_explicit_or_env_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Loads default_model, defaults, and models mapping from valid JSON."""
        cfg_file = _write_models_json(tmp_path / "custom_models.json")
        monkeypatch.setenv("RAP_MODELS_CONFIG", str(cfg_file))

        cfg = load_models_config()
        assert cfg.default_model == "gemma4-26b"
        assert cfg.defaults["mode"] == "hybrid"
        assert cfg.defaults["iterations"] == 10
        assert set(cfg.models.keys()) == {"gemma4-26b", "gpt4o-mini", "local-lfm"}
        assert cfg.models["gpt4o-mini"].api_base == "https://proxy.example.com/v1"
        assert cfg.models["local-lfm"].backend == "llama-server"

    @pytest.mark.parametrize(
        "raw_content",
        [
            "{invalid json",
            "[1, 2, 3]",
            '"just a string"',
            json.dumps({"models": ["not-a-mapping"]}),
            json.dumps({"models": {"m1": "not-a-dict"}}),
            json.dumps({"models": {"m1": {"tpm": 0}}}),
            json.dumps({"models": {"m1": {"rpm": -1}}}),
            json.dumps({"models": {"m1": {"timeout": -5.0}}}),
            json.dumps({"models": {"m1": {"iterations": 0}}}),
            json.dumps({"models": {"m1": {"tpm": True}}}),
        ],
    )
    def test_invalid_json_or_schema_raises_value_error(
        self, tmp_path: Path, raw_content: str
    ) -> None:
        """Malformed JSON, non-dict root/models, or invalid constraints raise ValueError."""
        bad_file = tmp_path / "bad_models.json"
        bad_file.write_text(raw_content, encoding="utf-8")
        with pytest.raises(ValueError):
            load_models_config(bad_file)


# ============================================================================
# 2. Rate-Limit Resolution & LiteLLMClient Integration
# ============================================================================


class TestModelsConfigRateLimitPrecedence:
    """Tests rate_limiter precedence with models.json and LiteLLMClient integration."""

    def test_resolve_model_rate_limits_full_precedence_with_models_json(
        self, tmp_path: Path
    ) -> None:
        """Priority: explicit > per-model env / RAP_MODEL_LIMITS > models.json > global env."""
        cfg_file = _write_models_json(tmp_path / "models.json")
        model = "gemini/gemma-4-26b-a4b-it"

        # 4. Global fallback when models.json has no matching model
        assert resolve_model_rate_limits(
            "unknown/model",
            environ={"RAP_TPM": "1000", "RAP_RPM": "10"},
            config_path=cfg_file,
        ) == (1000, 10)

        # 3. models.json beats global RAP_TPM / RAP_RPM
        assert resolve_model_rate_limits(
            model,
            environ={"RAP_TPM": "1000", "RAP_RPM": "10"},
            config_path=cfg_file,
        ) == (15000, 30)

        # 3b. Partial override: models.json tpm (15000) + global RAP_RPM when profile has no rpm
        partial_cfg = _write_models_json(
            tmp_path / "partial.json",
            {
                "models": {
                    "gemma4-26b": {
                        "backend": "litellm",
                        "model": "gemini/gemma-4-26b-a4b-it",
                        "tpm": 15000,
                    }
                }
            },
        )
        assert resolve_model_rate_limits(
            model,
            environ={"RAP_TPM": "1000", "RAP_RPM": "10"},
            config_path=partial_cfg,
        ) == (15000, 10)

        # 2a. RAP_MODEL_LIMITS beats models.json
        env_json = {
            "RAP_TPM": "1000",
            "RAP_RPM": "10",
            "RAP_MODEL_LIMITS": json.dumps({model: {"tpm": 22000, "rpm": 44}}),
        }
        assert resolve_model_rate_limits(
            model, environ=env_json, config_path=cfg_file
        ) == (22000, 44)

        # 2b. Per-model env vars beat RAP_MODEL_LIMITS and models.json
        env_per_model = {
            **env_json,
            "RAP_TPM_GEMINI_GEMMA_4_26B_A4B_IT": "33000",
            "RAP_RPM_GEMINI_GEMMA_4_26B_A4B_IT": "55",
        }
        assert resolve_model_rate_limits(
            model, environ=env_per_model, config_path=cfg_file
        ) == (33000, 55)

        # 1. Explicit args beat everything
        assert resolve_model_rate_limits(
            model,
            tokens_per_minute=5000,
            requests_per_minute=5,
            environ=env_per_model,
            config_path=cfg_file,
        ) == (5000, 5)

    def test_litellm_client_auto_configures_rate_limiter_from_rap_models_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LiteLLMClient picks up tpm/rpm from RAP_MODELS_CONFIG models.json automatically."""
        cfg_file = _write_models_json(tmp_path / "models.json")
        monkeypatch.setenv("RAP_MODELS_CONFIG", str(cfg_file))

        client = LiteLLMClient(model="gemini/gemma-4-26b-a4b-it")
        assert client.rate_limiter is not None
        assert client.rate_limiter.tokens_per_minute == 15000
        assert client.rate_limiter.requests_per_minute == 30

        # Explicit constructor args override models.json
        override_client = LiteLLMClient(
            model="gemini/gemma-4-26b-a4b-it",
            tokens_per_minute=4000,
            requests_per_minute=8,
        )
        assert override_client.rate_limiter is not None
        assert override_client.rate_limiter.tokens_per_minute == 4000
        assert override_client.rate_limiter.requests_per_minute == 8


# ============================================================================
# 3. CLI Integration Tests (`--config`, `rap models`, Alias Resolution)
# ============================================================================


class TestModelsConfigCLIIntegration:
    """Tests --config flag, `rap models` subcommand, alias resolution, and overrides."""

    @pytest.mark.parametrize("subcommand", ["run", "ui", "compare", "models"])
    def test_parser_supports_config_flag_and_preserves_mock_default(
        self, subcommand: str
    ) -> None:
        """build_parser() supports --config on run/ui/compare/models and keeps --backend mock default."""
        parser = build_parser()
        args = parser.parse_args([subcommand, "--config", "/tmp/models.json"])
        assert args.config == "/tmp/models.json"

        run_default = parser.parse_args(["run"])
        assert run_default.backend == "mock"
        assert run_default.config is None

    def test_cli_run_without_model_keeps_mock_backward_compatibility(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """main(['run']) without --model uses MockLLM even when models.json exists in CWD."""
        _write_models_json(tmp_path / "models.json")
        monkeypatch.chdir(tmp_path)

        created_llms: list[Any] = []
        orig_create = _create_llm

        def spy_create(*args: Any, **kwargs: Any) -> Any:
            llm = orig_create(*args, **kwargs)
            created_llms.append(llm)
            return llm

        with patch("rap_demo.cli._create_llm", side_effect=spy_create):
            rc = main(["run", "--iterations", "1", "--no-save"])

        assert rc == 0
        assert len(created_llms) == 1
        assert isinstance(created_llms[0], MockLLM)

    def test_unresolved_model_defaults_to_litellm_not_mock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`--model` with no models.json profile must not silently fall back to MockLLM.

        Regression: `rap ui --model ollama\\qwen3.5:9b` from a directory without
        models.json produced P(Yes)=P(No)=0.5 and uniform r1 values because the
        backend stayed at its "mock" default.
        """
        monkeypatch.chdir(tmp_path)  # no models.json here

        parser = build_parser()
        args = parser.parse_args(["ui", "--model", "ollama\\qwen3.5:9b"])
        resolved = _resolve_execution_llm_config(args)

        assert resolved["backend"] == "litellm"
        assert resolved["model"] == "ollama/qwen3.5:9b"
        assert resolved["notices"]

    def test_explicit_mock_backend_is_respected_for_unresolved_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit `--backend mock` keeps MockLLM even when the model is unknown."""
        monkeypatch.chdir(tmp_path)

        parser = build_parser()
        args = parser.parse_args(["run", "--model", "some-model", "--backend", "mock"])
        resolved = _resolve_execution_llm_config(args)

        assert resolved["backend"] == "mock"
        assert resolved["notices"]

    def test_cli_models_subcommand_lists_configured_models_and_handles_empty(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """rap models prints Rich table of models.json entries and handles empty config gracefully."""
        # Rich shrinks table cells to the detected terminal width; pin a wide console.
        monkeypatch.setenv("COLUMNS", "200")

        # 1. Empty config in CWD
        empty_cfg = tmp_path / "empty.json"
        empty_cfg.write_text("{}", encoding="utf-8")
        assert main(["models", "--config", str(empty_cfg)]) == 0
        empty_out = capsys.readouterr().out.lower()
        assert "no" in empty_out or "empty" in empty_out or "0" in empty_out

        # 2. Populated config
        cfg_file = _write_models_json(tmp_path / "models.json")
        assert main(["models", "--config", str(cfg_file)]) == 0
        out = capsys.readouterr().out
        assert "gemma4-26b" in out
        assert "gemini/gemma-4-26b-a4b-it" in out
        assert "15000" in out
        assert "30" in out
        assert "gpt4o-mini" in out

        # 3. Missing explicit --config -> exit 1
        assert main(["models", "--config", str(tmp_path / "missing.json")]) == 1

    def test_cli_run_resolves_model_alias_without_explicit_backend(
        self, tmp_path: Path
    ) -> None:
        """rap run --model gemma4-26b resolves backend, full model ID, tpm, rpm, timeout from models.json."""
        litellm = _import_litellm()
        cfg_file = _write_models_json(tmp_path / "models.json")

        created_llms: list[LiteLLMClient] = []
        orig_create = _create_llm

        def spy_create(*args: Any, **kwargs: Any) -> Any:
            llm = orig_create(*args, **kwargs)
            created_llms.append(llm)
            return llm

        def fake_completion(**kwargs: Any) -> SimpleNamespace:
            if kwargs.get("logprobs") or kwargs.get("max_tokens") == 1:
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="Yes"), logprobs=None)],
                    usage=SimpleNamespace(total_tokens=10),
                )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "thought": "Pick up red",
                                    "action": "pick_up",
                                    "block": "red",
                                    "target": None,
                                }
                            )
                        ),
                        logprobs=None,
                    )
                ],
                usage=SimpleNamespace(total_tokens=20),
            )

        with (
            patch("rap_demo.cli._create_llm", side_effect=spy_create),
            patch.object(litellm, "completion", side_effect=fake_completion) as mock_comp,
        ):
            rc = main(
                [
                    "run",
                    "--model",
                    "gemma4-26b",
                    "--config",
                    str(cfg_file),
                    "--iterations",
                    "1",
                    "--no-save",
                ]
            )

        assert rc == 0
        assert len(created_llms) == 1
        client = created_llms[0]
        assert isinstance(client, LiteLLMClient)
        assert client.model == "gemini/gemma-4-26b-a4b-it"
        assert client.timeout == 60.0
        assert client.rate_limiter is not None
        assert client.rate_limiter.tokens_per_minute == 15000
        assert client.rate_limiter.requests_per_minute == 30
        assert all(c.kwargs.get("model") == "gemini/gemma-4-26b-a4b-it" for c in mock_comp.call_args_list)

    def test_cli_ui_resolves_model_alias_without_explicit_backend(
        self, tmp_path: Path
    ) -> None:
        """rap ui --model gemma4-26b resolves profile from models.json and passes configured LLM."""
        cfg_file = _write_models_json(tmp_path / "models.json")

        created_llms: list[Any] = []
        orig_create = _create_llm

        def spy_create(*args: Any, **kwargs: Any) -> Any:
            llm = orig_create(*args, **kwargs)
            created_llms.append(llm)
            return llm

        with (
            patch("rap_demo.cli._create_llm", side_effect=spy_create),
            patch("uvicorn.run") as mock_uvicorn,
        ):
            rc = main(["ui", "--model", "gemma4-26b", "--config", str(cfg_file)])

        assert rc == 0
        mock_uvicorn.assert_called_once()
        assert len(created_llms) == 1
        client = created_llms[0]
        assert isinstance(client, LiteLLMClient)
        assert client.model == "gemini/gemma-4-26b-a4b-it"
        assert client.rate_limiter is not None
        assert client.rate_limiter.tokens_per_minute == 15000
        assert client.rate_limiter.requests_per_minute == 30

    def test_cli_explicit_flags_override_models_json_profile(
        self, tmp_path: Path
    ) -> None:
        """Explicit --backend, --tpm, --rpm, --api-base override models.json profile values."""
        litellm = _import_litellm()
        cfg_file = _write_models_json(tmp_path / "models.json")

        created_llms: list[Any] = []
        orig_create = _create_llm

        def spy_create(*args: Any, **kwargs: Any) -> Any:
            llm = orig_create(*args, **kwargs)
            created_llms.append(llm)
            return llm

        def fake_completion(**kwargs: Any) -> SimpleNamespace:
            if kwargs.get("logprobs") or kwargs.get("max_tokens") == 1:
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="Yes"), logprobs=None)],
                    usage=SimpleNamespace(total_tokens=10),
                )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "thought": "Pick up red",
                                    "action": "pick_up",
                                    "block": "red",
                                    "target": None,
                                }
                            )
                        ),
                        logprobs=None,
                    )
                ],
                usage=SimpleNamespace(total_tokens=20),
            )

        with (
            patch("rap_demo.cli._create_llm", side_effect=spy_create),
            patch.object(litellm, "completion", side_effect=fake_completion) as mock_comp,
        ):
            rc = main(
                [
                    "run",
                    "--model",
                    "gemma4-26b",
                    "--config",
                    str(cfg_file),
                    "--backend",
                    "litellm",
                    "--tpm",
                    "5000",
                    "--rpm",
                    "10",
                    "--api-base",
                    "https://override.example.com/v1",
                    "--iterations",
                    "1",
                    "--no-save",
                ]
            )

        assert rc == 0
        client = created_llms[0]
        assert isinstance(client, LiteLLMClient)
        assert client.model == "gemini/gemma-4-26b-a4b-it"
        assert client.api_base == "https://override.example.com/v1"
        assert client.rate_limiter is not None
        assert client.rate_limiter.tokens_per_minute == 5000
        assert client.rate_limiter.requests_per_minute == 10
        assert all(
            c.kwargs.get("api_base") == "https://override.example.com/v1"
            for c in mock_comp.call_args_list
        )

    def test_cli_forwards_reasoning_effort_from_profile(self, tmp_path: Path) -> None:
        """`reasoning_effort` in models.json reaches the LiteLLM client (needed by reasoning models)."""
        cfg_file = _write_models_json(
            tmp_path / "models.json",
            {
                "models": {
                    "reasoning-model": {
                        "backend": "litellm",
                        "model": "ollama/gpt-oss:20b",
                        "api_base": "http://localhost:11434",
                        "reasoning_effort": "low",
                    }
                }
            },
        )

        created_llms: list[Any] = []
        orig_create = _create_llm

        def spy_create(*args: Any, **kwargs: Any) -> Any:
            llm = orig_create(*args, **kwargs)
            created_llms.append(llm)
            return llm

        with (
            patch("rap_demo.cli._create_llm", side_effect=spy_create),
            patch("uvicorn.run") as mock_uvicorn,
        ):
            rc = main(["ui", "--model", "reasoning-model", "--config", str(cfg_file)])

        assert rc == 0
        mock_uvicorn.assert_called_once()
        client = created_llms[0]
        assert isinstance(client, LiteLLMClient)
        assert client.model == "ollama/gpt-oss:20b"
        assert client.reasoning_effort == "low"
