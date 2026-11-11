"""Configuration registry and loader for models.json profiles."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def get_models_config_path(config_path: str | Path | None = None) -> Path:
    """Resolve models.json path (explicit arg -> RAP_MODELS_CONFIG -> Path('models.json'))."""
    if config_path is not None:
        if isinstance(config_path, Path) or config_path.strip():
            return Path(config_path)

    env_path = os.environ.get("RAP_MODELS_CONFIG")
    if env_path and env_path.strip():
        return Path(env_path.strip())

    return Path("models.json")


def _validate_positive_int(value: Any, field_name: str) -> int | None:
    """Validate optional positive integer, explicitly rejecting booleans and <= 0."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Invalid {field_name}={value!r}: expected a positive integer.")
    return value


def _validate_positive_float(value: Any, field_name: str) -> float | None:
    """Validate optional positive number, explicitly rejecting booleans and <= 0."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid {field_name}={value!r}: expected a positive number.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"Invalid {field_name}={value!r}: must be greater than 0.")
    return numeric


@dataclass
class ModelProfile:
    """Named model configuration profile parsed from models.json."""

    alias: str
    backend: str = "litellm"
    model: str | None = None
    api_base: str | None = None
    tpm: int | None = None
    rpm: int | None = None
    timeout: float | None = None
    mode: str | None = None
    iterations: int | None = None
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.alias, str) or not self.alias.strip():
            raise ValueError(f"Invalid ModelProfile alias: {self.alias!r}")
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError(f"Invalid ModelProfile backend: {self.backend!r}")
        self.tpm = _validate_positive_int(self.tpm, "tpm")
        self.rpm = _validate_positive_int(self.rpm, "rpm")
        self.timeout = _validate_positive_float(self.timeout, "timeout")
        self.iterations = _validate_positive_int(self.iterations, "iterations")


@dataclass
class ModelsConfig:
    """Top-level models.json configuration registry."""

    default_model: str | None = None
    defaults: dict[str, Any] = field(default_factory=dict)
    models: dict[str, ModelProfile] = field(default_factory=dict)

    def resolve_model(self, name_or_alias: str) -> ModelProfile | None:
        """Resolve a model alias, full model ID, or short name to a ModelProfile."""
        if not isinstance(name_or_alias, str) or not name_or_alias.strip():
            return None

        query = name_or_alias.strip()
        query_lower = query.lower()

        # 1. Exact alias key
        if query in self.models:
            return self.models[query]

        # 2. Exact profile.model
        for profile in self.models.values():
            if profile.model == query:
                return profile

        # 3. Case-insensitive alias or profile.model
        for alias, profile in self.models.items():
            if alias.lower() == query_lower:
                return profile
        for profile in self.models.values():
            if profile.model is not None and profile.model.lower() == query_lower:
                return profile

        # 4. Short name after '/' (exact then case-insensitive)
        for profile in self.models.values():
            if profile.model and "/" in profile.model:
                short_name = profile.model.rsplit("/", 1)[-1]
                if short_name == query:
                    return profile
        for profile in self.models.values():
            if profile.model and "/" in profile.model:
                short_name = profile.model.rsplit("/", 1)[-1]
                if short_name.lower() == query_lower:
                    return profile

        return None


def load_models_config(config_path: str | Path | None = None) -> ModelsConfig:
    """Load and validate models.json into a ModelsConfig instance.

    Raises:
        FileNotFoundError: If an explicit config_path does not exist.
        ValueError: If JSON is malformed or violates the models.json schema.
    """
    is_explicit = config_path is not None and (
        isinstance(config_path, Path) or bool(config_path.strip())
    )
    resolved_path = get_models_config_path(config_path)

    if not resolved_path.is_file():
        if is_explicit:
            raise FileNotFoundError(f"Models configuration file not found: {resolved_path}")
        return ModelsConfig()

    try:
        raw = json.loads(resolved_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in models configuration '{resolved_path}': {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to read models configuration '{resolved_path}': {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(
            f"Invalid models configuration '{resolved_path}': root JSON value must be an object."
        )

    default_model = raw.get("default_model")
    if default_model is not None and not isinstance(default_model, str):
        raise ValueError("Invalid 'default_model': expected a string or null.")

    raw_defaults = raw.get("defaults", {})
    if raw_defaults is None:
        raw_defaults = {}
    if not isinstance(raw_defaults, dict):
        raise ValueError("Invalid 'defaults': expected a JSON object.")

    raw_models = raw.get("models", {})
    if raw_models is None:
        raw_models = {}
    if not isinstance(raw_models, dict):
        raise ValueError("Invalid 'models': expected a JSON object mapping aliases to profiles.")

    models: dict[str, ModelProfile] = {}
    for alias, entry in raw_models.items():
        if not isinstance(alias, str) or not isinstance(entry, dict):
            raise ValueError(
                f"Invalid model entry for {alias!r}: expected an object of profile attributes."
            )
        models[alias] = ModelProfile(
            alias=alias,
            backend=entry.get("backend", "litellm"),
            model=entry.get("model"),
            api_base=entry.get("api_base"),
            tpm=entry.get("tpm"),
            rpm=entry.get("rpm"),
            timeout=entry.get("timeout"),
            mode=entry.get("mode"),
            iterations=entry.get("iterations"),
            reasoning_effort=entry.get("reasoning_effort"),
        )

    return ModelsConfig(
        default_model=default_model,
        defaults=dict(raw_defaults),
        models=models,
    )
