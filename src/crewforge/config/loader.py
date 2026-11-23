"""YAML configuration loader with env-var interpolation and workspace overrides."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from crewforge.config.schemas import (
    AgentsRegistryConfig,
    CrewForgeConfig,
    ModelsRegistryConfig,
    WorkflowConfig,
)

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


class ConfigLoadError(Exception):
    """Raised when CrewForge YAML configuration cannot be loaded or validated."""


def default_config_dir() -> Path:
    """Resolve the default `config/` directory shipped with CrewForge."""
    repo_config = Path(__file__).resolve().parents[3] / "config"
    if repo_config.is_dir():
        return repo_config
    cwd_config = Path.cwd() / "config"
    if cwd_config.is_dir():
        return cwd_config
    msg = f"Could not locate default config directory (checked {repo_config} and {cwd_config})."
    raise ConfigLoadError(msg)


def _interpolate_string(raw: str, env: Mapping[str, str]) -> str:
    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default_val = match.group(2)
        if var_name in env:
            return env[var_name]
        if default_val is not None:
            return default_val
        msg = f"Environment variable '{var_name}' is referenced in config but not set."
        raise ConfigLoadError(msg)

    return _ENV_VAR_PATTERN.sub(_replace, raw)


def _interpolate_env_vars(value: Any, env: Mapping[str, str]) -> Any:
    """Recursively expand `${VAR}` and `${VAR:-default}` placeholders."""
    if isinstance(value, str):
        return _interpolate_string(value, env)
    if isinstance(value, list):
        return [_interpolate_env_vars(item, env) for item in value]
    if isinstance(value, dict):
        return {k: _interpolate_env_vars(v, env) for k, v in value.items()}
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override mapping onto base mapping without mutating inputs."""
    merged: dict[str, Any] = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        msg = f"Configuration file not found: {path}"
        raise ConfigLoadError(msg)
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"Invalid YAML syntax in {path}: {exc}"
        raise ConfigLoadError(msg) from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        msg = f"Expected a YAML mapping at top level of {path}, got {type(loaded).__name__}."
        raise ConfigLoadError(msg)
    return loaded


def _load_with_optional_override(
    *,
    base_dir: Path,
    override_dir: Path | None,
    filename: str,
    env: Mapping[str, str],
) -> dict[str, Any]:
    data = _read_yaml_mapping(base_dir / filename)
    if override_dir is not None:
        override_path = override_dir / filename
        if override_path.is_file():
            override_data = _read_yaml_mapping(override_path)
            data = _deep_merge(data, override_data)
    interpolated = _interpolate_env_vars(data, env)
    if not isinstance(interpolated, dict):
        msg = f"Interpolated config for {filename} must be a mapping."
        raise ConfigLoadError(msg)
    return interpolated


def load_config(
    *,
    config_dir: Path | None = None,
    workspace_dir: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> CrewForgeConfig:
    """Load and validate CrewForge YAML configuration files."""
    base_dir = config_dir if config_dir is not None else default_config_dir()
    env_map: Mapping[str, str] = os.environ if env is None else env

    initial_workflow_raw = _load_with_optional_override(
        base_dir=base_dir,
        override_dir=None,
        filename="workflow.yaml",
        env=env_map,
    )
    artifact_dir_name = str(initial_workflow_raw.get("artifact_dir_name", ".ai-team"))
    override_dir = (
        workspace_dir / artifact_dir_name / "config" if workspace_dir is not None else None
    )

    workflow_raw = _load_with_optional_override(
        base_dir=base_dir,
        override_dir=override_dir,
        filename="workflow.yaml",
        env=env_map,
    )
    models_raw = _load_with_optional_override(
        base_dir=base_dir,
        override_dir=override_dir,
        filename="models.yaml",
        env=env_map,
    )
    agents_raw = _load_with_optional_override(
        base_dir=base_dir,
        override_dir=override_dir,
        filename="agents.yaml",
        env=env_map,
    )

    try:
        models_cfg = ModelsRegistryConfig.model_validate(models_raw)
        agents_cfg = AgentsRegistryConfig.model_validate(agents_raw)
        workflow_cfg = WorkflowConfig.model_validate(workflow_raw)
        return CrewForgeConfig(
            models=models_cfg,
            agents=agents_cfg,
            workflow=workflow_cfg,
        )
    except ValidationError as exc:
        msg = f"Configuration validation failed:\n{exc}"
        raise ConfigLoadError(msg) from exc