"""Unit tests for CrewForge YAML configuration loader and state schemas."""

from __future__ import annotations

from pathlib import Path

import pytest

from crewforge.config import (
    AgentRole,
    ConfigLoadError,
    ModelProvider,
    load_config,
)
from crewforge.state import (
    ArtifactManifest,
    RunStatus,
    create_initial_state,
)


def test_load_default_repository_config() -> None:
    cfg = load_config(env={})

    assert cfg.models.default_profile == "local_reasoning"
    assert cfg.models.profiles["local_reasoning"].provider == ModelProvider.OLLAMA
    assert cfg.models.profiles["local_reasoning"].base_url == "http://localhost:11434"
    assert len(cfg.agents.agents) == 6
    assert cfg.resolve_model_for_agent(AgentRole.BACKEND_DEVELOPER).model == "qwen2.5-coder:7b"
    assert cfg.workflow.max_qa_retries == 3
    assert cfg.workflow.enable_hitl_plan_gate is True


def test_env_var_interpolation_overrides_default() -> None:
    cfg = load_config(env={"OLLAMA_BASE_URL": "http://127.0.0.1:11435"})
    assert cfg.models.profiles["local_reasoning"].base_url == "http://127.0.0.1:11435"


def test_workspace_override_deep_merges_config(tmp_path: Path) -> None:
    override_dir = tmp_path / ".ai-team" / "config"
    override_dir.mkdir(parents=True)
    (override_dir / "workflow.yaml").write_text("max_qa_retries: 5\n", encoding="utf-8")
    (override_dir / "agents.yaml").write_text(
        "agents:\n  business_analyst:\n    model_profile: local_fast\n",
        encoding="utf-8",
    )

    cfg = load_config(workspace_dir=tmp_path, env={})

    assert cfg.workflow.max_qa_retries == 5
    assert cfg.workflow.enable_hitl_plan_gate is True
    assert cfg.agents.get_agent(AgentRole.BUSINESS_ANALYST).model_profile == "local_fast"
    assert cfg.agents.get_agent(AgentRole.BUSINESS_ANALYST).name == "Business Analyst"


def test_unknown_agent_model_profile_raises_config_error(tmp_path: Path) -> None:
    override_dir = tmp_path / ".ai-team" / "config"
    override_dir.mkdir(parents=True)
    (override_dir / "agents.yaml").write_text(
        "agents:\n  qa_reviewer:\n    model_profile: nonexistent_model\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigLoadError, match="nonexistent_model"):
        load_config(workspace_dir=tmp_path, env={})


def test_initial_state_and_artifact_manifest() -> None:
    state = create_initial_state(
        run_id="run-001",
        workspace_path="/tmp/demo",
        artifact_dir=".ai-team",
        feature_request="Add healthcheck endpoint",
    )
    assert state["status"] == RunStatus.PENDING.value
    assert state["qa_retry_count"] == 0
    assert state["artifacts"] == {}

    manifest = ArtifactManifest(
        run_id="run-001",
        feature_request="Add healthcheck endpoint",
        workspace_path="/tmp/demo",
    )
    first = manifest.record_artifact(
        name="01_prd.md",
        relative_path=".ai-team/01_prd.md",
        producer=AgentRole.BUSINESS_ANALYST,
        sha256="abc123",
    )
    second = manifest.record_artifact(
        name="01_prd.md",
        relative_path=".ai-team/01_prd.md",
        producer=AgentRole.BUSINESS_ANALYST,
        sha256="def456",
    )
    assert first.revision == 1
    assert second.revision == 2
    assert manifest.artifacts["01_prd.md"].sha256 == "def456"