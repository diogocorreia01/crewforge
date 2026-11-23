"""Pydantic v2 configuration schemas for CrewForge."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelProvider(StrEnum):
    """Supported LLM backend providers."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    FAKE = "fake"


class AgentRole(StrEnum):
    """Canonical roles in the CrewForge virtual engineering team."""

    BUSINESS_ANALYST = "business_analyst"
    UI_UX_DESIGNER = "ui_ux_designer"
    TECH_ARCHITECT = "tech_architect"
    BACKEND_DEVELOPER = "backend_developer"
    FRONTEND_DEVELOPER = "frontend_developer"
    QA_REVIEWER = "qa_reviewer"


class CheckpointerBackend(StrEnum):
    """Supported LangGraph checkpoint persistence backends."""

    SQLITE = "sqlite"
    MEMORY = "memory"


class ModelProfileConfig(BaseModel):
    """Configuration for a single named LLM profile."""

    model_config = ConfigDict(extra="forbid")

    provider: ModelProvider
    model: str = Field(..., min_length=1, description="Provider model identifier.")
    base_url: str | None = Field(
        default=None,
        description="Optional custom endpoint URL (e.g., Ollama server URL).",
    )
    api_key_env: str | None = Field(
        default=None,
        description="Environment variable holding the provider API key, if required.",
    )
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=128)
    timeout_seconds: int = Field(default=120, ge=5)
    extra_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provider-specific keyword arguments.",
    )


class ModelsRegistryConfig(BaseModel):
    """Registry of named model profiles and the default profile."""

    model_config = ConfigDict(extra="forbid")

    default_profile: str = Field(..., min_length=1)
    profiles: dict[str, ModelProfileConfig] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_default_profile_exists(self) -> ModelsRegistryConfig:
        if self.default_profile not in self.profiles:
            available = ", ".join(sorted(self.profiles))
            msg = (
                f"default_profile '{self.default_profile}' not found in profiles "
                f"(available: {available})"
            )
            raise ValueError(msg)
        return self


class AgentConfig(BaseModel):
    """Config-driven persona, artifact contract, and tool allowlist for an agent."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    model_profile: str = Field(..., min_length=1)
    reads_artifacts: list[str] = Field(default_factory=list)
    writes_artifacts: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    system_prompt: str = Field(..., min_length=1)
    max_iterations: int = Field(default=15, ge=1, le=100)


class AgentsRegistryConfig(BaseModel):
    """Registry mapping each virtual team role to its AgentConfig."""

    model_config = ConfigDict(extra="forbid")

    agents: dict[AgentRole, AgentConfig] = Field(..., min_length=1)

    def get_agent(self, role: AgentRole) -> AgentConfig:
        """Return the configuration for a given role."""
        return self.agents[role]


class CheckpointerConfig(BaseModel):
    """Persistence configuration for LangGraph checkpoints."""

    model_config = ConfigDict(extra="forbid")

    backend: CheckpointerBackend = CheckpointerBackend.SQLITE
    sqlite_filename: str = Field(default="checkpoints.sqlite", min_length=1)


class WorkflowConfig(BaseModel):
    """Orchestration topology and safety settings."""

    model_config = ConfigDict(extra="forbid")

    artifact_dir_name: str = Field(default=".ai-team", min_length=1)
    max_qa_retries: int = Field(default=3, ge=0, le=10)
    enable_hitl_plan_gate: bool = True
    parallel_design_and_arch: bool = False
    parallel_dev: bool = False
    checkpointer: CheckpointerConfig = Field(default_factory=CheckpointerConfig)


class CrewForgeConfig(BaseModel):
    """Top-level validated configuration bundle for CrewForge."""

    model_config = ConfigDict(extra="forbid")

    models: ModelsRegistryConfig
    agents: AgentsRegistryConfig
    workflow: WorkflowConfig

    @model_validator(mode="after")
    def _validate_agent_model_profiles(self) -> CrewForgeConfig:
        missing: list[str] = []
        for role, agent in self.agents.agents.items():
            if agent.model_profile not in self.models.profiles:
                missing.append(f"{role.value} -> '{agent.model_profile}'")
        if missing:
            available = ", ".join(sorted(self.models.profiles))
            msg = (
                f"Agents reference unknown model profiles: {', '.join(missing)} "
                f"(available: {available})"
            )
            raise ValueError(msg)
        return self

    def resolve_model_for_agent(self, role: AgentRole) -> ModelProfileConfig:
        """Resolve the ModelProfileConfig assigned to the specified agent role."""
        agent = self.agents.get_agent(role)
        return self.models.profiles[agent.model_profile]