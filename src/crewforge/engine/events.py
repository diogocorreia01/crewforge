"""Typed run outcomes and lifecycle models shared by CLI and future web APIs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from crewforge.state.artifacts import RunStatus


class RunOutcome(BaseModel):
    """Serializable snapshot of a CrewForge workflow run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: RunStatus
    current_stage: str
    interrupted: bool = False
    interrupt_payload: dict[str, Any] | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    qa_passed: bool = False
    qa_retry_count: int = 0
    errors: list[str] = Field(default_factory=list)