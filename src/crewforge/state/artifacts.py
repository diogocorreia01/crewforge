"""Structured artifact, QA report, and HITL domain models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from crewforge.config.schemas import AgentRole


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


class RunStatus(StrEnum):
    """High-level lifecycle status of an orchestrator run."""

    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"


class HITLDecisionStatus(StrEnum):
    """Decision submitted by a human reviewer at an approval gate."""

    APPROVED = "approved"
    REJECTED = "rejected"
    REVISION_REQUESTED = "revision_requested"


class HITLDecision(BaseModel):
    """Structured human-in-the-loop gate decision."""

    model_config = ConfigDict(extra="forbid")

    status: HITLDecisionStatus
    feedback: str = ""
    decided_at: datetime = Field(default_factory=_utc_now)


class IssueSeverity(StrEnum):
    """Severity level for QA review findings."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class QAIssue(BaseModel):
    """Single actionable finding raised by the QA Reviewer."""

    model_config = ConfigDict(extra="forbid")

    severity: IssueSeverity
    owner: AgentRole
    file_path: str | None = None
    description: str = Field(..., min_length=1)
    suggested_fix: str = ""


class QAReport(BaseModel):
    """Structured QA verdict persisted to `.ai-team/06_qa_report.json`."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    summary: str = Field(..., min_length=1)
    checks_run: list[str] = Field(default_factory=list)
    issues: list[QAIssue] = Field(default_factory=list)


class ArtifactEntry(BaseModel):
    """Metadata for a single artifact stored inside `.ai-team/`."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    relative_path: str = Field(..., min_length=1)
    producer: AgentRole
    revision: int = Field(default=1, ge=1)
    sha256: str = Field(..., min_length=1)
    updated_at: datetime = Field(default_factory=_utc_now)


class ArtifactManifest(BaseModel):
    """Manifest persisted at `<workspace>/.ai-team/manifest.json`."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., min_length=1)
    feature_request: str = Field(..., min_length=1)
    workspace_path: str = Field(..., min_length=1)
    artifacts: dict[str, ArtifactEntry] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=_utc_now)

    def record_artifact(
        self,
        *,
        name: str,
        relative_path: str,
        producer: AgentRole,
        sha256: str,
    ) -> ArtifactEntry:
        """Insert or bump the revision of an artifact in the manifest."""
        now = _utc_now()
        existing = self.artifacts.get(name)
        revision = existing.revision + 1 if existing is not None else 1
        entry = ArtifactEntry(
            name=name,
            relative_path=relative_path,
            producer=producer,
            revision=revision,
            sha256=sha256,
            updated_at=now,
        )
        self.artifacts[name] = entry
        self.updated_at = now
        return entry