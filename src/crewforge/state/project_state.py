"""LangGraph ProjectState TypedDict schema and reducers."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from crewforge.state.artifacts import RunStatus


def _merge_str_dicts(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    """Merge artifact path dictionaries cleanly across sequential or parallel nodes."""
    merged = dict(left)
    merged.update(right)
    return merged


class ExecutionLogEntry(TypedDict):
    """Compact audit record emitted by each graph node."""

    role: str
    action: str
    summary: str
    timestamp: str


class ProjectState(TypedDict, total=False):
    """Shared LangGraph state for a CrewForge feature run.

    Designed to remain lightweight: heavy documents live as files in
    `<workspace>/.ai-team/` while this state tracks artifact pointers,
    gate decisions, QA loop counters, and compact audit entries.
    """

    run_id: str
    workspace_path: str
    artifact_dir: str
    feature_request: str
    status: str
    current_stage: str

    # Maps artifact filename (e.g. "01_prd.md") -> relative path inside workspace
    artifacts: Annotated[dict[str, str], _merge_str_dicts]

    # Human-in-the-Loop approval gate state
    hitl_approved: bool
    hitl_feedback: str

    # QA verification loop state
    qa_passed: bool
    qa_retry_count: int
    max_qa_retries: int
    qa_report: dict[str, Any] | None

    # Append-only audit trail and error list
    execution_log: Annotated[list[ExecutionLogEntry], operator.add]
    errors: Annotated[list[str], operator.add]


def create_initial_state(
    *,
    run_id: str,
    workspace_path: str,
    artifact_dir: str,
    feature_request: str,
    max_qa_retries: int = 3,
) -> ProjectState:
    """Create a normalized initial ProjectState for a new workflow run."""
    return ProjectState(
        run_id=run_id,
        workspace_path=workspace_path,
        artifact_dir=artifact_dir,
        feature_request=feature_request,
        status=RunStatus.PENDING.value,
        current_stage="init",
        artifacts={},
        hitl_approved=False,
        hitl_feedback="",
        qa_passed=False,
        qa_retry_count=0,
        max_qa_retries=max_qa_retries,
        qa_report=None,
        execution_log=[],
        errors=[],
    )