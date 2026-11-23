"""LangGraph node wrappers for virtual team agents, HITL gate, and terminal states."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

from langgraph.types import interrupt
from pydantic import ValidationError

from crewforge.agents.factory import AgentFactory
from crewforge.config.schemas import AgentRole
from crewforge.state.artifacts import QAReport, RunStatus
from crewforge.state.project_state import ExecutionLogEntry, ProjectState


class GraphNode(Protocol):
    """Callable node signature compatible with LangGraph StateGraph.add_node."""

    def __call__(self, state: ProjectState) -> dict[str, Any]: ...


def _log_entry(role: str, action: str, summary: str) -> ExecutionLogEntry:
    return ExecutionLogEntry(
        role=role,
        action=action,
        summary=summary,
        timestamp=datetime.now(tz=UTC).isoformat(),
    )


def make_agent_node(role: AgentRole, factory: AgentFactory) -> GraphNode:
    """Create a LangGraph node that runs a single configured TeamAgent."""

    def _node(state: ProjectState) -> dict[str, Any]:
        agent = factory.create_agent(role)
        result = agent.invoke(state)
        summary = (
            f"Completed with {result.tool_calls_made} tool call(s); "
            f"artifacts={sorted(result.written_artifacts)}"
        )
        return {
            "status": RunStatus.RUNNING.value,
            "current_stage": role.value,
            "artifacts": result.written_artifacts,
            "execution_log": [_log_entry(role.value, "completed", summary)],
        }

    return _node


def make_hitl_gate_node(*, enabled: bool) -> GraphNode:
    """Create the Human-in-the-Loop approval gate node."""

    def _hitl_node(state: ProjectState) -> dict[str, Any]:
        if not enabled:
            return {
                "status": RunStatus.RUNNING.value,
                "current_stage": "hitl_plan_gate",
                "hitl_approved": True,
                "hitl_feedback": "",
                "execution_log": [
                    _log_entry("hitl_plan_gate", "auto_approved", "HITL gate disabled in config.")
                ],
            }

        payload = {
            "gate": "plan_approval",
            "feature_request": state.get("feature_request", ""),
            "artifacts": state.get("artifacts", {}),
            "message": "Review planning artifacts in .ai-team/ and approve or request changes.",
        }
        decision = interrupt(payload)

        approved = False
        feedback = ""
        if isinstance(decision, dict):
            approved = bool(decision.get("approved", False))
            feedback = str(decision.get("feedback", "")).strip()
        elif isinstance(decision, bool):
            approved = decision

        action = "approved" if approved else "revision_requested"
        summary = "Human approved plan." if approved else f"Changes requested: {feedback}"
        return {
            "status": RunStatus.RUNNING.value,
            "current_stage": "hitl_plan_gate",
            "hitl_approved": approved,
            "hitl_feedback": "" if approved else feedback,
            "execution_log": [_log_entry("hitl_plan_gate", action, summary)],
        }

    return _hitl_node


def make_qa_node(factory: AgentFactory) -> GraphNode:
    """Create the QA Reviewer node that parses and validates `06_qa_report.json`."""

    def _qa_node(state: ProjectState) -> dict[str, Any]:
        agent = factory.create_agent(AgentRole.QA_REVIEWER)
        result = agent.invoke(state)

        qa_passed = False
        qa_report_dict: dict[str, Any] | None = None
        errors: list[str] = []

        report_path = factory.workspace_ctx.resolve_artifact_path("06_qa_report.json")
        if not report_path.is_file():
            errors.append("QA Reviewer did not write 06_qa_report.json.")
        else:
            try:
                report = QAReport.model_validate_json(report_path.read_text(encoding="utf-8"))
                qa_passed = report.passed
                qa_report_dict = report.model_dump(mode="json")
            except (OSError, ValidationError, ValueError) as exc:
                errors.append(f"Invalid 06_qa_report.json: {exc}")

        current_retries = state.get("qa_retry_count", 0)
        next_retries = current_retries if qa_passed else current_retries + 1
        summary = "QA passed." if qa_passed else f"QA failed (attempt {next_retries})."

        update: dict[str, Any] = {
            "status": RunStatus.RUNNING.value,
            "current_stage": AgentRole.QA_REVIEWER.value,
            "artifacts": result.written_artifacts,
            "qa_passed": qa_passed,
            "qa_retry_count": next_retries,
            "qa_report": qa_report_dict,
            "execution_log": [_log_entry(AgentRole.QA_REVIEWER.value, "reviewed", summary)],
        }
        if errors:
            update["errors"] = errors
        return update

    return _qa_node


def complete_node(state: ProjectState) -> dict[str, Any]:
    """Terminal node for a successfully verified feature run."""
    return {
        "status": RunStatus.COMPLETED.value,
        "current_stage": "completed",
        "execution_log": [_log_entry("orchestrator", "completed", "Workflow finished cleanly.")],
    }


def escalate_node(state: ProjectState) -> dict[str, Any]:
    """Terminal node when QA fails and reaches `max_qa_retries`."""
    retries = state.get("qa_retry_count", 0)
    max_retries = state.get("max_qa_retries", 3)
    msg = f"Escalated to human after {retries}/{max_retries} failed QA cycles."
    return {
        "status": RunStatus.ESCALATED.value,
        "current_stage": "escalated",
        "execution_log": [_log_entry("orchestrator", "escalated", msg)],
        "errors": [msg],
    }