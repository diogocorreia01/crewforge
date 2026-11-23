"""Pure conditional routing functions for LangGraph edges."""

from __future__ import annotations

from typing import Literal

from crewforge.state.project_state import ProjectState


def route_after_hitl(
    state: ProjectState,
) -> Literal["backend_developer", "business_analyst"]:
    """Route to implementation if approved, or back to Business Analyst if changes requested."""
    if state.get("hitl_approved", False):
        return "backend_developer"
    return "business_analyst"


def route_after_qa(
    state: ProjectState,
) -> Literal["complete", "backend_developer", "escalate"]:
    """Route based on QA verdict and bounded retry counter."""
    if state.get("qa_passed", False):
        return "complete"

    retries = state.get("qa_retry_count", 0)
    max_retries = state.get("max_qa_retries", 3)
    if retries < max_retries:
        return "backend_developer"
    return "escalate"