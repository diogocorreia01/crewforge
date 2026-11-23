"""Unit tests for pure LangGraph conditional routing functions."""

from __future__ import annotations

from crewforge.graph.routing import route_after_hitl, route_after_qa
from crewforge.state import ProjectState


def test_route_after_hitl() -> None:
    assert route_after_hitl(ProjectState(hitl_approved=True)) == "backend_developer"
    assert route_after_hitl(ProjectState(hitl_approved=False)) == "business_analyst"


def test_route_after_qa() -> None:
    assert route_after_qa(ProjectState(qa_passed=True, qa_retry_count=0, max_qa_retries=3)) == (
        "complete"
    )
    assert route_after_qa(ProjectState(qa_passed=False, qa_retry_count=1, max_qa_retries=3)) == (
        "backend_developer"
    )
    assert route_after_qa(ProjectState(qa_passed=False, qa_retry_count=3, max_qa_retries=3)) == (
        "escalate"
    )