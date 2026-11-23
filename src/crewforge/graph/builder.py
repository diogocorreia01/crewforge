"""LangGraph StateGraph assembly for CrewForge."""

from __future__ import annotations

from typing import Any, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from crewforge.agents.factory import AgentFactory
from crewforge.config.schemas import AgentRole, CrewForgeConfig
from crewforge.graph.nodes import (
    complete_node,
    escalate_node,
    make_agent_node,
    make_hitl_gate_node,
    make_qa_node,
)
from crewforge.graph.routing import route_after_hitl, route_after_qa
from crewforge.state.project_state import ProjectState


def build_workflow_graph(
    *,
    config: CrewForgeConfig,
    agent_factory: AgentFactory,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Assemble and compile the CrewForge LangGraph workflow."""
    graph: StateGraph[Any] = StateGraph(cast(Any, ProjectState))

    # 1. Register nodes
    graph.add_node("business_analyst", make_agent_node(AgentRole.BUSINESS_ANALYST, agent_factory))
    graph.add_node("ui_ux_designer", make_agent_node(AgentRole.UI_UX_DESIGNER, agent_factory))
    graph.add_node("tech_architect", make_agent_node(AgentRole.TECH_ARCHITECT, agent_factory))
    graph.add_node(
        "hitl_plan_gate",
        make_hitl_gate_node(enabled=config.workflow.enable_hitl_plan_gate),
    )
    graph.add_node("backend_developer", make_agent_node(AgentRole.BACKEND_DEVELOPER, agent_factory))
    graph.add_node(
        "frontend_developer", make_agent_node(AgentRole.FRONTEND_DEVELOPER, agent_factory)
    )
    graph.add_node("qa_reviewer", make_qa_node(agent_factory))
    graph.add_node("complete", complete_node)
    graph.add_node("escalate", escalate_node)

    # 2. Planning phase edges (sequential by default for local models, optional parallel fan-out)
    graph.add_edge(START, "business_analyst")
    if config.workflow.parallel_design_and_arch:
        graph.add_edge("business_analyst", "ui_ux_designer")
        graph.add_edge("business_analyst", "tech_architect")
        graph.add_edge("ui_ux_designer", "hitl_plan_gate")
        graph.add_edge("tech_architect", "hitl_plan_gate")
    else:
        graph.add_edge("business_analyst", "ui_ux_designer")
        graph.add_edge("ui_ux_designer", "tech_architect")
        graph.add_edge("tech_architect", "hitl_plan_gate")

    # 3. Human-in-the-Loop approval gate
    graph.add_conditional_edges(
        "hitl_plan_gate",
        route_after_hitl,
        {
            "backend_developer": "backend_developer",
            "business_analyst": "business_analyst",
        },
    )

    # 4. Implementation & QA verification loop
    graph.add_edge("backend_developer", "frontend_developer")
    graph.add_edge("frontend_developer", "qa_reviewer")
    graph.add_conditional_edges(
        "qa_reviewer",
        route_after_qa,
        {
            "complete": "complete",
            "backend_developer": "backend_developer",
            "escalate": "escalate",
        },
    )

    # 5. Terminal edges
    graph.add_edge("complete", END)
    graph.add_edge("escalate", END)

    return graph.compile(checkpointer=checkpointer)