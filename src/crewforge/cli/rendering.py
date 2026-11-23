"""Rich console rendering helpers for CrewForge CLI."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from crewforge.config import CrewForgeConfig
from crewforge.engine import RunOutcome
from crewforge.state import RunStatus

_STATUS_COLORS: dict[RunStatus, str] = {
    RunStatus.PENDING: "yellow",
    RunStatus.RUNNING: "cyan",
    RunStatus.AWAITING_APPROVAL: "magenta",
    RunStatus.COMPLETED: "green",
    RunStatus.FAILED: "red",
    RunStatus.ESCALATED: "bold red",
}


def render_outcome(outcome: RunOutcome, *, console: Console | None = None) -> None:
    """Display a formatted summary panel for a RunOutcome."""
    out = console or Console()
    color = _STATUS_COLORS.get(outcome.status, "white")

    lines = [
        f"[bold]Run ID:[/bold] {outcome.run_id}",
        f"[bold]Status:[/bold] [{color}]{outcome.status.value}[/{color}]",
        f"[bold]Stage:[/bold] {outcome.current_stage}",
        f"[bold]QA Passed:[/bold] {outcome.qa_passed} (retries: {outcome.qa_retry_count})",
    ]
    out.print(Panel("\n".join(lines), title="CrewForge Run", border_style=color))

    if outcome.artifacts:
        table = Table(title="Artifacts (.ai-team/)")
        table.add_column("Artifact", style="cyan")
        table.add_column("Workspace Path", style="white")
        for name, rel_path in sorted(outcome.artifacts.items()):
            table.add_row(name, rel_path)
        out.print(table)

    if outcome.errors:
        out.print("[bold red]Errors:[/bold red]")
        for err in outcome.errors:
            out.print(f"  - {err}")

    if outcome.interrupted:
        out.print(
            Panel(
                "Execution paused at the Human-in-the-Loop approval gate.\n"
                "Review the artifacts in `.ai-team/`, then run:\n\n"
                f"  [bold green]crewforge resume {outcome.run_id} --approve[/bold green]\n"
                "or request changes:\n"
                f'  [bold yellow]crewforge resume {outcome.run_id} --reject '
                '--feedback "Your notes"[/bold yellow]',
                title="Action Required",
                border_style="magenta",
            )
        )


def render_team_table(config: CrewForgeConfig, *, console: Console | None = None) -> None:
    """Display the resolved virtual engineering team and model assignments."""
    out = console or Console()
    table = Table(title="CrewForge Virtual Team")
    table.add_column("Role", style="cyan")
    table.add_column("Model Profile", style="green")
    table.add_column("Provider / Model", style="white")
    table.add_column("Writes Artifacts", style="yellow")

    for role, agent_cfg in config.agents.agents.items():
        model_cfg = config.resolve_model_for_agent(role)
        table.add_row(
            role.value,
            agent_cfg.model_profile,
            f"{model_cfg.provider.value}:{model_cfg.model}",
            ", ".join(agent_cfg.writes_artifacts) or "-",
        )
    out.print(table)