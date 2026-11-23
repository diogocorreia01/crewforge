"""Thin Typer CLI entrypoint for CrewForge."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from crewforge.cli.rendering import render_outcome, render_team_table
from crewforge.config import ConfigLoadError, load_config
from crewforge.engine import OrchestratorRunner

app = typer.Typer(
    name="crewforge",
    help="Multi-Agent Software Development Orchestrator powered by LangGraph.",
    no_args_is_help=True,
)
_console = Console()


@app.command()
def init(
    workspace: Annotated[
        Path,
        typer.Option("--workspace", "-w", help="Target project directory."),
    ] = Path("."),
) -> None:
    """Initialize the `.ai-team/` folder inside a target workspace."""
    root = workspace.resolve()
    config_override_dir = root / ".ai-team" / "config"
    config_override_dir.mkdir(parents=True, exist_ok=True)
    _console.print(f"[green]Initialized CrewForge workspace at:[/green] {root / '.ai-team'}")


@app.command()
def doctor(
    workspace: Annotated[
        Path | None,
        typer.Option("--workspace", "-w", help="Optional target workspace to check overrides."),
    ] = None,
) -> None:
    """Validate YAML configurations and display resolved team model assignments."""
    try:
        cfg = load_config(workspace_dir=workspace)
    except ConfigLoadError as exc:
        _console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    render_team_table(cfg, console=_console)


@app.command()
def run(
    feature_request: Annotated[
        str,
        typer.Argument(help="Feature description for the virtual engineering team."),
    ],
    workspace: Annotated[
        Path,
        typer.Option("--workspace", "-w", help="Target project directory."),
    ] = Path("."),
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Optional explicit run identifier."),
    ] = None,
    no_hitl: Annotated[
        bool,
        typer.Option("--no-hitl", help="Disable the Human-in-the-Loop plan approval gate."),
    ] = False,
) -> None:
    """Start a new multi-agent feature workflow against a target workspace."""
    cfg = load_config(workspace_dir=workspace)
    if no_hitl:
        cfg.workflow.enable_hitl_plan_gate = False

    runner = OrchestratorRunner(workspace_dir=workspace, config=cfg)
    outcome = runner.start(feature_request, run_id=run_id)
    render_outcome(outcome, console=_console)


@app.command()
def resume(
    run_id: Annotated[str, typer.Argument(help="Run ID paused at an HITL gate.")],
    workspace: Annotated[
        Path,
        typer.Option("--workspace", "-w", help="Target project directory."),
    ] = Path("."),
    approve: Annotated[
        bool,
        typer.Option("--approve/--reject", help="Approve the plan or request changes."),
    ] = True,
    feedback: Annotated[
        str,
        typer.Option("--feedback", "-f", help="Optional feedback when requesting changes."),
    ] = "",
) -> None:
    """Resume a paused Human-in-the-Loop workflow run."""
    runner = OrchestratorRunner(workspace_dir=workspace)
    outcome = runner.resume(run_id, approved=approve, feedback=feedback)
    render_outcome(outcome, console=_console)


@app.command()
def status(
    run_id: Annotated[str, typer.Argument(help="Run ID to inspect.")],
    workspace: Annotated[
        Path,
        typer.Option("--workspace", "-w", help="Target project directory."),
    ] = Path("."),
) -> None:
    """Inspect the latest saved checkpoint status for a run."""
    runner = OrchestratorRunner(workspace_dir=workspace)
    outcome = runner.inspect_status(run_id)
    if outcome is None:
        _console.print(f"[yellow]No checkpoint found for run '{run_id}'.[/yellow]")
        raise typer.Exit(code=1)
    render_outcome(outcome, console=_console)