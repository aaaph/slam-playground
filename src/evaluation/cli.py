from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from evo import EvoException

from evaluation.ape import DEFAULT_MAX_TIMESTAMP_DIFF_SECONDS, evaluate_ape, format_ape_summary, show_ape_plot
from evaluation.tum_export import TumEntity, create_tum_exports
from pipeline.result import PipelineResult, PipelineResultError

HELP_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

app = typer.Typer(context_settings=HELP_CONTEXT_SETTINGS)

RunOption = Annotated[
    str,
    typer.Option(
        "--run",
        help=(
            "Run selector: latest, current, a run id under pipeline/out, "
            "a run directory, or a current-run.json path."
        ),
    ),
]
EntityOption = Annotated[TumEntity, typer.Option("--entity", help="Trajectory entity to export.")]
RepoRootOption = Annotated[Path | None, typer.Option("--repo-root", help="Repository root.")]
AlignOption = Annotated[
    bool,
    typer.Option("--align/--no-align", help="Align estimate to raw dataset ground truth."),
]
MaxTimestampDiffOption = Annotated[
    float,
    typer.Option("--max-timestamp-diff", min=0.0, help="Maximum timestamp association difference in seconds."),
]
SaveResultOption = Annotated[
    bool,
    typer.Option("--save-result/--no-save-result", help="Save evo result artifacts next to the run."),
]
PlotOption = Annotated[
    bool,
    typer.Option("-p", "--plot/--no-plot", help="Show evo APE plot window."),
]


@app.callback(context_settings=HELP_CONTEXT_SETTINGS)
def evaluation_cli() -> None:
    """Evaluate pipeline run results."""


@app.command("create-tum", context_settings=HELP_CONTEXT_SETTINGS)
def create_tum(
    run: RunOption = "latest",
    entity: EntityOption = TumEntity.ALL,
    repo_root: RepoRootOption = None,
) -> None:
    """Create TUM trajectory files from a pipeline Rerun recording."""
    try:
        result = _resolve_pipeline_result(run, repo_root=repo_root)
        exports = create_tum_exports(result, entity=entity)
    except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError, PipelineResultError, RuntimeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    for export in exports:
        typer.echo(f"{export.entity.value}: {export.path} ({export.samples_count} poses)")


@app.command("ape", context_settings=HELP_CONTEXT_SETTINGS)
def run_ape(  # noqa: PLR0913
    *,
    run: RunOption = "latest",
    repo_root: RepoRootOption = None,
    align: AlignOption = True,
    max_timestamp_diff: MaxTimestampDiffOption = DEFAULT_MAX_TIMESTAMP_DIFF_SECONDS,
    save_result: SaveResultOption = False,
    plot: PlotOption = False,
) -> None:
    """Compute offline evo APE against the run's dataset ground truth."""
    try:
        result = _resolve_pipeline_result(run, repo_root=repo_root)
        artifacts = evaluate_ape(
            result,
            align=align,
            max_timestamp_diff_seconds=max_timestamp_diff,
            save_result=save_result,
        )
    except (
        FileNotFoundError,
        NotADirectoryError,
        json.JSONDecodeError,
        PipelineResultError,
        RuntimeError,
        EvoException,
    ) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(format_ape_summary(artifacts))
    if plot:
        try:
            show_ape_plot(artifacts)
        except (ImportError, RuntimeError, ValueError, EvoException) as exc:
            typer.echo(f"Error: could not show plot: {exc}", err=True)
            raise typer.Exit(1) from exc


def _resolve_pipeline_result(run: str, *, repo_root: Path | None = None) -> PipelineResult:
    resolved_repo_root = (repo_root or Path.cwd()).resolve()
    if run == "current":
        return PipelineResult.current(repo_root=resolved_repo_root)
    if run == "latest":
        try:
            return PipelineResult.current(repo_root=resolved_repo_root)
        except (FileNotFoundError, json.JSONDecodeError, TypeError, PipelineResultError):
            return PipelineResult.latest(repo_root=resolved_repo_root)

    run_path = Path(run)
    if run_path.suffix == ".json":
        return PipelineResult.from_state_file(run_path, repo_root=resolved_repo_root)
    if run_path.exists():
        return PipelineResult.from_log_dir(run_path, repo_root=resolved_repo_root)
    return PipelineResult.from_log_dir(resolved_repo_root / "pipeline" / "out" / run, repo_root=resolved_repo_root)


def main() -> None:
    """CLI entrypoint for evaluation helpers."""
    app()


if __name__ == "__main__":
    main()
