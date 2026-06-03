from typing import Annotated

import typer
from yaml import safe_dump

from pipeline.profiles import PipelineProfileResolver, ProfileOverrides, RunMode, VisualizationSink

app = typer.Typer()
pipeline_app = typer.Typer()
app.add_typer(pipeline_app, name="pipeline")

ProfileOption = Annotated[str | None, typer.Option(help="Composite pipeline profile name.")]
DatasetOption = Annotated[str | None, typer.Option(help="Dataset profile override.")]
DataflowOption = Annotated[str | None, typer.Option(help="Dataflow profile override.")]
VisualizationSinkOption = Annotated[
    VisualizationSink | None,
    typer.Option("--visualization-sink", "--visualizer-sink", help="Visualization sink override."),
]
RunModeOption = Annotated[RunMode | None, typer.Option("--run-mode", help="Run mode override.")]
FractionOption = Annotated[float | None, typer.Option(min=0.0, max=1.0, help="Dataset fraction override.")]


def _resolve_profile(profile: str | None, overrides: ProfileOverrides) -> str:
    resolver = PipelineProfileResolver()
    resolved = resolver.resolve(profile=profile, overrides=overrides)
    return safe_dump(resolved.model_dump(mode="json"), sort_keys=False)


@pipeline_app.command("resolve")
def resolve_pipeline(  # noqa: PLR0913
    profile: ProfileOption = None,
    dataset: DatasetOption = None,
    dataflow: DataflowOption = None,
    visualization_sink: VisualizationSinkOption = None,
    run_mode: RunModeOption = None,
    fraction: FractionOption = None,
) -> None:
    """Resolve profile selectors and CLI overrides into a run config snapshot."""
    typer.echo(
        _resolve_profile(
            profile,
            ProfileOverrides(
                dataset=dataset,
                dataflow=dataflow,
                visualization_sink=visualization_sink,
                run_mode=run_mode,
                fraction=fraction,
            ),
        )
    )


@pipeline_app.command("run")
def run_pipeline(  # noqa: PLR0913
    profile: ProfileOption = None,
    dataset: DatasetOption = None,
    dataflow: DataflowOption = None,
    visualization_sink: VisualizationSinkOption = None,
    run_mode: RunModeOption = None,
    fraction: FractionOption = None,
) -> None:
    """Resolve the requested pipeline run. Actual Dora launch is the next layer."""
    typer.echo(
        _resolve_profile(
            profile,
            ProfileOverrides(
                dataset=dataset,
                dataflow=dataflow,
                visualization_sink=visualization_sink,
                run_mode=run_mode,
                fraction=fraction,
            ),
        )
    )


def main() -> None:
    """CLI entrypoint for the pipeline package."""
    app()


if __name__ == "__main__":
    main()
