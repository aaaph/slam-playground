import os
from typing import Annotated

import typer
from dora import build as dora_build
from dora import run as dora_run
from yaml import safe_dump

from pipeline.profiles import PipelineProfileResolver, ProfileOverrides, RunMode, VisualizationSink

app = typer.Typer()
pipeline_app = typer.Typer()
profile_app = typer.Typer()
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(profile_app, name="profile")

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


@profile_app.command("resolve")
def resolve_profile(  # noqa: PLR0913
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
    """Resolve and launch the requested pipeline run."""
    profile_resolver = PipelineProfileResolver()
    profile_overrides = ProfileOverrides(
        dataset=dataset,
        dataflow=dataflow,
        visualization_sink=visualization_sink,
        run_mode=run_mode,
        fraction=fraction,
    )
    resolved_profile = profile_resolver.resolve(
        profile=profile,
        overrides=profile_overrides,
    )
    dataflow_path = resolved_profile.dataflow.template
    if resolved_profile.dataflow.build:
        dora_build(
            dataflow_path=str(dataflow_path),
            uv=True,
        )

    os.environ["REPO_ROOT"] = str(resolved_profile.repo_root)
    os.environ["PIPELINE_PROFILE"] = resolved_profile.profile or "empty"
    os.environ["DATASET_NAME"] = resolved_profile.dataset.name
    os.environ["DATAFLOW_NAME"] = resolved_profile.dataflow.name
    os.environ["DATASET_ROOT"] = str(resolved_profile.dataset.root)
    os.environ["DATASET_RIG_PATH"] = str(resolved_profile.dataset.rig)
    os.environ["RUN_MODE"] = resolved_profile.run.mode.value
    os.environ["FRACTION"] = str(resolved_profile.run.fraction)
    os.environ["AUTOSTART_AFTER_READY"] = str(resolved_profile.run.autostart_after_ready)
    os.environ["STOP_AFTER_DATASET_DONE"] = str(resolved_profile.run.stop_after_dataset_done)

    dora_run(
        dataflow_path=str(dataflow_path),
        uv=True,
    )


def main() -> None:
    """CLI entrypoint for the pipeline package."""
    app()


if __name__ == "__main__":
    main()
