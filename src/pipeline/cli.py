import json
import os
import signal
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any, cast

import typer
from dora import build as dora_build
from dora import run as dora_run
from yaml import safe_dump

from pipeline.profiles import (
    PipelineProfileResolver,
    ProfileOverrides,
    ResolvedPipelineProfile,
    RunMode,
    VisualizationSink,
)
from pipeline.runtime_config import PIPELINE_NODE_CONFIG_ENV, ControlNodeRuntimeConfig

RUN_STATE_PATH = Path("pipeline/out/current-run.json")
RUN_STATE_POLL_INTERVAL_SECONDS = 0.2
DORA_STOP_TIMEOUT_SECONDS = 5.0
DORA_KILL_TIMEOUT_SECONDS = 2.0
HELP_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

app = typer.Typer(context_settings=HELP_CONTEXT_SETTINGS)
pipeline_app = typer.Typer(context_settings=HELP_CONTEXT_SETTINGS)
profile_app = typer.Typer(context_settings=HELP_CONTEXT_SETTINGS)
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(profile_app, name="profile")

ProfileOption = Annotated[str | None, typer.Option(help="Composite pipeline profile name.")]
DatasetOption = Annotated[str | None, typer.Option(help="Dataset profile override.")]
DataflowOption = Annotated[str | None, typer.Option(help="Dataflow profile override.")]
VisualizationSinkOption = Annotated[
    VisualizationSink | None,
    typer.Option("--viz", "--visualization-sink", "--visualizer-sink", help="Visualization sink override."),
]
RunModeOption = Annotated[RunMode | None, typer.Option("--run-mode", help="Run mode override.")]
FractionOption = Annotated[float | None, typer.Option(min=0.0, max=1.0, help="Dataset fraction override.")]


def _resolve_profile(profile: str | None, overrides: ProfileOverrides) -> str:
    resolver = PipelineProfileResolver()
    resolved = resolver.resolve(profile=profile, overrides=overrides)
    return safe_dump(resolved.model_dump(mode="json"), sort_keys=False)


@contextmanager
def _materialized_runtime_dataflow(
    dataflow_path: Path,
    resolved_profile: ResolvedPipelineProfile,
) -> Iterator[Path]:
    source_path = dataflow_path.resolve()
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=source_path.parent,
        prefix=f".{source_path.stem}.",
        suffix=".runtime.yml",
        delete=False,
    ) as tmp_file:
        runtime_path = Path(tmp_file.name)
        safe_dump(resolved_profile.dataflow.runtime_dataflow, tmp_file, sort_keys=False)

    try:
        yield runtime_path
    finally:
        runtime_path.unlink(missing_ok=True)


@profile_app.command("resolve", context_settings=HELP_CONTEXT_SETTINGS)
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


@pipeline_app.command("run", context_settings=HELP_CONTEXT_SETTINGS)
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
    dataflow_path = resolved_profile.repo_root / resolved_profile.dataflow.template

    os.environ["REPO_ROOT"] = str(resolved_profile.repo_root)
    os.environ["PIPELINE_PROFILE"] = resolved_profile.profile or "empty"
    os.environ["DATASET_NAME"] = resolved_profile.dataset.name
    os.environ["DATAFLOW_NAME"] = resolved_profile.dataflow.name
    ready_nodes = json.dumps(_expected_ready_nodes(resolved_profile.dataflow.runtime_dataflow))
    os.environ["PIPELINE_READY_NODES"] = ready_nodes
    os.environ["CONTROL_NODE_EXTECTING_NODES"] = ready_nodes
    os.environ["DATASET_ROOT"] = str(resolved_profile.dataset.root)
    os.environ["DATASET_RIG_PATH"] = str(resolved_profile.dataset.rig)
    os.environ["RUN_MODE"] = resolved_profile.run.mode.value
    os.environ["FRACTION"] = str(resolved_profile.run.fraction)
    os.environ["AUTOSTART_AFTER_READY"] = str(resolved_profile.run.autostart_after_ready)
    os.environ["STOP_AFTER_DATASET_DONE"] = str(resolved_profile.run.stop_after_dataset_done)

    with _materialized_runtime_dataflow(dataflow_path, resolved_profile) as runtime_dataflow_path:
        if resolved_profile.dataflow.build:
            dora_build(
                dataflow_path=str(runtime_dataflow_path),
                uv=True,
            )

        _run_dora_dataflow(
            runtime_dataflow_path,
            uv=True,
            repo_root=resolved_profile.repo_root,
            stop_on_completed=resolved_profile.run.stop_after_dataset_done,
        )


def _expected_ready_nodes(runtime_dataflow: dict[str, object]) -> list[str]:
    raw_nodes = runtime_dataflow.get("nodes", [])
    if not isinstance(raw_nodes, list):
        return []
    for raw_node in raw_nodes:
        if not isinstance(raw_node, dict):
            continue
        raw_node_config = cast("dict[str, Any]", raw_node)
        if raw_node_config.get("id") != "control":
            continue
        raw_env = raw_node_config.get("env", {})
        if not isinstance(raw_env, dict):
            return []
        raw_config = raw_env.get(PIPELINE_NODE_CONFIG_ENV)
        if not isinstance(raw_config, str):
            return []
        return ControlNodeRuntimeConfig.model_validate_json(raw_config).expected_ready_nodes
    return []


def _run_dora_dataflow(
    dataflow_path: Path,
    *,
    uv: bool,
    repo_root: Path,
    stop_on_completed: bool,
) -> None:
    if not stop_on_completed:
        dora_run(dataflow_path=str(dataflow_path), uv=uv)
        return

    _run_dora_dataflow_until_completed(
        dataflow_path,
        uv=uv,
        state_file=repo_root / RUN_STATE_PATH,
    )


def _run_dora_dataflow_until_completed(
    dataflow_path: Path,
    *,
    uv: bool,
    state_file: Path,
    poll_interval_seconds: float = RUN_STATE_POLL_INTERVAL_SECONDS,
) -> None:
    state_file.unlink(missing_ok=True)
    process = subprocess.Popen(  # noqa: S603 - dora CLI is the configured pipeline runner.
        _dora_run_command(dataflow_path, uv=uv),
        start_new_session=True,
    )

    try:
        while True:
            return_code = process.poll()
            if _pipeline_completed(state_file):
                _stop_dora_run_process(process)
                return
            if return_code is not None:
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, _dora_run_command(dataflow_path, uv=uv))
                return
            time.sleep(poll_interval_seconds)
    except KeyboardInterrupt:
        _stop_dora_run_process(process)
        raise


def _dora_run_command(dataflow_path: Path, *, uv: bool) -> list[str]:
    command = ["dora", "run"]
    if uv:
        command.append("--uv")
    command.append(str(dataflow_path))
    return command


def _pipeline_completed(state_file: Path) -> bool:
    try:
        raw_state = json.loads(state_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    if not isinstance(raw_state, dict):
        return False
    return raw_state.get("status") == "completed"


def _stop_dora_run_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return

    _send_interrupt(process)
    try:
        process.wait(timeout=DORA_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=DORA_KILL_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _send_interrupt(process: subprocess.Popen[bytes]) -> None:
    if hasattr(os, "killpg"):
        os.killpg(process.pid, signal.SIGINT)
        return
    process.send_signal(signal.SIGINT)


def main() -> None:
    """CLI entrypoint for the pipeline package."""
    app()


if __name__ == "__main__":
    main()
