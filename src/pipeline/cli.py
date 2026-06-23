import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any, cast

import typer
from dora import build as dora_build
from dora import run as dora_run
from yaml import safe_dump
from zenoh import Config
from zenoh import open as zenoh_open

from dataset.factory import DatasetFactory
from pipeline.profiles import (
    PipelineProfileResolver,
    ProfileOverrides,
    ResolvedPipelineProfile,
    RunMode,
    VisualizationSink,
)
from pipeline.runtime_config import PIPELINE_NODE_CONFIG_ENV, ControlNodeRuntimeConfig
from pipeline.transport import ControlNodeTransport

RUN_STATE_PATH = Path("pipeline/out/current-run.json")
RUN_STATE_POLL_INTERVAL_SECONDS = 0.2
DORA_STOP_TIMEOUT_SECONDS = 5.0
DORA_KILL_TIMEOUT_SECONDS = 2.0
CONTROL_KEY = "pipeline/control"
CONTROL_HTTP_TIMEOUT_SECONDS = 2.0
CONTROL_READY_TIMEOUT_SECONDS = 30.0
ZENOH_COMMAND_SETTLE_SECONDS = 0.25
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
ControlTransportOption = Annotated[
    ControlNodeTransport | None,
    typer.Option("--control-transport", help="External control ingress override."),
]
ControlHttpHostOption = Annotated[str | None, typer.Option("--control-http-host", help="HTTP control bind host.")]
ControlHttpPortOption = Annotated[int | None, typer.Option("--control-http-port", help="HTTP control bind port.")]
ControlCommandZenohOption = Annotated[bool, typer.Option("--zenoh", help="Send control command via Zenoh.")]
ControlCommandHttpOption = Annotated[bool, typer.Option("--http", help="Send control command via HTTP.")]
ControlCommandTargetOption = Annotated[str, typer.Option("--target", help="Control command target.")]
ControlCommandKeyOption = Annotated[str, typer.Option("--key", help="Zenoh control key.")]
ControlCommandHostOption = Annotated[
    str,
    typer.Option("--host", "--control-http-host", help="HTTP control host."),
]
ControlCommandPortOption = Annotated[
    int,
    typer.Option("--port", "--control-http-port", min=0, max=65535, help="HTTP control port."),
]
ControlCommandTimeoutOption = Annotated[
    float,
    typer.Option("--timeout", min=0.1, help="HTTP control request timeout in seconds."),
]
ControlCommandReadyTimeoutOption = Annotated[
    float,
    typer.Option("--ready-timeout", min=0.0, help="Seconds to wait for the control node to become ready."),
]


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
    control_transport: ControlTransportOption = None,
    control_http_host: ControlHttpHostOption = None,
    control_http_port: ControlHttpPortOption = None,
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
                control_transport=control_transport,
                control_http_host=control_http_host,
                control_http_port=control_http_port,
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
    control_transport: ControlTransportOption = None,
    control_http_host: ControlHttpHostOption = None,
    control_http_port: ControlHttpPortOption = None,
) -> None:
    """Resolve and launch the requested pipeline run."""
    profile_resolver = PipelineProfileResolver()
    profile_overrides = ProfileOverrides(
        dataset=dataset,
        dataflow=dataflow,
        visualization_sink=visualization_sink,
        run_mode=run_mode,
        fraction=fraction,
        control_transport=control_transport,
        control_http_host=control_http_host,
        control_http_port=control_http_port,
    )
    resolved_profile = profile_resolver.resolve(
        profile=profile,
        overrides=profile_overrides,
    )
    _dataset_pre_cache(resolved_profile)

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


@pipeline_app.command("start", context_settings=HELP_CONTEXT_SETTINGS)
def start_pipeline(  # noqa: PLR0913
    *,
    zenoh: ControlCommandZenohOption = False,
    http: ControlCommandHttpOption = False,
    target: ControlCommandTargetOption = "ds",
    key: ControlCommandKeyOption = CONTROL_KEY,
    host: ControlCommandHostOption = "127.0.0.1",
    port: ControlCommandPortOption = 8765,
    timeout: ControlCommandTimeoutOption = CONTROL_HTTP_TIMEOUT_SECONDS,
    ready_timeout: ControlCommandReadyTimeoutOption = CONTROL_READY_TIMEOUT_SECONDS,
) -> None:
    """Send a dataset start command to the running control node."""
    _send_dataset_control_command(
        "start",
        value=None,
        zenoh=zenoh,
        http=http,
        target=target,
        key=key,
        host=host,
        port=port,
        timeout=timeout,
        ready_timeout=ready_timeout,
    )


@pipeline_app.command("stop", context_settings=HELP_CONTEXT_SETTINGS)
def stop_pipeline(  # noqa: PLR0913
    *,
    zenoh: ControlCommandZenohOption = False,
    http: ControlCommandHttpOption = False,
    target: ControlCommandTargetOption = "ds",
    key: ControlCommandKeyOption = CONTROL_KEY,
    host: ControlCommandHostOption = "127.0.0.1",
    port: ControlCommandPortOption = 8765,
    timeout: ControlCommandTimeoutOption = CONTROL_HTTP_TIMEOUT_SECONDS,
    ready_timeout: ControlCommandReadyTimeoutOption = CONTROL_READY_TIMEOUT_SECONDS,
) -> None:
    """Send a dataset stop command to the running control node."""
    _send_dataset_control_command(
        "stop",
        value=None,
        zenoh=zenoh,
        http=http,
        target=target,
        key=key,
        host=host,
        port=port,
        timeout=timeout,
        ready_timeout=ready_timeout,
    )


@pipeline_app.command("step", context_settings=HELP_CONTEXT_SETTINGS)
def step_pipeline(  # noqa: PLR0913
    value: Annotated[str, typer.Argument(help="Dataset step value, e.g. 1, 2s, or 10%.")],
    *,
    zenoh: ControlCommandZenohOption = False,
    http: ControlCommandHttpOption = False,
    target: ControlCommandTargetOption = "ds",
    key: ControlCommandKeyOption = CONTROL_KEY,
    host: ControlCommandHostOption = "127.0.0.1",
    port: ControlCommandPortOption = 8765,
    timeout: ControlCommandTimeoutOption = CONTROL_HTTP_TIMEOUT_SECONDS,
    ready_timeout: ControlCommandReadyTimeoutOption = CONTROL_READY_TIMEOUT_SECONDS,
) -> None:
    """Send a bounded dataset step command to the running control node."""
    _send_dataset_control_command(
        "step",
        value=value,
        zenoh=zenoh,
        http=http,
        target=target,
        key=key,
        host=host,
        port=port,
        timeout=timeout,
        ready_timeout=ready_timeout,
    )


def _send_dataset_control_command(  # noqa: PLR0913
    command: str,
    *,
    value: str | None,
    zenoh: bool,
    http: bool,
    target: str,
    key: str,
    host: str,
    port: int,
    timeout: float,
    ready_timeout: float,
) -> None:
    line = f"{target}:{command}" if value is None else f"{target}:{command}:{value}"
    transport = _control_command_transport_from_flags(zenoh=zenoh, http=http)
    _wait_for_control_node_ready(timeout_seconds=ready_timeout)
    _send_control_line(
        line,
        transport=transport,
        key=key,
        host=host,
        port=port,
        timeout=timeout,
    )
    typer.echo(f"Sent control command via {transport.value}: {line}")


def _control_command_transport_from_flags(*, zenoh: bool, http: bool) -> ControlNodeTransport:
    if zenoh and http:
        msg = "choose either --zenoh or --http"
        raise typer.BadParameter(msg)
    if zenoh:
        return ControlNodeTransport.ZENOH
    return ControlNodeTransport.HTTP


def _send_control_line(  # noqa: PLR0913
    line: str,
    *,
    transport: ControlNodeTransport,
    key: str,
    host: str,
    port: int,
    timeout: float,
) -> None:
    if transport == ControlNodeTransport.ZENOH:
        _send_zenoh_control_line(line, key=key)
        return
    _send_http_control_line(line, host=host, port=port, timeout=timeout)


def _send_zenoh_control_line(line: str, *, key: str) -> None:
    session = zenoh_open(Config())
    try:
        session.put(key, line)
        time.sleep(ZENOH_COMMAND_SETTLE_SECONDS)
    finally:
        session.close()


def _send_http_control_line(line: str, *, host: str, port: int, timeout: float) -> None:
    request = urllib.request.Request(
        f"http://{host}:{port}/control",
        data=line.encode("utf-8"),
        headers={"Content-Type": "text/plain"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - local control endpoint
            if response.status != HTTPStatus.ACCEPTED:
                msg = f"HTTP control command failed with status {response.status}"
                _abort_control_command(msg)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        msg = f"HTTP control command failed with status {exc.code}: {detail}"
        _abort_control_command(msg)
    except urllib.error.URLError as exc:
        msg = f"HTTP control command failed: {exc.reason}"
        _abort_control_command(msg)


def _abort_control_command(message: str) -> None:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(1)


def _wait_for_control_node_ready(*, timeout_seconds: float) -> None:
    if timeout_seconds <= 0:
        return

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() <= deadline:
        if _current_control_node_is_ready():
            return
        time.sleep(RUN_STATE_POLL_INTERVAL_SECONDS)

    msg = f"control node did not report readiness within {timeout_seconds:g}s"
    _abort_control_command(msg)


def _current_control_node_is_ready() -> bool:
    try:
        raw_state = json.loads(RUN_STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    if not isinstance(raw_state, dict) or raw_state.get("status") != "running":
        return False

    logs = raw_state.get("logs")
    if not isinstance(logs, dict):
        return False
    control_log = logs.get("control")
    if not isinstance(control_log, str):
        return False

    control_log_path = Path(control_log)
    if not control_log_path.is_absolute():
        control_log_path = Path.cwd() / control_log_path
    try:
        return "Nodes are ready" in control_log_path.read_text(encoding="utf-8")
    except OSError:
        return False


def _dataset_pre_cache(resolved_profile: ResolvedPipelineProfile) -> None:
    typer.echo(f"Dataset {resolved_profile.dataset.name}: cache_path={resolved_profile.dataset.cache}")
    typer.echo("Pre-caching dataset transforms...")
    ds = (
        DatasetFactory(repo_root=resolved_profile.repo_root)
        .load_vio_dataset(resolved_profile.dataset.name)
        .imu_and_stereo(decode_images=False)
    )
    typer.echo(f"Dataset features: {ds.features}")


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
    return [
        sys.executable,
        "-c",
        "import sys\nfrom dora import run\nrun(sys.argv[1], uv=sys.argv[2] == '1')",
        str(dataflow_path),
        "1" if uv else "0",
    ]


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
