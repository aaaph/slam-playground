import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from queue import Empty as QueueEmptyException
from queue import Queue
from threading import Event, Thread
from typing import Literal, cast

import pyarrow as pa
from dora import Node
from zenoh import Config, Sample, Session
from zenoh import open as zenoh_open

from logger import spawn_logger
from pipeline.annotations import Event as DoraEvent
from pipeline.decorators import on_input, on_inputs, on_stop, reactive
from pipeline.nodes.base import PipelineNode
from pipeline.runtime_config import ControlNodeConfig

type CommndValue = str | None
type Command = str | None
type JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]

RUN_DIR_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


@dataclass(frozen=True)
class PipelineRunStateWrite:
    """A queued pipeline run state write."""

    status: str
    dataflow_id: JsonValue
    node_config: JsonValue
    done: Event


class CommandTarget(Enum):
    """Command target."""

    DATASET = "ds"
    UNKNOWN = "unknown"


class PipelineRunState:
    """Persist the current Dora run location for external debugging tools."""

    def __init__(self, out_dir: Path | None = None, repo_root: Path | None = None) -> None:
        """Initialize the pipeline run state writer."""
        self.repo_root = repo_root or Path(__file__).resolve().parents[3]
        self.out_dir = out_dir or self.repo_root / "pipeline" / "out"
        self.state_file = self.out_dir / "current-run.json"
        self.latest_link = self.out_dir / "latest"

    def write(self, *, status: str, node: Node) -> None:
        """Write the current run state atomically."""
        dataflow_id, node_config = self.node_metadata(node)
        self.write_snapshot(status=status, dataflow_id=dataflow_id, node_config=node_config)

    def write_snapshot(self, *, status: str, dataflow_id: JsonValue, node_config: JsonValue) -> None:
        """Write a pre-captured run state snapshot atomically."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        log_dir = self._latest_log_dir()
        if log_dir is not None:
            self._update_latest_link(log_dir)

        state = {
            "schema_version": 1,
            "status": status,
            "updated_at": self._utc_now(),
            "pid": os.getpid(),
            "cwd": str(Path.cwd()),
            "dataflow_id": dataflow_id,
            "node_config": node_config,
            "out_dir": self._display_path(self.out_dir),
            "log_dir": self._display_path(log_dir) if log_dir is not None else None,
            "logs": self._log_files(log_dir) if log_dir is not None else {},
        }

        tmp_file = self.state_file.with_suffix(".json.tmp")
        tmp_file.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        tmp_file.replace(self.state_file)

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    def _latest_log_dir(self) -> Path | None:
        if not self.out_dir.exists():
            return None

        candidates = [
            path for path in self.out_dir.iterdir() if path.is_dir() and RUN_DIR_PATTERN.match(path.name)
        ]
        if not candidates:
            return None

        return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))

    def _update_latest_link(self, log_dir: Path) -> None:
        if self.latest_link.exists() or self.latest_link.is_symlink():
            if not self.latest_link.is_symlink() and self.latest_link.is_dir():
                return
            self.latest_link.unlink()
        self.latest_link.symlink_to(log_dir.name, target_is_directory=True)

    def _log_files(self, log_dir: Path) -> dict[str, str]:
        return {
            log_path.stem.removeprefix("log_"): self._display_path(log_path)
            for log_path in sorted(log_dir.glob("log_*.txt"))
        }

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.repo_root))
        except ValueError:
            return str(path)

    @classmethod
    def node_metadata(cls, node: Node) -> tuple[JsonValue, JsonValue]:
        """Capture Dora node metadata before enqueueing any background file I/O."""
        return cls._node_method_value(node, "dataflow_id"), cls._node_method_value(node, "node_config")

    @classmethod
    def _node_method_value(cls, node: Node, method_name: str) -> JsonValue:
        method = getattr(node, method_name, None)
        if not callable(method):
            return None
        try:
            return cls._json_safe(method())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None

    @staticmethod
    def _json_safe(value: object) -> JsonValue:
        if value is None or isinstance(value, str | int | float | bool):
            return value
        if isinstance(value, list | tuple):
            return [PipelineRunState._json_safe(item) for item in value]
        if isinstance(value, dict):
            return {str(key): PipelineRunState._json_safe(item) for key, item in value.items()}
        return repr(value)


class BackgroundPipelineRunState:
    """Queue pipeline run state writes onto a daemon worker thread."""

    def __init__(self, state: PipelineRunState | None = None) -> None:
        """Initialize the background state writer."""
        self.state = state or PipelineRunState()
        self.logger = spawn_logger(app="pipeline_run_state_writer")
        self.queue: Queue[PipelineRunStateWrite | None] = Queue()
        self.thread = Thread(target=self._run, name="pipeline-run-state-writer", daemon=True)
        self.thread.start()

    def write(self, *, status: str, node: Node) -> Event:
        """Enqueue a run state write and return an event that is set when it finishes."""
        dataflow_id, node_config = self.state.node_metadata(node)
        done = Event()
        self.queue.put_nowait(
            PipelineRunStateWrite(
                status=status,
                dataflow_id=dataflow_id,
                node_config=node_config,
                done=done,
            )
        )
        return done

    def close(self, *, timeout: float = 0.25) -> None:
        """Ask the worker to stop without letting file I/O hold node shutdown indefinitely."""
        self.queue.put_nowait(None)
        self.thread.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                self.state.write_snapshot(
                    status=item.status,
                    dataflow_id=item.dataflow_id,
                    node_config=item.node_config,
                )
                item.done.set()
            except OSError as exc:
                self.logger.warning(f"Could not write pipeline run state: {exc}")
            finally:
                if item is not None:
                    item.done.set()
                self.queue.task_done()


@reactive
class ZenohControlNode(PipelineNode):
    """Zenoh control node."""

    def __init__(
        self,
        node: Node | None = None,
        session: Session | None = None,
        run_state: BackgroundPipelineRunState | PipelineRunState | None = None,
        nodes_to_watch: set[str] | None = None,
        config: ControlNodeConfig | None = None,
    ) -> None:
        """Initialize the zenoh control node."""
        self.node: Node = node or Node()
        self.config = config or self.runtime_config_as(ControlNodeConfig)
        zenoh_config = Config()

        self.session: Session = session or zenoh_open(zenoh_config)
        self.signal_queue: Queue[
            dict[Literal["target", "command", "value"], CommandTarget | Command | CommndValue]
        ] = Queue()
        self.logger = spawn_logger(app="zenoh_control_node")
        self.run_state = run_state or BackgroundPipelineRunState()
        self.nodes_to_watch = (
            nodes_to_watch if nodes_to_watch is not None else set(self.config.expected_ready_nodes)
        )
        self.ready_nodes = set()
        self.all_nodes_ready = False

        def callback(data: Sample) -> None:
            line = data.payload.to_bytes().decode("utf-8").strip().lower()
            if line:
                self.logger.trace(f"Received command: {line}")
                target, command, value = self.parse_command(line)
                self.signal_queue.put({"target": target, "command": command, "value": value})

        self.sub = self.session.declare_subscriber("pipeline/control", callback)
        self.logger.info(f"Zenoh control node initialized: zid: {self.session.zid()}")
        self._write_run_state(status="running")

    @on_inputs("transport_tick", "tick")
    def handle_transport_tick(self) -> None:
        """Pooling of queue for commands from zenoh, if there are commands in the queue, send them to dataflow."""
        try:
            while not self.signal_queue.empty():
                obj = self.signal_queue.get_nowait()
                target, command, value = obj["target"], obj["command"], obj["value"]
                array = pa.array([command, value]) if value is not None else pa.array([command])
                target = cast("CommandTarget", target)
                self.node.send_output(target.value, array)
                self.logger.debug(f"Target: {target}, Command: {command}, Value: {value}")
        except QueueEmptyException:
            pass

    @on_input("startup_tick")
    def handle_startup_tick(self) -> None:
        """Handle startup tick."""
        self.all_nodes_ready = self._all_nodes_ready()
        self.logger.info(
            "Startup tick: "
            f"ready_nodes={self.ready_nodes}, "
            f"nodes_to_watch={self.nodes_to_watch}, "
            f"all_nodes_ready={self.all_nodes_ready}"
        )

    @on_input("*_status")
    def handle_status(self, event: DoraEvent) -> None:
        """Handle status updates."""
        arrow = event["value"]
        raw = arrow[0].as_py()

        payload = json.loads(raw)
        event_id = event["id"]
        node = self.config.ready_inputs.get(event_id, payload["node"])
        status = payload["state"]
        if status == "ready":
            self.ready_nodes.add(node)

    def _all_nodes_ready(self) -> bool:
        return self.nodes_to_watch.issubset(self.ready_nodes)

    @on_stop
    def graceful_shutdown(self) -> None:
        """Graceful shutdown."""
        self._write_run_state(status="stopped")
        self._close_run_state()
        self.sub.undeclare()
        self.session.close()
        self.logger.info("Zenoh control node stopped")

    def _write_run_state(self, *, status: str) -> None:
        try:
            self.run_state.write(status=status, node=self.node)
        except OSError as exc:
            self.logger.warning(f"Could not write pipeline run state: {exc}")

    def _close_run_state(self) -> None:
        close = getattr(self.run_state, "close", None)
        if callable(close):
            close(timeout=0.25)

    def parse_command(self, line: str) -> tuple[CommandTarget, Command, CommndValue]:
        """Parse the command."""
        try:
            target_raw, command, *values = line.split(":")
        except ValueError:
            return CommandTarget.UNKNOWN, None, None
        try:
            target = CommandTarget(target_raw)
        except ValueError:
            target = CommandTarget.UNKNOWN
        if not values or values == [""]:
            values = None
        return target, command, ":".join(values) if values is not None else None


if __name__ == "__main__":
    import os

    control_config = ZenohControlNode.runtime_config_as(ControlNodeConfig)
    raw_nodes_to_watch = os.getenv("PIPELINE_READY_NODES") or os.getenv("CONTROL_NODE_EXTECTING_NODES")
    nodes_to_watch = set(json.loads(raw_nodes_to_watch)) if raw_nodes_to_watch is not None else None
    ZenohControlNode(config=control_config, nodes_to_watch=nodes_to_watch).run()
