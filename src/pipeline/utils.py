import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from queue import Queue
from threading import Event, Thread

from dora import Node

from logger import spawn_logger

type JsonValue = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None

RUN_DIR_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


@dataclass(frozen=True)
class PipelineRunStateWrite:
    """A queued pipeline run state write."""

    status: str
    dataflow_id: JsonValue
    node_config: JsonValue
    done: Event


class PipelineRunState:
    """Persist the current Dora run location for external debugging tools."""

    def __init__(self, out_dir: Path | None = None, repo_root: Path | None = None) -> None:
        """Initialize the pipeline run state writer."""
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]
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
