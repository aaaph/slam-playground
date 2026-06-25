from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Self

from pipeline.utils import RUN_DIR_PATTERN

type JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
type JsonObject = dict[str, JsonValue]

DEFAULT_ESTIMATE_PROPERTY = "slam_pose"
DEFAULT_REFERENCE_PROPERTY = "ground_truth_aligned_se3"


class PipelineResultError(RuntimeError):
    """Raised when a pipeline result cannot provide required artifacts."""


@dataclass(frozen=True)
class RerunStream:
    """One configured stream from a Rerun recording manifest."""

    property_name: str
    entity_path: str
    branch: str | None
    module: str | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class EvoRerunInputs:
    """Validated Rerun inputs needed to export evo trajectories."""

    rrd_path: Path
    rerun_manifest_path: Path
    rerun_blueprint_path: Path | None
    estimate_stream: RerunStream
    reference_stream: RerunStream
    output_dir: Path


@dataclass(frozen=True)
class PipelineResult:
    """Read-side wrapper around a pipeline run manifest."""

    repo_root: Path
    state_path: Path | None
    manifest: JsonObject

    @classmethod
    def current(cls, *, repo_root: Path | None = None, state_path: Path | None = None) -> Self:
        """Load the current pipeline result from pipeline/out/current-run.json."""
        resolved_repo_root = (repo_root or _default_repo_root()).resolve()
        resolved_state_path = state_path or resolved_repo_root / "pipeline" / "out" / "current-run.json"
        return cls.from_state_file(resolved_state_path, repo_root=resolved_repo_root)

    @classmethod
    def latest(cls, *, repo_root: Path | None = None, out_dir: Path | None = None) -> Self:
        """Build a result wrapper for the newest UUID-named run directory."""
        resolved_repo_root = (repo_root or _default_repo_root()).resolve()
        resolved_out_dir = _resolve_path(out_dir or Path("pipeline/out"), repo_root=resolved_repo_root)
        latest_log_dir = _latest_log_dir(resolved_out_dir)
        if latest_log_dir is None:
            msg = f"No pipeline run directories found in {resolved_out_dir}"
            raise FileNotFoundError(msg)

        return cls.from_log_dir(latest_log_dir, repo_root=resolved_repo_root)

    @classmethod
    def from_log_dir(cls, log_dir: Path, *, repo_root: Path | None = None) -> Self:
        """Build a result wrapper directly from one pipeline run log directory."""
        resolved_repo_root = (repo_root or _default_repo_root()).resolve()
        resolved_log_dir = _resolve_path(log_dir, repo_root=resolved_repo_root)
        if not resolved_log_dir.exists():
            msg = f"Pipeline run log directory not found: {resolved_log_dir}"
            raise FileNotFoundError(msg)
        if not resolved_log_dir.is_dir():
            msg = f"Pipeline run log path is not a directory: {resolved_log_dir}"
            raise NotADirectoryError(msg)

        out_dir = resolved_log_dir.parent
        manifest: JsonObject = {
            "schema_version": 1,
            "status": "unknown",
            "dataflow_id": resolved_log_dir.name,
            "out_dir": _display_path(out_dir, repo_root=resolved_repo_root),
            "log_dir": _display_path(resolved_log_dir, repo_root=resolved_repo_root),
            "logs": {
                log_path.stem.removeprefix("log_"): _display_path(log_path, repo_root=resolved_repo_root)
                for log_path in sorted(resolved_log_dir.glob("log_*.txt"))
            },
        }
        return cls(repo_root=resolved_repo_root, state_path=None, manifest=manifest)

    @classmethod
    def from_state_file(cls, state_path: Path, *, repo_root: Path | None = None) -> Self:
        """Load a pipeline result from a current-run.json file."""
        resolved_repo_root = (repo_root or _default_repo_root()).resolve()
        resolved_state_path = _resolve_path(state_path, repo_root=resolved_repo_root)
        raw_manifest = json.loads(resolved_state_path.read_text(encoding="utf-8"))
        if not isinstance(raw_manifest, dict):
            msg = f"Pipeline result manifest must be a JSON object: {resolved_state_path}"
            raise TypeError(msg)
        return cls(repo_root=resolved_repo_root, state_path=resolved_state_path, manifest=raw_manifest)

    @property
    def status(self) -> str | None:
        """Pipeline run status recorded by the control node."""
        status = self.manifest.get("status")
        return status if isinstance(status, str) else None

    @property
    def dataflow_id(self) -> JsonValue:
        """Dora dataflow id from the current-run manifest."""
        return self.manifest.get("dataflow_id")

    @property
    def run_id(self) -> str:
        """Stable run id, preferring dataflow_id and falling back to the log directory name."""
        dataflow_id = self.dataflow_id
        if isinstance(dataflow_id, str) and dataflow_id:
            return dataflow_id

        log_dir = self.log_dir
        if log_dir is not None:
            return log_dir.name

        msg = "Pipeline result has neither dataflow_id nor log_dir"
        raise PipelineResultError(msg)

    @property
    def out_dir(self) -> Path | None:
        """Resolved pipeline output directory if present in the manifest."""
        return self._optional_path("out_dir")

    @property
    def log_dir(self) -> Path | None:
        """Resolved run log directory if present in the manifest."""
        return self._optional_path("log_dir")

    @property
    def logs(self) -> dict[str, Path]:
        """Resolved log files keyed by node id."""
        raw_logs = self.manifest.get("logs")
        if not isinstance(raw_logs, dict):
            return {}

        result: dict[str, Path] = {}
        for key, value in raw_logs.items():
            if isinstance(value, str):
                result[str(key)] = _resolve_path(Path(value), repo_root=self.repo_root)
        return result

    @property
    def rerun_manifest_path(self) -> Path:
        """Expected Rerun recording manifest path for this run."""
        return self.require_log_dir() / "rerun_manifest.json"

    @property
    def rerun_config_path(self) -> Path:
        """Legacy resolved Rerun config sidecar path for older runs."""
        return self.require_log_dir() / "rerun_config.json"

    @property
    def rerun_blueprint_path(self) -> Path | None:
        """Resolved Rerun blueprint sidecar path for this run, if recorded."""
        files = self.rerun_manifest.get("files")
        if isinstance(files, dict):
            raw_blueprint = files.get("rerun_blueprint")
            if isinstance(raw_blueprint, str) and raw_blueprint:
                return _resolve_path(Path(raw_blueprint), repo_root=self.repo_root)
        default_path = self.require_log_dir() / "rerun_blueprint.rbl"
        return default_path if default_path.exists() else None

    @property
    def evo_dir(self) -> Path:
        """Default directory for evo export artifacts."""
        return self.require_log_dir() / "evo"

    @cached_property
    def rerun_manifest(self) -> JsonObject:
        """Load the Rerun sidecar manifest written next to data.rrd."""
        path = self.rerun_manifest_path
        raw_manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_manifest, dict):
            msg = f"Rerun manifest must be a JSON object: {path}"
            raise TypeError(msg)
        return raw_manifest

    @property
    def rrd_path(self) -> Path:
        """Resolved data.rrd path from the Rerun manifest."""
        files = self.rerun_manifest.get("files")
        if isinstance(files, dict):
            raw_rrd = files.get("rrd")
            if isinstance(raw_rrd, str) and raw_rrd:
                return _resolve_path(Path(raw_rrd), repo_root=self.repo_root)
        return self.require_log_dir() / "data.rrd"

    @property
    def rerun_runtime_config(self) -> dict[str, JsonValue]:
        """Runtime config embedded in the Rerun sidecar manifest."""
        runtime_config = self.rerun_manifest.get("runtime_config")
        return runtime_config if isinstance(runtime_config, dict) else {}

    @property
    def dataset_name(self) -> str | None:
        """Dataset selector recorded for the run, if available."""
        dataset_name = self.rerun_runtime_config.get("dataset_name")
        return dataset_name if isinstance(dataset_name, str) and dataset_name else None

    def require_dataset_name(self) -> str:
        """Return the recorded dataset selector, or raise when it is unavailable."""
        dataset_name = self.dataset_name
        if dataset_name is not None:
            return dataset_name
        msg = f"Pipeline result has no dataset_name in {self.rerun_manifest_path}"
        raise PipelineResultError(msg)

    def require_log_dir(self) -> Path:
        """Return the resolved log directory, or raise when the manifest does not identify one."""
        log_dir = self.log_dir
        if log_dir is None:
            location = self.state_path if self.state_path is not None else "synthetic latest result"
            msg = f"Pipeline result has no log_dir: {location}"
            raise PipelineResultError(msg)
        return log_dir

    def require_rerun_recording(self) -> Path:
        """Return data.rrd, requiring that the file exists."""
        path = self.rrd_path
        if not path.exists():
            msg = f"Rerun recording not found: {path}"
            raise FileNotFoundError(msg)
        return path

    def rerun_streams(self) -> list[RerunStream]:
        """Return configured Rerun streams from the sidecar manifest."""
        stream_index = self.rerun_manifest.get("stream_index")
        if not isinstance(stream_index, list):
            return []

        streams: list[RerunStream] = []
        for raw_stream in stream_index:
            if not isinstance(raw_stream, dict):
                continue
            stream = _parse_rerun_stream(raw_stream)
            if stream is not None:
                streams.append(stream)
        return streams

    def find_rerun_stream(
        self,
        property_name: str,
        *,
        branch: str | None = None,
        module: str | None = None,
    ) -> RerunStream | None:
        """Find one configured Rerun stream by pipeline context property name."""
        for stream in self.rerun_streams():
            if stream.property_name != property_name:
                continue
            if branch is not None and stream.branch != branch:
                continue
            if module is not None and stream.module != module:
                continue
            return stream
        return None

    def require_rerun_stream(
        self,
        property_name: str,
        *,
        branch: str | None = None,
        module: str | None = None,
    ) -> RerunStream:
        """Return one configured Rerun stream, raising a clear error when absent."""
        stream = self.find_rerun_stream(property_name, branch=branch, module=module)
        if stream is not None:
            return stream

        filters = [f"property_name={property_name!r}"]
        if branch is not None:
            filters.append(f"branch={branch!r}")
        if module is not None:
            filters.append(f"module={module!r}")
        msg = f"Rerun stream not found in {self.rerun_manifest_path}: {', '.join(filters)}"
        raise PipelineResultError(msg)

    def require_evo_rerun_inputs(
        self,
        *,
        estimate_property: str = DEFAULT_ESTIMATE_PROPERTY,
        reference_property: str = DEFAULT_REFERENCE_PROPERTY,
    ) -> EvoRerunInputs:
        """Validate the sidecar artifacts needed to export evo TUM trajectories."""
        return EvoRerunInputs(
            rrd_path=self.require_rerun_recording(),
            rerun_manifest_path=self.rerun_manifest_path,
            rerun_blueprint_path=self.rerun_blueprint_path,
            estimate_stream=self.require_rerun_stream(
                estimate_property,
                branch="slam_frame",
                module="dynamic_transform",
            ),
            reference_stream=self.require_rerun_stream(
                reference_property,
                branch="trajectory_evaluator_frame",
                module="dynamic_transform",
            ),
            output_dir=self.evo_dir,
        )

    def _optional_path(self, key: str) -> Path | None:
        value = self.manifest.get(key)
        if not isinstance(value, str) or not value:
            return None
        return _resolve_path(Path(value), repo_root=self.repo_root)


Result = PipelineResult


def _parse_rerun_stream(raw_stream: dict[str, Any]) -> RerunStream | None:
    property_name = raw_stream.get("property_name")
    entity_path = raw_stream.get("entity_path")
    if not isinstance(property_name, str) or not isinstance(entity_path, str):
        return None

    branch = raw_stream.get("branch")
    module = raw_stream.get("module")
    return RerunStream(
        property_name=property_name,
        entity_path=entity_path,
        branch=branch if isinstance(branch, str) else None,
        module=module if isinstance(module, str) else None,
        raw=raw_stream,
    )


def _latest_log_dir(out_dir: Path) -> Path | None:
    if not out_dir.exists():
        return None

    candidates = [path for path in out_dir.iterdir() if path.is_dir() and RUN_DIR_PATTERN.match(path.name)]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _resolve_path(path: Path, *, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _display_path(path: Path, *, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
