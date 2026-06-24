from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import pyarrow as pa
from rerun.experimental import RrdReader

from pipeline.result import (
    DEFAULT_ESTIMATE_PROPERTY,
    DEFAULT_REFERENCE_PROPERTY,
    PipelineResult,
    PipelineResultError,
    RerunStream,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


FRAME_TIME_COLUMN = "frame_time"
TRANSLATION_COLUMN = "Transform3D:translation"
QUATERNION_COLUMN = "Transform3D:quaternion"
TRANSFORM_COMPONENTS = [TRANSLATION_COLUMN, QUATERNION_COLUMN]


class TumEntity(StrEnum):
    """Trajectory selector for TUM export."""

    ALL = "all"
    SLAM_OUTPUT = "slam_output"
    GROUND_TRUTH = "ground_truth"


class TumExportError(RuntimeError):
    """Raised when a Rerun trajectory cannot be exported to TUM."""


@dataclass(frozen=True)
class TumPose:
    """One TUM trajectory pose."""

    timestamp_ns: int
    translation: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class TumEntitySpec:
    """Mapping from a semantic entity selector to a Rerun stream."""

    entity: TumEntity
    property_name: str
    branch: str
    module: str
    output_name: str


@dataclass(frozen=True)
class TumExport:
    """One created TUM export artifact."""

    entity: TumEntity
    path: Path
    samples_count: int
    stream: RerunStream


ENTITY_SPECS = {
    TumEntity.SLAM_OUTPUT: TumEntitySpec(
        entity=TumEntity.SLAM_OUTPUT,
        property_name=DEFAULT_ESTIMATE_PROPERTY,
        branch="slam_frame",
        module="dynamic_transform",
        output_name="slam_output.tum",
    ),
    TumEntity.GROUND_TRUTH: TumEntitySpec(
        entity=TumEntity.GROUND_TRUTH,
        property_name=DEFAULT_REFERENCE_PROPERTY,
        branch="trajectory_evaluator_frame",
        module="dynamic_transform",
        output_name="ground_truth_aligned.tum",
    ),
}


def create_tum_exports(
    result: PipelineResult,
    *,
    entity: TumEntity = TumEntity.ALL,
    output_dir: Path | None = None,
) -> list[TumExport]:
    """Create one or more TUM trajectory files for a pipeline result."""
    specs = _selected_specs(entity)
    rrd_path = result.require_rerun_recording()
    resolved_output_dir = output_dir or result.evo_dir
    exports: list[TumExport] = []

    for spec in specs:
        stream = result.require_rerun_stream(
            spec.property_name,
            branch=spec.branch,
            module=spec.module,
        )
        poses = read_transform_trajectory(rrd_path, stream.entity_path)
        path = resolved_output_dir / spec.output_name
        write_tum_trajectory(path, poses)
        exports.append(TumExport(entity=spec.entity, path=path, samples_count=len(poses), stream=stream))

    return exports


def read_transform_trajectory(rrd_path: Path, entity_path: str) -> list[TumPose]:
    """Read a Transform3D trajectory from a Rerun recording."""
    chunks = []
    store = RrdReader(rrd_path).store()
    for candidate_path in _entity_path_candidates(entity_path):
        chunks = (
            store.stream()
            .filter(content=candidate_path, is_static=False, components=TRANSFORM_COMPONENTS)
            .to_chunks()
        )
        if chunks:
            break

    if not chunks:
        msg = f"Transform3D entity not found in {rrd_path}: {entity_path}"
        raise PipelineResultError(msg)

    poses: list[TumPose] = []
    for chunk in chunks:
        poses.extend(_record_batch_poses(chunk.to_record_batch(), entity_path=entity_path))

    poses.sort(key=lambda pose: pose.timestamp_ns)
    if not poses:
        msg = f"Transform3D entity has no poses in {rrd_path}: {entity_path}"
        raise PipelineResultError(msg)
    return poses


def write_tum_trajectory(path: Path, poses: Iterable[TumPose]) -> None:
    """Write a trajectory in TUM RGB-D format."""
    rows = list(poses)
    if not rows:
        msg = f"Refusing to write empty TUM trajectory: {path}"
        raise TumExportError(msg)

    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(_format_tum_pose(pose) for pose in rows)
    path.write_text(content, encoding="utf-8")


def _record_batch_poses(record_batch: pa.RecordBatch, *, entity_path: str) -> list[TumPose]:
    missing_columns = [
        name
        for name in (FRAME_TIME_COLUMN, TRANSLATION_COLUMN, QUATERNION_COLUMN)
        if name not in record_batch.schema.names
    ]
    if missing_columns:
        msg = f"Transform3D chunk for {entity_path} is missing columns: {', '.join(missing_columns)}"
        raise TumExportError(msg)

    timestamps_ns = record_batch.column(FRAME_TIME_COLUMN).cast(pa.int64()).to_pylist()
    translations = record_batch.column(TRANSLATION_COLUMN)
    quaternions = record_batch.column(QUATERNION_COLUMN)

    poses: list[TumPose] = []
    for row_idx, timestamp_ns in enumerate(timestamps_ns):
        translation = _component_values(
            translations[row_idx].as_py(),
            expected_len=3,
            column=TRANSLATION_COLUMN,
        )
        quaternion = _component_values(
            quaternions[row_idx].as_py(),
            expected_len=4,
            column=QUATERNION_COLUMN,
        )
        poses.append(
            TumPose(
                timestamp_ns=int(timestamp_ns),
                translation=(translation[0], translation[1], translation[2]),
                quaternion_xyzw=(quaternion[0], quaternion[1], quaternion[2], quaternion[3]),
            )
        )
    return poses


def _component_values(raw_value: object, *, expected_len: int, column: str) -> list[float]:
    if not isinstance(raw_value, list):
        msg = f"{column} should be a list component, got {type(raw_value).__name__}"
        raise TumExportError(msg)

    values = raw_value
    if len(values) == 1 and isinstance(values[0], list):
        values = values[0]
    if len(values) != expected_len:
        msg = f"{column} should have {expected_len} values, got {len(values)}"
        raise TumExportError(msg)

    result: list[float] = []
    for value in values:
        if not isinstance(value, int | float):
            msg = f"{column} values should be numeric, got {type(value).__name__}"
            raise TumExportError(msg)
        result.append(float(value))
    return result


def _selected_specs(entity: TumEntity) -> list[TumEntitySpec]:
    if entity == TumEntity.ALL:
        return [ENTITY_SPECS[TumEntity.SLAM_OUTPUT], ENTITY_SPECS[TumEntity.GROUND_TRUTH]]
    return [ENTITY_SPECS[entity]]


def _entity_path_candidates(entity_path: str) -> list[str]:
    if entity_path.startswith("/"):
        return [entity_path, entity_path.removeprefix("/")]
    return [entity_path, f"/{entity_path}"]


def _format_tum_pose(pose: TumPose) -> str:
    tx, ty, tz = pose.translation
    qx, qy, qz, qw = pose.quaternion_xyzw
    return (
        f"{_format_timestamp(pose.timestamp_ns)} "
        f"{tx:.17g} {ty:.17g} {tz:.17g} "
        f"{qx:.17g} {qy:.17g} {qz:.17g} {qw:.17g}\n"
    )


def _format_timestamp(timestamp_ns: int) -> str:
    sec, nsec = divmod(timestamp_ns, 1_000_000_000)
    return f"{sec}.{nsec:09d}"
