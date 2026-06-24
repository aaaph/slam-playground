from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from evo.core import metrics, sync
from evo.main_ape import ape
from evo.tools import file_interface

from dataset.registry import DatasetRegistry
from evaluation.tum_export import TumEntity, create_tum_exports
from pipeline.result import PipelineResult, PipelineResultError

if TYPE_CHECKING:
    from pathlib import Path

    from evo.core.trajectory import PoseTrajectory3D

    from dataset.manifest import DatasetManifest

DEFAULT_MAX_TIMESTAMP_DIFF_SECONDS = 0.01


class ApeEvaluationError(RuntimeError):
    """Raised when offline APE evaluation cannot be completed."""


@dataclass(frozen=True)
class DatasetReference:
    """Dataset reference trajectory loaded directly from the dataset."""

    dataset_name: str
    dataset_type: str
    source_path: Path
    source_format: str
    trajectory: PoseTrajectory3D


@dataclass(frozen=True)
class ApeArtifacts:
    """Artifacts produced by an APE evaluation run."""

    dataset_reference: DatasetReference
    estimate_tum_path: Path
    result_zip_path: Path | None
    result_json_path: Path | None
    result_txt_path: Path | None
    stats: dict[str, float]
    samples_count: int
    aligned: bool
    pretty_output: str


def evaluate_ape(
    result: PipelineResult,
    *,
    align: bool = True,
    max_timestamp_diff_seconds: float = DEFAULT_MAX_TIMESTAMP_DIFF_SECONDS,
    save_result: bool = False,
) -> ApeArtifacts:
    """Export trajectories for a run and compute evo APE against dataset ground truth."""
    output_dir = result.evo_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_reference = load_dataset_reference(result)
    estimate_export = create_tum_exports(result, entity=TumEntity.SLAM_OUTPUT, output_dir=output_dir)[0]

    traj_ref = dataset_reference.trajectory
    traj_est = file_interface.read_tum_trajectory_file(estimate_export.path)
    synced_ref, synced_est = sync.associate_trajectories(
        traj_ref,
        traj_est,
        max_diff=max_timestamp_diff_seconds,
        first_name="dataset ground truth",
        snd_name="slam_output",
    )

    ape_result = ape(
        synced_ref,
        synced_est,
        metrics.PoseRelation.translation_part,
        align=align,
        ref_name="dataset_ground_truth",
        est_name="slam_output",
    )
    stats = {str(key): float(value) for key, value in ape_result.stats.items()}
    pretty_output = ape_result.pretty_str()

    if save_result:
        result_zip_path = output_dir / "ape.zip"
        result_json_path = output_dir / "ape.json"
        result_txt_path = output_dir / "ape.txt"
        file_interface.save_res_file(result_zip_path, ape_result, confirm_overwrite=False)
        result_txt_path.write_text(pretty_output, encoding="utf-8")
        result_json_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "dataset": {
                        "name": dataset_reference.dataset_name,
                        "type": dataset_reference.dataset_type,
                        "ground_truth": str(dataset_reference.source_path),
                        "ground_truth_format": dataset_reference.source_format,
                    },
                    "estimate_tum": str(estimate_export.path),
                    "result_zip": str(result_zip_path),
                    "result_text": str(result_txt_path),
                    "alignment": "se3_umeyama" if align else "none",
                    "pose_relation": "translation_part",
                    "max_timestamp_diff_seconds": max_timestamp_diff_seconds,
                    "samples_count": len(synced_est.timestamps),
                    "stats": stats,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    else:
        result_zip_path = None
        result_json_path = None
        result_txt_path = None

    return ApeArtifacts(
        dataset_reference=dataset_reference,
        estimate_tum_path=estimate_export.path,
        result_zip_path=result_zip_path,
        result_json_path=result_json_path,
        result_txt_path=result_txt_path,
        stats=stats,
        samples_count=len(synced_est.timestamps),
        aligned=align,
        pretty_output=pretty_output,
    )


def load_dataset_reference(result: PipelineResult) -> DatasetReference:
    """Load the run's dataset ground truth trajectory directly from the dataset."""
    dataset_name = result.require_dataset_name()
    registry = DatasetRegistry(repo_root=result.repo_root)
    manifest = registry.find(dataset_name)
    if manifest.type == "euroc":
        return _load_euroc_reference(registry=registry, manifest=manifest)

    msg = f"APE dataset reference export is not implemented for dataset type '{manifest.type}'"
    raise ApeEvaluationError(msg)


def _load_euroc_reference(
    *,
    registry: DatasetRegistry,
    manifest: DatasetManifest,
) -> DatasetReference:
    ground_truth_stream = manifest.streams.ground_truth
    if ground_truth_stream is None:
        msg = f"Dataset '{manifest.name}' has no ground_truth stream"
        raise PipelineResultError(msg)

    root = registry.resolve_path(manifest.root)
    ground_truth_path = ground_truth_stream if ground_truth_stream.is_absolute() else root / ground_truth_stream
    if not ground_truth_path.exists():
        msg = f"Dataset ground truth file not found: {ground_truth_path}"
        raise FileNotFoundError(msg)

    trajectory = file_interface.read_euroc_csv_trajectory(ground_truth_path)
    return DatasetReference(
        dataset_name=manifest.name,
        dataset_type=manifest.type,
        source_path=ground_truth_path,
        source_format="euroc_csv",
        trajectory=trajectory,
    )


def format_ape_summary(artifacts: ApeArtifacts) -> str:
    """Build evo-style human-readable output for the CLI."""
    lines = [
        f"reference: {artifacts.dataset_reference.source_path}",
        f"estimate: {artifacts.estimate_tum_path}",
        "",
        artifacts.pretty_output.rstrip("\n"),
    ]
    if artifacts.result_zip_path is not None:
        lines.extend(
            [
                "",
                f"result_zip: {artifacts.result_zip_path}",
                f"result_json: {artifacts.result_json_path}",
                f"result_txt: {artifacts.result_txt_path}",
            ]
        )
    return "\n".join(lines)
