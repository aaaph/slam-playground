import json
import re
from pathlib import Path

import pytest

from pipeline.result import PipelineResult, PipelineResultError

RUN_ID = "019e6e61-d4ab-766e-a886-d44ab4e041cb"


class TestPipelineResult:
    """Unit tests for pipeline result artifact discovery."""

    def test_current_resolves_run_and_evo_inputs(self, tmp_path: Path) -> None:
        """A current-run manifest should resolve Rerun inputs needed by evo."""
        log_dir = _write_run(tmp_path)
        result = PipelineResult.current(repo_root=tmp_path)

        evo_inputs = result.require_evo_rerun_inputs()

        assert result.status == "completed"
        assert result.run_id == RUN_ID
        assert result.log_dir == log_dir
        assert result.rerun_manifest_path == log_dir / "rerun_manifest.json"
        assert result.rerun_blueprint_path == log_dir / "rerun_blueprint.rbl"
        assert result.rrd_path == log_dir / "data.rrd"
        assert result.logs == {"rerun": log_dir / "log_rerun.txt"}
        assert evo_inputs.rrd_path == log_dir / "data.rrd"
        assert evo_inputs.rerun_manifest_path == log_dir / "rerun_manifest.json"
        assert evo_inputs.rerun_blueprint_path == log_dir / "rerun_blueprint.rbl"
        assert evo_inputs.estimate_stream.entity_path == "world/estimates/world_map/slam_output/base_link"
        assert evo_inputs.reference_stream.entity_path == "world/estimates/world_map/ground_truth/base_link"
        assert evo_inputs.output_dir == log_dir / "evo"

    def test_latest_falls_back_to_newest_uuid_run_dir(self, tmp_path: Path) -> None:
        """The latest helper should synthesize a result from the newest UUID log dir."""
        older = _write_run(tmp_path, run_id="019e6e5c-d033-746a-9bfd-358774c72bee", write_current=False)
        newer = _write_run(tmp_path, run_id=RUN_ID, write_current=False)
        older.touch()
        newer.touch()

        result = PipelineResult.latest(repo_root=tmp_path)

        assert result.state_path is None
        assert result.run_id == RUN_ID
        assert result.log_dir == newer
        assert result.logs == {"rerun": newer / "log_rerun.txt"}

    def test_require_evo_inputs_fails_when_stream_is_missing(self, tmp_path: Path) -> None:
        """Missing manifest streams should fail before trying to export trajectories."""
        log_dir = _write_run(tmp_path)
        _write_json(
            log_dir / "rerun_manifest.json",
            {
                "files": {"rrd": "pipeline/out/019e6e61-d4ab-766e-a886-d44ab4e041cb/data.rrd"},
                "stream_index": [
                    {
                        "branch": "slam_frame",
                        "module": "dynamic_transform",
                        "property_name": "slam_pose",
                        "entity_path": "world/estimates/world_map/slam_output/base_link",
                    }
                ],
            },
        )
        result = PipelineResult.current(repo_root=tmp_path)

        with pytest.raises(PipelineResultError, match="ground_truth_aligned_se3"):
            result.require_evo_rerun_inputs()

    def test_require_rerun_recording_fails_when_rrd_is_missing(self, tmp_path: Path) -> None:
        """Missing data.rrd should be reported as a missing required artifact."""
        log_dir = _write_run(tmp_path)
        (log_dir / "data.rrd").unlink()
        result = PipelineResult.current(repo_root=tmp_path)

        with pytest.raises(FileNotFoundError, match=re.escape("data.rrd")):
            result.require_evo_rerun_inputs()


def _write_run(
    repo_root: Path,
    *,
    run_id: str = RUN_ID,
    write_current: bool = True,
) -> Path:
    log_dir = repo_root / "pipeline" / "out" / run_id
    log_dir.mkdir(parents=True)
    (log_dir / "log_rerun.txt").write_text("rerun log", encoding="utf-8")
    (log_dir / "data.rrd").write_bytes(b"rrd")
    (log_dir / "rerun_blueprint.rbl").write_bytes(b"rbl")

    _write_json(
        log_dir / "rerun_manifest.json",
        {
            "files": {
                "rrd": f"pipeline/out/{run_id}/data.rrd",
                "rerun_blueprint": f"pipeline/out/{run_id}/rerun_blueprint.rbl",
            },
            "stream_index": [
                {
                    "branch": "slam_frame",
                    "module": "dynamic_transform",
                    "property_name": "slam_pose",
                    "entity_path": "world/estimates/world_map/slam_output/base_link",
                },
                {
                    "branch": "trajectory_evaluator_frame",
                    "module": "dynamic_transform",
                    "property_name": "ground_truth_aligned_se3",
                    "entity_path": "world/estimates/world_map/ground_truth/base_link",
                },
            ],
        },
    )

    if write_current:
        _write_json(
            repo_root / "pipeline" / "out" / "current-run.json",
            {
                "schema_version": 1,
                "status": "completed",
                "dataflow_id": run_id,
                "log_dir": f"pipeline/out/{run_id}",
                "out_dir": "pipeline/out",
                "logs": {"rerun": f"pipeline/out/{run_id}/log_rerun.txt"},
            },
        )

    return log_dir


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
