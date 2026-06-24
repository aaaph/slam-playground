import json
import uuid
from pathlib import Path

import rerun as rr
from typer.testing import CliRunner

from evaluation.ape import evaluate_ape, format_ape_summary
from evaluation.cli import app, run_ape
from evaluation.tum_export import TumEntity, create_tum_exports
from pipeline.result import PipelineResult

RUN_ID = "019e6e61-d4ab-766e-a886-d44ab4e041cb"
SLAM_ENTITY_PATH = "world/estimates/world_map/slam_output/base_link"
GROUND_TRUTH_ENTITY_PATH = "world/estimates/world_map/ground_truth/base_link"


class TestTumExport:
    """Unit tests for TUM trajectory export."""

    def test_create_tum_exports_writes_selected_entity(self, tmp_path: Path) -> None:
        """Exporting one selected entity should write one TUM file."""
        _write_pipeline_result(tmp_path)
        result = PipelineResult.current(repo_root=tmp_path)

        exports = create_tum_exports(result, entity=TumEntity.SLAM_OUTPUT)

        assert [export.entity for export in exports] == [TumEntity.SLAM_OUTPUT]
        tum_path = tmp_path / "pipeline" / "out" / RUN_ID / "evo" / "slam_output.tum"
        assert exports[0].path == tum_path
        assert exports[0].samples_count == 3
        assert tum_path.read_text(encoding="utf-8").splitlines() == [
            "1.000000000 1 2 3 0 0 0 1",
            "2.500000000 4 5 6 0 0 0 1",
            "4.000000000 7 8 10 0 0 0 1",
        ]

    def test_create_tum_exports_writes_all_entities(self, tmp_path: Path) -> None:
        """The default selector should write both evo trajectories."""
        _write_pipeline_result(tmp_path)
        result = PipelineResult.current(repo_root=tmp_path)

        exports = create_tum_exports(result)

        assert [export.entity for export in exports] == [TumEntity.SLAM_OUTPUT, TumEntity.GROUND_TRUTH]
        assert (tmp_path / "pipeline" / "out" / RUN_ID / "evo" / "slam_output.tum").exists()
        assert (tmp_path / "pipeline" / "out" / RUN_ID / "evo" / "ground_truth_aligned.tum").exists()

    def test_create_tum_cli_uses_latest_run(self, tmp_path: Path) -> None:
        """The CLI should create a selected TUM file from the current latest run."""
        _write_pipeline_result(tmp_path)

        result = CliRunner().invoke(
            app,
            ["create-tum", "--run", "latest", "--entity", "slam_output", "--repo-root", str(tmp_path)],
        )

        assert result.exit_code == 0, result.output
        assert "slam_output:" in result.output
        assert "(3 poses)" in result.output
        assert (tmp_path / "pipeline" / "out" / RUN_ID / "evo" / "slam_output.tum").exists()

    def test_evaluate_ape_exports_dataset_reference_and_metrics(self, tmp_path: Path) -> None:
        """APE evaluation should compare slam output against raw dataset ground truth."""
        _write_pipeline_result(tmp_path)
        pipeline_result = PipelineResult.current(repo_root=tmp_path)

        artifacts = evaluate_ape(pipeline_result)
        summary = format_ape_summary(artifacts)
        lines = summary.splitlines()

        assert lines[0].endswith("datasets/euroc_test/state_groundtruth_estimate0/data.csv")
        assert lines[1].endswith(f"pipeline/out/{RUN_ID}/evo/slam_output.tum")
        assert lines[2] == ""
        assert lines[3] == "APE w.r.t. translation part (m)"
        assert lines[4] == "(with SE(3) Umeyama alignment)"
        assert lines[5] == ""
        stat_lines = lines[6:]
        assert [line.split("\t")[0].strip() for line in stat_lines] == [
            "max",
            "mean",
            "median",
            "min",
            "rmse",
            "sse",
            "std",
        ]
        assert all(line.endswith("0.000000") for line in stat_lines)
        evo_dir = tmp_path / "pipeline" / "out" / RUN_ID / "evo"
        assert (evo_dir / "slam_output.tum").exists()
        assert not (evo_dir / "ground_truth_raw.tum").exists()
        assert not (evo_dir / "ape.zip").exists()
        assert not (evo_dir / "ape.txt").exists()
        assert not (evo_dir / "ape.json").exists()

    def test_evaluate_ape_can_save_result_artifacts(self, tmp_path: Path) -> None:
        """APE result artifact persistence should be opt-in."""
        _write_pipeline_result(tmp_path)
        pipeline_result = PipelineResult.current(repo_root=tmp_path)

        artifacts = evaluate_ape(pipeline_result, save_result=True)
        summary = format_ape_summary(artifacts)

        assert "result_zip:" in summary
        assert "result_json:" in summary
        assert "result_txt:" in summary
        evo_dir = tmp_path / "pipeline" / "out" / RUN_ID / "evo"
        assert (evo_dir / "ape.zip").exists()
        assert (evo_dir / "ape.txt").exists()
        payload = json.loads((evo_dir / "ape.json").read_text(encoding="utf-8"))
        assert payload["dataset"]["name"] == "euroc_test"
        assert payload["dataset"]["type"] == "euroc"
        assert payload["dataset"]["ground_truth_format"] == "euroc_csv"
        assert payload["dataset"]["ground_truth"].endswith(
            "datasets/euroc_test/state_groundtruth_estimate0/data.csv"
        )
        assert payload["alignment"] == "se3_umeyama"
        assert payload["samples_count"] == 3
        assert payload["stats"]["rmse"] < 1e-12

    def test_ape_cli_can_show_plot(self, tmp_path: Path, monkeypatch, capsys) -> None:
        """The APE CLI should expose an opt-in interactive evo plot."""
        _write_pipeline_result(tmp_path)
        plotted = []

        def fake_show_ape_plot(artifacts) -> None:
            plotted.append(artifacts)

        monkeypatch.setattr("evaluation.cli.show_ape_plot", fake_show_ape_plot)

        run_ape(run="latest", repo_root=tmp_path, plot=True)
        output = capsys.readouterr().out

        assert "APE w.r.t. translation part (m)" in output
        assert len(plotted) == 1


def _write_pipeline_result(repo_root: Path) -> None:
    _write_dataset(repo_root)
    log_dir = repo_root / "pipeline" / "out" / RUN_ID
    log_dir.mkdir(parents=True)
    (log_dir / "log_rerun.txt").write_text("rerun log", encoding="utf-8")
    _write_rrd(log_dir / "data.rrd")

    _write_json(
        log_dir / "rerun_manifest.json",
        {
            "files": {"rrd": f"pipeline/out/{RUN_ID}/data.rrd"},
            "runtime_config": {"dataset_name": "euroc_test"},
            "stream_index": [
                {
                    "branch": "slam_frame",
                    "module": "dynamic_transform",
                    "property_name": "slam_pose",
                    "entity_path": SLAM_ENTITY_PATH,
                },
                {
                    "branch": "trajectory_evaluator_frame",
                    "module": "dynamic_transform",
                    "property_name": "ground_truth_aligned_se3",
                    "entity_path": GROUND_TRUTH_ENTITY_PATH,
                },
            ],
        },
    )
    _write_json(
        repo_root / "pipeline" / "out" / "current-run.json",
        {
            "schema_version": 1,
            "status": "completed",
            "dataflow_id": RUN_ID,
            "log_dir": f"pipeline/out/{RUN_ID}",
            "out_dir": "pipeline/out",
            "logs": {"rerun": f"pipeline/out/{RUN_ID}/log_rerun.txt"},
        },
    )


def _write_rrd(path: Path) -> None:
    rr.init("tum-export-test", recording_id=str(uuid.uuid4()), spawn=False)
    rr.save(path)
    try:
        rr.set_time("frame_time", timestamp=1.0)
        rr.log(
            SLAM_ENTITY_PATH,
            rr.Transform3D(translation=[1.0, 2.0, 3.0], quaternion=rr.Quaternion(xyzw=[0.0, 0.0, 0.0, 1.0])),
        )
        rr.log(
            GROUND_TRUTH_ENTITY_PATH,
            rr.Transform3D(
                translation=[10.0, 20.0, 30.0],
                quaternion=rr.Quaternion(xyzw=[0.0, 0.0, 0.0, 1.0]),
            ),
        )

        rr.set_time("frame_time", timestamp=2.5)
        rr.log(
            SLAM_ENTITY_PATH,
            rr.Transform3D(translation=[4.0, 5.0, 6.0], quaternion=rr.Quaternion(xyzw=[0.0, 0.0, 0.0, 1.0])),
        )
        rr.log(
            GROUND_TRUTH_ENTITY_PATH,
            rr.Transform3D(
                translation=[40.0, 50.0, 60.0],
                quaternion=rr.Quaternion(xyzw=[0.0, 0.0, 0.0, 1.0]),
            ),
        )

        rr.set_time("frame_time", timestamp=4.0)
        rr.log(
            SLAM_ENTITY_PATH,
            rr.Transform3D(
                translation=[7.0, 8.0, 10.0],
                quaternion=rr.Quaternion(xyzw=[0.0, 0.0, 0.0, 1.0]),
            ),
        )
        rr.log(
            GROUND_TRUTH_ENTITY_PATH,
            rr.Transform3D(
                translation=[70.0, 80.0, 100.0],
                quaternion=rr.Quaternion(xyzw=[0.0, 0.0, 0.0, 1.0]),
            ),
        )
    finally:
        rr.disconnect()


def _write_dataset(repo_root: Path) -> None:
    dataset_root = repo_root / "datasets" / "euroc_test"
    ground_truth_dir = dataset_root / "state_groundtruth_estimate0"
    ground_truth_dir.mkdir(parents=True)
    (ground_truth_dir / "data.csv").write_text(
        "#timestamp,p_RS_R_x,p_RS_R_y,p_RS_R_z,q_RS_w,q_RS_x,q_RS_y,q_RS_z\n"
        "1000000000,1,2,3,1,0,0,0\n"
        "2500000000,4,5,6,1,0,0,0\n"
        "4000000000,7,8,10,1,0,0,0\n",
        encoding="utf-8",
    )
    _write_json(
        repo_root / "datasets" / "euroc_test.yaml",
        {
            "name": "euroc_test",
            "type": "euroc",
            "root": "datasets/euroc_test",
            "rig": "config/dataset_rig/euroc.yaml",
            "streams": {
                "cam0": "cam0/data.csv",
                "cam1": "cam1/data.csv",
                "imu0": "imu0/data.csv",
                "ground_truth": "state_groundtruth_estimate0/data.csv",
            },
        },
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
