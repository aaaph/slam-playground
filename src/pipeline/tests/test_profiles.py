import json
from pathlib import Path
from typing import Any, cast

import pytest

from pipeline.profiles import PipelineProfileResolver, ProfileOverrides, RunMode, VisualizationSink
from pipeline.runtime_config import PIPELINE_NODE_CONFIG_ENV


class TestPipelineProfileResolver:
    """Tests for pipeline profile resolution."""

    def test_resolve_quick_vio_euroc_profile(self) -> None:
        """Resolve a complete composite profile."""
        resolved = PipelineProfileResolver(repo_root=Path.cwd()).resolve(profile="quick_vio_euroc")

        assert resolved.repo_root == Path.cwd().resolve()
        assert resolved.profile == "quick_vio_euroc"
        assert resolved.dataset.name == "euroc_mh_01"
        assert resolved.dataset.type == "euroc"
        assert resolved.dataset.rig == Path("config/dataset_rig/euroc.yaml")
        assert resolved.dataset.streams.cam0 == Path("cam0/data.csv")
        assert resolved.rig.name == "euroc"
        assert resolved.rig.cam0.resolution == (752, 480)
        assert resolved.rig.cam0.body_sensor_transform.rows == 4
        assert resolved.dataflow.name == "vio-dataflow.yml"
        assert resolved.dataflow.template == Path("pipeline/vio-dataflow.yml")
        assert resolved.dataflow.build is False
        runtime_nodes = cast("list[dict[str, Any]]", resolved.dataflow.runtime_dataflow["nodes"])
        runtime_env_by_id = {str(node["id"]): cast("dict[str, Any]", node["env"]) for node in runtime_nodes}
        assert list(runtime_env_by_id) == ["control", "dataset", "frontend", "fixed_lag_smoother", "rerun"]
        control_config = json.loads(runtime_env_by_id["control"][PIPELINE_NODE_CONFIG_ENV])
        dataset_config = json.loads(runtime_env_by_id["dataset"][PIPELINE_NODE_CONFIG_ENV])
        assert control_config["emit_ready_status"] is False
        assert control_config["expected_ready_nodes"] == [
            "dataset",
            "frontend",
            "fixed_lag_smoother",
            "rerun",
        ]
        assert control_config["ready_inputs"] == {
            "dataset_status": "dataset",
            "fixed_lag_smoother_status": "fixed_lag_smoother",
            "frontend_status": "frontend",
            "rerun_status": "rerun",
        }
        assert dataset_config["emit_ready_status"] is True
        assert resolved.visualization.sink == VisualizationSink.FILE
        assert resolved.run.mode == RunMode.BATCH_FRACTION
        assert resolved.run.fraction == 0.05
        assert resolved.run.autostart_after_ready is True
        assert resolved.run.stop_after_dataset_done is True

    def test_load_profile_keeps_profile_name(self) -> None:
        """Composite profiles carry their own explicit name."""
        profile = PipelineProfileResolver(repo_root=Path.cwd()).load_profile("quick_vio_euroc")

        assert profile.name == "quick_vio_euroc"

    def test_cli_overrides_replace_only_explicit_fields(self) -> None:
        """CLI overrides should not erase unrelated profile fields."""
        resolved = PipelineProfileResolver(repo_root=Path.cwd()).resolve(
            profile="my_slam_euroc",
            overrides=ProfileOverrides(
                dataflow="vio-dataflow.yml",
                visualization_sink=VisualizationSink.BOTH,
            ),
        )

        assert resolved.dataset.name == "euroc_mh_01"
        assert resolved.dataflow.name == "vio-dataflow.yml"
        assert resolved.dataflow.build is False
        assert resolved.visualization.sink == VisualizationSink.BOTH
        assert resolved.run.mode == RunMode.MANUAL

    def test_resolve_dataset_viz_profile_parses_status_routes(self) -> None:
        """Parsed dataflow exposes dynamic status producers and routes."""
        resolved = PipelineProfileResolver(repo_root=Path.cwd()).resolve(profile="dataset_viz")

        runtime_nodes = cast("list[dict[str, Any]]", resolved.dataflow.runtime_dataflow["nodes"])
        runtime_nodes_by_id = {str(node["id"]): node for node in runtime_nodes}
        runtime_env_by_id = {str(node["id"]): cast("dict[str, Any]", node["env"]) for node in runtime_nodes}
        assert PIPELINE_NODE_CONFIG_ENV in runtime_env_by_id["control"]
        assert PIPELINE_NODE_CONFIG_ENV in runtime_env_by_id["dataset"]
        assert PIPELINE_NODE_CONFIG_ENV in runtime_env_by_id["rerun"]
        control_config = json.loads(runtime_env_by_id["control"][PIPELINE_NODE_CONFIG_ENV])
        dataset_config = json.loads(runtime_env_by_id["dataset"][PIPELINE_NODE_CONFIG_ENV])
        rerun_config = json.loads(runtime_env_by_id["rerun"][PIPELINE_NODE_CONFIG_ENV])
        assert control_config["expected_ready_nodes"] == ["dataset", "rerun"]
        assert control_config["ready_inputs"] == {
            "dataset_status": "dataset",
            "rerun_status": "rerun",
        }
        assert dataset_config["emit_ready_status"] is True
        assert rerun_config["emit_ready_status"] is True
        control_inputs = cast("dict[str, Any]", runtime_nodes_by_id["control"]["inputs"])
        assert control_inputs["startup_tick"] == "dora/timer/millis/100"
        assert control_inputs["dataset_status"] == "dataset/status"
        assert control_inputs["rerun_status"] == "rerun/status"
        assert "status" in cast("list[str]", runtime_nodes_by_id["dataset"]["outputs"])
        assert "status" in cast("list[str]", runtime_nodes_by_id["rerun"]["outputs"])

    def test_fraction_override_implies_batch_fraction_mode(self) -> None:
        """A fraction override should switch a manual profile to batch-fraction mode."""
        resolved = PipelineProfileResolver(repo_root=Path.cwd()).resolve(
            profile="my_slam_euroc",
            overrides=ProfileOverrides(fraction=0.05),
        )

        assert resolved.run.mode == RunMode.BATCH_FRACTION
        assert resolved.run.fraction == 0.05

    def test_missing_required_selector_raises(self) -> None:
        """Without profile or explicit selectors, resolution cannot proceed."""
        with pytest.raises(ValueError, match="dataset must be provided"):
            PipelineProfileResolver(repo_root=Path.cwd()).resolve()

    def test_legacy_dataflow_string_defaults_build_to_false(self, tmp_path: Path) -> None:
        """Profiles can still use the old dataflow string syntax."""
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()
        (profile_dir / "legacy.yaml").write_text(
            """
name: legacy
dataset: euroc_mh_01
dataflow: vio-dataflow.yml
""".lstrip(),
            encoding="utf-8",
        )

        resolved = PipelineProfileResolver(repo_root=Path.cwd(), profile_dir=profile_dir).resolve(profile="legacy")

        assert resolved.dataflow.name == "vio-dataflow.yml"
        assert resolved.dataflow.build is False
