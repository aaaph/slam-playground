import json
from pathlib import Path
from unittest.mock import Mock

import pyarrow as pa

from pipeline.annotations import EXECUTION_TIME_MS_METADATA_FIELD
from pipeline.context import PipelineContext
from pipeline.nodes.rerun_node import RerunNode
from pipeline.runtime_config import RerunNodeRuntimeConfig, RerunNodeSink
from visualizer.rerun.factories.rerun_config_factory import RerunConfigFactory
from visualizer.rerun.schemas import EntitySchema, ModuleType, RerunConfigSchema, ViewSchema, ViewType


class TestRerunNode:
    """Test Rerun node."""

    def test_build_app_name_includes_dataset_name(self) -> None:
        """Rerun app names should identify the dataset and dataflow run."""
        assert RerunNode.build_app_name(dataset_name="euroc_mh_01", dataflow_id="run-123") == "euroc_mh_01_run-123"

    def test_build_app_name_falls_back_without_dataset_name(self) -> None:
        """Manual rerun node runs without runtime dataset metadata keep the old prefix."""
        assert RerunNode.build_app_name(dataset_name=None, dataflow_id="run-123") == "rerun_run-123"

    def test_visualize_branch_materializes_execution_time_metadata(self) -> None:
        """RerunNode should store execution metadata in the context before visualization."""
        node = RerunNode.__new__(RerunNode)
        node.logger = Mock()
        node.vizualize = Mock()

        record_batch = pa.RecordBatch.from_pydict({"DatasetNode": [1.25]})
        node.visualize_branch("dataset_frame", PipelineContext.from_timestamp(1.0), record_batch)

        branch, sent_ctx = node.vizualize.send.call_args.args[0]
        assert branch == "dataset_frame"
        assert sent_ctx.get_record_batch(EXECUTION_TIME_MS_METADATA_FIELD).schema.names == ["DatasetNode"]

    def test_visualize_branch_accepts_empty_execution_time_metadata(self) -> None:
        """Empty execution metadata should still be a valid context field."""
        node = RerunNode.__new__(RerunNode)
        node.logger = Mock()
        node.vizualize = Mock()

        record_batch = pa.RecordBatch.from_arrays([], schema=pa.schema([]))
        node.visualize_branch("dataset_frame", PipelineContext.from_timestamp(1.0), record_batch)

        branch, sent_ctx = node.vizualize.send.call_args.args[0]
        assert branch == "dataset_frame"
        assert sent_ctx.get_record_batch(EXECUTION_TIME_MS_METADATA_FIELD).schema.names == []

    def test_resolve_save_path_uses_profile_output(self, tmp_path) -> None:
        """Relative profile output paths should resolve under the repo root."""
        node = RerunNode.__new__(RerunNode)
        node.node = Mock()
        config = RerunNodeRuntimeConfig(
            node_id="rerun",
            repo_root=tmp_path,
            sink=RerunNodeSink.FILE,
            output=Path("artifacts/data.rrd"),
        )

        assert node.resolve_save_path(config) == tmp_path / "artifacts" / "data.rrd"

    def test_resolve_save_path_defaults_to_dataflow_run_dir(self, tmp_path) -> None:
        """File sinks without explicit output should save next to pipeline run artifacts."""
        node = RerunNode.__new__(RerunNode)
        node.node = Mock()
        node.node.dataflow_id.return_value = "run-123"
        config = RerunNodeRuntimeConfig(node_id="rerun", repo_root=tmp_path, sink=RerunNodeSink.FILE)

        assert node.resolve_save_path(config) == tmp_path / "pipeline" / "out" / "run-123" / "data.rrd"

    def test_resolve_save_path_skips_app_sink(self, tmp_path) -> None:
        """App-only sinks should not create an RRD output path."""
        node = RerunNode.__new__(RerunNode)
        node.node = Mock()
        config = RerunNodeRuntimeConfig(node_id="rerun", repo_root=tmp_path, sink=RerunNodeSink.APP)

        assert node.resolve_save_path(config) is None

    def test_write_recording_artifacts(self, monkeypatch, tmp_path) -> None:
        """Rerun sidecar artifacts should describe the recording and stream index."""
        monkeypatch.setenv("VISUALIZE_CONFIG", "config/visualization/dataset_view_config.yaml")
        node = RerunNode.__new__(RerunNode)
        node.logger = Mock()
        node.node = Mock()
        node.node.dataflow_id.return_value = "run-123"
        node.node_runtime_config = RerunNodeRuntimeConfig(
            node_id="rerun",
            repo_root=tmp_path,
            dataset_name="euroc_mh_01",
            sink=RerunNodeSink.FILE,
        )
        node.config = RerunConfigSchema(
            app_name="euroc_mh_01_run-123",
            views=[
                ViewSchema(
                    name="Gyro",
                    type=ViewType.TIME_SERIES,
                    branch="dataset_frame",
                    origin="/sensors/imu/gyro",
                    streams=[
                        EntitySchema(
                            id="gyro",
                            module=ModuleType.PLOT_COLUMN,
                            entity=".",
                            options={
                                "mapping": [
                                    {"index": 0, "label": "gyro_x"},
                                ]
                            },
                        )
                    ],
                )
            ],
        )
        node.vizualizer = RerunConfigFactory.from_config(node.config, spawn=False)
        save_path = tmp_path / "pipeline" / "out" / "run-123" / "data.rrd"

        node.write_recording_artifacts(save_path)

        blueprint_path = save_path.parent / "rerun_blueprint.rbl"
        manifest_path = save_path.parent / "rerun_manifest.json"
        assert blueprint_path.exists()
        assert blueprint_path.stat().st_size > 0
        assert not (save_path.parent / "rerun_config.json").exists()
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["dataflow_id"] == "run-123"
        assert manifest["app_name"] == "euroc_mh_01_run-123"
        assert manifest["files"]["rrd"] == str(save_path)
        assert manifest["files"]["rerun_blueprint"] == str(blueprint_path)
        assert manifest["source_config_path"] == "config/visualization/dataset_view_config.yaml"
        assert manifest["stream_index"][0]["logged_entities"][0]["entity_path"] == "/sensors/imu/gyro/gyro_x"
