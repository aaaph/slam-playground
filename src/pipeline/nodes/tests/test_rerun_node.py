import json
from pathlib import Path
from unittest.mock import Mock

import pyarrow as pa

from pipeline.annotations import EXECUTION_TIME_MS_METADATA_FIELD
from pipeline.context import PipelineContext
from pipeline.nodes.rerun_node import RerunNode
from pipeline.runtime_config import RerunNodeRuntimeConfig, RerunNodeSink
from visualizer.rerun.factories.rerun_config_factory import RerunConfigFactory
from visualizer.rerun.loaders import RerunConfigLoader
from visualizer.rerun.schemas import EntitySchema, ModuleType, RerunConfigSchema, ViewSchema, ViewType


def find_view_by_name(views: list[ViewSchema], name: str) -> ViewSchema:
    """Find a view recursively by name."""
    view = find_view_by_name_or_none(views, name)
    if view is None:
        raise StopIteration
    return view


def find_view_by_name_or_none(views: list[ViewSchema], name: str) -> ViewSchema | None:
    """Find a view recursively by name, returning None when absent."""
    for view in views:
        if view.name == name:
            return view
        child_view = find_view_by_name_or_none(view.views, name)
        if child_view is not None:
            return child_view
    return None


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

    def test_handle_mapping_frame_forwards_mapping_branch(self) -> None:
        """Mapping frames should use a dedicated Rerun branch."""
        node = RerunNode.__new__(RerunNode)
        node.visualize_branch = Mock()
        ctx = PipelineContext.from_timestamp(1.0)
        record_batch = pa.RecordBatch.from_arrays([], schema=pa.schema([]))

        node.handle_mapping_frame(ctx, record_batch)

        node.visualize_branch.assert_called_once_with("mapping_frame", ctx, record_batch)

    def test_slam_config_camera_map_view_excludes_dense_mapping_streams(self) -> None:
        """The SLAM Camera&Map view should contain only local-map and rectified-image views."""
        config = RerunConfigLoader.from_path(Path("config/visualization/slam_view_config.yaml"))

        camera_map = find_view_by_name(config.views, "Camera&Map")
        local_map = find_view_by_name(config.views, "Local Map")
        rectified_left = find_view_by_name(config.views, "Rectified Left Image")

        assert camera_map.type == ViewType.CONTAINER
        assert [view.name for view in camera_map.views] == ["Local Map", "Rectified Left Image"]
        assert rectified_left.branch == "frontend_frame"
        assert all(stream.branch != "mapping_frame" for stream in local_map.streams)
        assert find_view_by_name_or_none(config.views, "Mapping Depth") is None

    def test_slam_config_includes_local_map_point_covariance_stream(self) -> None:
        """The Local Map view should visualize accumulated local-map point covariance."""
        config = RerunConfigLoader.from_path(Path("config/visualization/slam_view_config.yaml"))

        local_map = find_view_by_name(config.views, "Local Map")
        local_map_stream = next(
            stream
            for stream in local_map.streams
            if stream.branch == "frontend_frame"
            and stream.id == "local_map_points"
            and stream.entity == "world/estimates/local_map/map/landmarks"
        )

        assert local_map_stream.module == ModuleType.POINTCLOUD
        assert local_map_stream.options["points_size_prop_name"] == "local_map_points_size"
        assert local_map_stream.options["visualize_covariance"] is True
        assert local_map_stream.options["covariance_color"] == [80, 220, 255]

    def test_smart_factor_metrics_are_logged_below_view_origin(self) -> None:
        """Smart-factor scalar entities must be visible from their TimeSeries origin."""
        config = RerunConfigLoader.from_path(Path("config/visualization/slam_view_config.yaml"))

        smart_factor_fit = find_view_by_name(config.views, "Smart-Factor Fit")

        assert all(stream.entity.startswith(f"{smart_factor_fit.origin}/") for stream in smart_factor_fit.streams)

    def test_vio_local_map_visualizes_frontend_points_under_selected_pose(self) -> None:
        """The Local Map view should attach frontend points to the selected pose."""
        config = RerunConfigLoader.from_path(Path("config/visualization/vio_view_config.yaml"))

        local_map = find_view_by_name(config.views, "Local Map")
        frontend_streams = {
            stream.entity: stream for stream in local_map.streams if stream.branch == "frontend_frame"
        }
        selected_stream = frontend_streams["world/estimates/local_map/selected/base_link"]
        stereo_stream = frontend_streams["world/estimates/local_map/selected/base_link/cam0/landmarks"]

        assert selected_stream.id == "pose_estimate"
        assert selected_stream.module == ModuleType.DYNAMIC_TRANSFORM
        assert selected_stream.options["show_axes"] is True
        assert all(stream.id not in {"pim_pose", "pnp_pose"} for stream in frontend_streams.values())
        assert stereo_stream.id == "stereo_points"
        assert stereo_stream.module == ModuleType.POINTCLOUD
        assert stereo_stream.options["points_size_prop_name"] == "stereo_points_size"
        assert all(stream.id != "initialized_landmarks" for stream in frontend_streams.values())

    def test_slam_config_excludes_mapping_node_execution_stream(self) -> None:
        """Reactive metrics should not include MappingNode when dense mapping is disabled."""
        config = RerunConfigLoader.from_path(Path("config/visualization/slam_view_config.yaml"))

        node_execution = next(view for view in config.views if view.name == "Node Execution Time")
        assert all(stream.branch != "mapping_frame" for stream in node_execution.streams)
        assert all(stream.options.get("arrow_field") != "MappingNode" for stream in node_execution.streams)

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
