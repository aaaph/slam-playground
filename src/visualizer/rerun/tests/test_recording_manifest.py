from visualizer.rerun.recording_manifest import build_rerun_stream_index
from visualizer.rerun.schemas import EntitySchema, ModuleType, RerunConfigSchema, ViewSchema, ViewType


class TestRerunRecordingManifest:
    """Unit tests for agent-readable rerun recording manifests."""

    def test_build_stream_index_expands_plot_column_mapping(self) -> None:
        """Plot-column streams should expose concrete scalar entity paths."""
        config = RerunConfigSchema(
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
                                "time_idx": "imu_ts",
                                "timeline": "sim_time",
                                "mapping": [
                                    {"index": 0, "label": "gyro_x"},
                                    {"index": 1, "label": "gyro_y"},
                                ],
                            },
                        )
                    ],
                )
            ]
        )

        [stream] = build_rerun_stream_index(config)

        assert stream["branch"] == "dataset_frame"
        assert stream["property_name"] == "gyro"
        assert stream["entity_path"] == "/sensors/imu/gyro"
        assert stream["logged_entities"] == [
            {
                "entity_path": "/sensors/imu/gyro/gyro_x",
                "component": "Scalars:scalars",
                "timeline": "sim_time",
                "source_column": 0,
                "label": "gyro_x",
            },
            {
                "entity_path": "/sensors/imu/gyro/gyro_y",
                "component": "Scalars:scalars",
                "timeline": "sim_time",
                "source_column": 1,
                "label": "gyro_y",
            },
        ]

    def test_build_stream_index_describes_depth_image_stream(self) -> None:
        """Depth image streams should expose their logged depth image component."""
        config = RerunConfigSchema(
            views=[
                ViewSchema(
                    name="Mapping Depth",
                    type=ViewType.SPATIAL_2D,
                    branch="mapping_frame",
                    origin="/mapping/depth",
                    streams=[
                        EntitySchema(
                            id="mapping_depth",
                            module=ModuleType.DEPTH_IMAGE,
                            entity=".",
                        )
                    ],
                )
            ]
        )

        [stream] = build_rerun_stream_index(config)

        assert stream["branch"] == "mapping_frame"
        assert stream["property_name"] == "mapping_depth"
        assert stream["logged_entities"] == [
            {
                "entity_path": "/mapping/depth",
                "component": "DepthImage",
                "timeline": "sim_time",
            }
        ]
