import numpy as np

from core.dense_mapping.voxel_schema import VoxelSchema
from core.dense_mapping.voxel_store import VoxelStatus
from pipeline.context import PipelineContext
from visualizer.rerun.modules.voxel_visualize_module import VoxelVisualizeModule


class TestVoxelVisualizeModule:
    """Unit tests for VoxelVisualizeModule."""

    def test_process_logs_colored_voxel_boxes(self, mocker) -> None:
        """Voxel rows should render as colored boxes."""
        log_mock = mocker.patch("visualizer.rerun.modules.voxel_visualize_module.rr.log")
        boxes3d_mock = mocker.patch(
            "visualizer.rerun.modules.voxel_visualize_module.rr.Boxes3D",
            side_effect=lambda **kwargs: ("boxes3d", kwargs),
        )
        module = VoxelVisualizeModule(
            "mapping_confirmed_voxels",
            "/world/estimates/local_map/mapping/voxels/confirmed",
            {
                "points_size_prop_name": "mapping_confirmed_voxels_size",
                "voxel_size_m": 0.1,
                "label_prefix": "obstacle",
            },
        )
        ctx = (
            PipelineContext.from_timestamp(1.0)
            .set_scalar("mapping_confirmed_voxels_size", 2)
            .set_ndarray(
                "mapping_confirmed_voxels",
                self.schema_voxels(),
            )
            .reassemble()
        )

        module.process(ctx)

        assert log_mock.call_args.args[0] == "/world/estimates/local_map/mapping/voxels/confirmed"
        assert log_mock.call_args.args[1] == (
            "boxes3d",
            boxes3d_mock.call_args.kwargs,
        )
        kwargs = boxes3d_mock.call_args.kwargs
        assert np.allclose(kwargs["centers"], np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
        assert np.allclose(kwargs["sizes"], np.full((2, 3), 0.1, dtype=np.float32))
        assert np.array_equal(kwargs["colors"], np.array([[20, 30, 40], [220, 230, 240]], dtype=np.uint8))
        assert np.array_equal(kwargs["labels"], np.array(["obstacle_10_20_30", "obstacle_11_21_31"]))

    def test_process_uses_fallback_color_for_legacy_voxel_rows(self, mocker) -> None:
        """Legacy 5-column voxel rows should still render."""
        mocker.patch("visualizer.rerun.modules.voxel_visualize_module.rr.log")
        points3d_mock = mocker.patch(
            "visualizer.rerun.modules.voxel_visualize_module.rr.Points3D",
            side_effect=lambda **kwargs: ("points3d", kwargs),
        )
        module = VoxelVisualizeModule(
            "mapping_voxels",
            "/world/estimates/local_map/mapping/voxels/raw",
            {
                "points_size_prop_name": "mapping_voxels_size",
                "draw_mode": "points",
                "fallback_color": [1, 2, 3],
            },
        )
        ctx = (
            PipelineContext.from_timestamp(1.0)
            .set_scalar("mapping_voxels_size", 1)
            .set_ndarray(
                "mapping_voxels",
                np.array([[10.0, 1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
            )
            .reassemble()
        )

        module.process(ctx)

        assert np.array_equal(points3d_mock.call_args.kwargs["colors"], np.array([[1, 2, 3]], dtype=np.uint8))

    def schema_voxels(self) -> np.ndarray:
        """Build voxel rows using the dense mapping VoxelSchema layout."""
        voxels = np.zeros((2, VoxelSchema.count()), dtype=np.float32)
        voxels[:, VoxelSchema.VOXEL_KEY] = np.array([[10, 20, 30], [11, 21, 31]], dtype=np.float32)
        voxels[:, VoxelSchema.VOXEL_HITS] = np.array([4, 7], dtype=np.float32)
        voxels[:, VoxelSchema.VOXEL_OBSERVATIONS] = np.array([2, 3], dtype=np.float32)
        voxels[:, VoxelSchema.VOXEL_COLOR] = np.array([[20, 30, 40], [220, 230, 240]], dtype=np.float32)
        voxels[:, VoxelSchema.VOXEL_CENTER] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        voxels[:, VoxelSchema.VOXEL_STATUS] = VoxelStatus.CONFIRMED.value
        return voxels
