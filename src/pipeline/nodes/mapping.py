import time
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from core.camera_model.stereo_camera_model import StereoCameraModel
from core.dense_mapping.depth_filter import DepthFilter, DepthFilterConfig
from core.dense_mapping.point_cloud_builder import PointCloudBuilder
from core.dense_mapping.stereo_depth_estimator import (
    PostprocessingMode,
    PreprocessingMode,
    StereoDepthEstimator,
    StereoDepthEstimatorConfig,
    StereoSGBMConfig,
)
from core.dense_mapping.voxel_map import VoxelConfig as VoxelMapConfig
from core.dense_mapping.voxel_map import VoxelMap
from core.dense_mapping.voxel_schema import VoxelSchema
from core.dense_mapping.voxel_store import VoxelStatus
from core.transformations.special_euclidian_3_dim import SE3
from logger import spawn_logger
from pipeline.annotations import Ctx
from pipeline.decorators import on_input, on_stop, reactive, to_output
from pipeline.nodes.base import PipelineNode

GRAYSCALE_IMAGE_NDIM = 2


@dataclass(frozen=True)
class VoxelConfig:
    """Voxel configuration."""

    voxel_size_m: float = 0.1
    depth_stride_px: int = 8
    min_confirmed_hits: int = 8
    min_confirmed_observations: int = 2


@dataclass
class Voxel:
    """Aggregated occupied voxel endpoint evidence."""

    hits: int
    observations: int
    centroid: NDArray[np.float32]
    color_rgb: NDArray[np.float32]
    last_seen_ts: float


@reactive
class MappingNode(PipelineNode):
    """Mapping node."""

    def __init__(self, camera_model: StereoCameraModel) -> None:
        """Initialize the mapping node."""
        self.logger = spawn_logger(app="mapping")
        self.stereo_ctx = camera_model.as_stereo_ctx()
        self.depth_filter = self.create_depth_filter(DepthFilterConfig())
        self.depth_estimator = self.create_depth_estimator(
            camera_model,
        )
        self.point_cloud_builder = PointCloudBuilder.default_factory(self.stereo_ctx.stereo_k)
        self.voxel_map = VoxelMap.default_factory(VoxelMapConfig())
        self.voxel_config = VoxelConfig()
        self.voxels: dict[tuple[int, int, int], Voxel] = {}

    @staticmethod
    def create_depth_estimator(
        camera_model: StereoCameraModel,
    ) -> StereoDepthEstimator:
        """Create the stereo depth estimator used by mapping."""
        return StereoDepthEstimator(
            camera_model,
            StereoDepthEstimatorConfig(
                preprocessing_mode=PreprocessingMode.BLUR | PreprocessingMode.CLAHE,
                postprocessing_mode=(
                    PostprocessingMode.NONE if not StereoDepthEstimator.supports_wls() else PostprocessingMode.WLS
                ),
                sgbm=StereoSGBMConfig(
                    min_disparity=0,
                    num_disparities=96,
                    block_size=5,
                    disp12_max_diff=1,
                    uniqueness_ratio=15,
                    speckle_window_size=150,
                    speckle_range=2,
                    pre_filter_cap=63,
                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
                ),
            ),
        )

    def create_depth_filter(self, config: DepthFilterConfig) -> DepthFilter:
        """Create the mapping depth filter from stereo calibration."""
        focal_baseline_m = float(self.stereo_ctx.stereo_k[0, 0]) * float(self.stereo_ctx.baseline)
        return DepthFilter(config, focal_baseline_m=focal_baseline_m)

    @on_input("keyframes")
    @to_output("frame")
    def handle_keyframes(self, ctx: Ctx) -> Ctx:
        """Handle the keyframe."""
        pose_estimate_se3 = SE3.from_matrix(ctx.get_ndarray("pose_estimate", (4, 4)))
        height, width = ctx.get_scalar("height"), ctx.get_scalar("width")
        left, right = ctx.get_image("left", (height, width)), ctx.get_image("right", (height, width))
        t1 = time.perf_counter()
        raw_depth = self.depth_estimator.estimate_depth(left, right)
        filtered_depth = self.depth_filter.apply(raw_depth)
        t2 = time.perf_counter()
        self.logger.info(f"[Mapping]: Depth estimation took {((t2 - t1) * 1000):.2f} ms")

        valid_count = int(np.count_nonzero(filtered_depth.valid_mask))
        if valid_count == 0:
            self.logger.warning("[Mapping]: No valid depth pixels found -> No update to mapping")
            self.append_depth_output(ctx, filtered_depth.depth_m)
            self.append_points_in_odom_output(ctx, np.empty((0, 3), dtype=np.float32))
            self.append_voxel_outputs(ctx)
            return ctx

        cam0_in_odom = pose_estimate_se3 * self.stereo_ctx.cam0_in_body_se3
        t3 = time.perf_counter()
        point_cloud_observations = self.point_cloud_builder.build_from_depth(cam0_in_odom, filtered_depth)
        voxel_observations = self.voxel_map.builder.build_from_point_cloud(point_cloud_observations)
        t4 = time.perf_counter()
        self.logger.info(f"[Mapping]: Point cloud and Voxel building took {((t4 - t3) * 1000):.2f} ms")
        updated_voxels_count = self.voxel_map.integrate_voxels(voxel_observations)
        self.logger.info(
            f"[Mapping]: Updated {updated_voxels_count} voxels to the voxel map. "
            f"Map size: {self.voxel_map.map_size()}"
        )
        t5 = time.perf_counter()
        self.logger.info(f"[Mapping]: Voxel integration took {((t5 - t4) * 1000):.2f} ms")
        self.append_depth_output(ctx, filtered_depth.depth_m)
        self.append_points_in_odom_output(ctx, point_cloud_observations[:, :3])
        self.append_voxel_outputs(ctx)
        return ctx

    def append_voxel_outputs(self, ctx: Ctx) -> Ctx:
        """Attach voxel pointclouds to the mapping output context."""
        mapping_voxels = self.voxel_map.store.active_voxel_view()
        mapping_confirmed_voxels = mapping_voxels[
            mapping_voxels[:, VoxelSchema.VOXEL_STATUS] == VoxelStatus.CONFIRMED.value
        ]
        return (
            ctx.set_ndarray("mapping_voxels", mapping_voxels)
            .set_scalar("mapping_voxels_size", mapping_voxels.shape[0])
            .set_ndarray("mapping_confirmed_voxels", mapping_confirmed_voxels)
            .set_scalar("mapping_confirmed_voxels_size", mapping_confirmed_voxels.shape[0])
        )

    def append_depth_output(self, ctx: Ctx, depth: NDArray[np.float32]) -> Ctx:
        """Attach the filtered metric depth image to the mapping output context."""
        return ctx.set_ndarray("mapping_depth", depth.astype(np.float32, copy=False))

    def append_points_in_odom_output(self, ctx: Ctx, points_in_odom: NDArray[np.float32]) -> Ctx:
        """Attach sampled depth endpoints in odom coordinates to the mapping output context."""
        return ctx.set_ndarray("points_in_odom", points_in_odom).set_scalar(
            "points_in_odom_size",
            points_in_odom.shape[0],
        )

    @on_stop
    def shutdown(self) -> None:
        """Shutdown the mapping node."""
        self.logger.info("Mapping node stopped")


if __name__ == "__main__":
    MappingNode(camera_model=MappingNode.create_stereo_camera_model()).run()
