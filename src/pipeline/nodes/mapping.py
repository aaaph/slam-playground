from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from core.camera_model.stereo_camera_model import StereoCameraModel
from core.front_end.keyframe import KF
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


@dataclass(frozen=True)
class DepthFilterConfig:
    """Filtering thresholds for stereo depth used by dense mapping."""

    min_depth_m: float = 0.3
    max_depth_m: float = 12.0
    min_disparity_px: float = 1.0
    median_kernel_size: int = 5
    mask_open_kernel_size: int = 3


@reactive
class MappingNode(PipelineNode):
    """Mapping node."""

    def __init__(self, camera_model: StereoCameraModel) -> None:
        """Initialize the mapping node."""
        self.logger = spawn_logger(app="mapping")
        self.stereo_ctx = camera_model.as_stereo_ctx()
        self.depth_filter = DepthFilterConfig()
        self.voxel_config = VoxelConfig()
        self.voxels: dict[tuple[int, int, int], Voxel] = {}
        self.matcher = cv2.StereoSGBM.create(
            minDisparity=0,
            numDisparities=96,
            blockSize=5,
            P1=8 * 1 * 5 * 5,
            P2=32 * 1 * 5 * 5,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    @on_input("keyframes")
    @to_output("frame")
    def handle_keyframes(self, ctx: Ctx) -> Ctx:
        """Handle the keyframe."""
        timestamp_ns = ctx.get_scalar("timestamp")
        keyframes = KF.list_from_arrow(ctx.get_record_batch("keyframes"))
        kf = keyframes[0]
        kf_id = kf.keyframe_id
        pose_estimate = ctx.get_ndarray("pose_estimate", (4, 4))
        pose_estimate_se3 = SE3.from_matrix(pose_estimate)

        height = ctx.get_scalar("height")
        width = ctx.get_scalar("width")

        left_rect = ctx.get_image("left_rect", (height, width))
        right_rect = ctx.get_image("right_rect", (height, width))

        self.logger.info(f"Processing keyframe {kf_id} at timestamp {timestamp_ns}")

        disp, depth, valid_mask = self.compute_depth(left_rect, right_rect)
        ctx = self.append_depth_output(ctx, depth)
        valid_count = int(np.count_nonzero(valid_mask))
        valid_ratio = valid_count / valid_mask.size if valid_mask.size else 0.0

        if valid_count == 0:
            self.logger.warning(f"Depth shape: {depth.shape}, valid_ratio={valid_ratio:.3f}, no valid depth")
            return self.append_voxel_outputs(ctx)

        valid_depth = depth[valid_mask]

        cam0_in_odom = pose_estimate_se3 * self.stereo_ctx.cam0_in_body_se3
        frame_points_odom, frame_colors_rgb = self.depth_to_odom_points_with_colors(
            depth,
            valid_mask,
            left_rect,
            cam0_in_odom,
        )
        updated_voxels = self.integrate_points(frame_points_odom, timestamp_ns, frame_colors_rgb)
        confirmed_voxels = self.confirmed_voxel_count()
        self.logger.info(
            f"Depth shape: {depth.shape}, disparity shape: {disp.shape}, valid_ratio={valid_ratio:.3f}, "
            f"depth_min={np.min(valid_depth):.3f}, depth_median={np.median(valid_depth):.3f}, "
            f"depth_max={np.max(valid_depth):.3f}, "
            f"frame_points={frame_points_odom.shape[0]}, updated_voxels={updated_voxels}, "
            f"total_voxels={len(self.voxels)}, confirmed_voxels={confirmed_voxels}, "
            f"pose_estimate_se3={pose_estimate_se3}"
        )
        return self.append_voxel_outputs(ctx)

    def compute_depth(
        self,
        left_rect: np.ndarray,
        right_rect: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute filtered disparity and depth from a rectified stereo pair."""
        disp = self.matcher.compute(left_rect, right_rect).astype(np.float32) / 16.0
        return self.filter_disparity(disp)

    def filter_disparity(self, disp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter stereo disparity and convert it to a sparse depth image."""
        disp = disp.astype(np.float32, copy=True)
        invalid_disparity = ~np.isfinite(disp) | (disp <= self.depth_filter.min_disparity_px)
        disp[invalid_disparity] = 0.0

        if self.depth_filter.median_kernel_size > 1:
            disp = cv2.medianBlur(disp, self.depth_filter.median_kernel_size)

        with np.errstate(divide="ignore", invalid="ignore"):
            depth = self.stereo_ctx.stereo_k[0, 0] * self.stereo_ctx.baseline / disp

        valid_mask = (
            np.isfinite(depth)
            & (disp > self.depth_filter.min_disparity_px)
            & (depth >= self.depth_filter.min_depth_m)
            & (depth <= self.depth_filter.max_depth_m)
        )

        if self.depth_filter.mask_open_kernel_size > 1:
            kernel_size = self.depth_filter.mask_open_kernel_size
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            valid_mask = cv2.morphologyEx(valid_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)

        filtered_depth = np.zeros_like(depth, dtype=np.float32)
        filtered_depth[valid_mask] = depth[valid_mask]
        filtered_disp = np.zeros_like(disp, dtype=np.float32)
        filtered_disp[valid_mask] = disp[valid_mask]
        return filtered_disp, filtered_depth, valid_mask

    def depth_to_odom_points(
        self,
        depth: NDArray[np.float32],
        valid_mask: NDArray[np.bool_],
        cam0_in_odom: SE3,
    ) -> NDArray[np.float32]:
        """Back-project valid depth pixels and transform endpoints into odom frame."""
        rows, cols = self.sampled_depth_pixel_indices(valid_mask)
        return self.depth_pixels_to_odom_points(depth, rows, cols, cam0_in_odom)

    def depth_to_odom_points_with_colors(
        self,
        depth: NDArray[np.float32],
        valid_mask: NDArray[np.bool_],
        image: NDArray[np.uint8],
        cam0_in_odom: SE3,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Back-project valid depth pixels and return their source image colors."""
        rows, cols = self.sampled_depth_pixel_indices(valid_mask)
        return (
            self.depth_pixels_to_odom_points(depth, rows, cols, cam0_in_odom),
            self.pixel_colors_rgb(image, rows, cols),
        )

    def sampled_depth_pixel_indices(
        self,
        valid_mask: NDArray[np.bool_],
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Return sampled valid depth pixel coordinates."""
        if not np.any(valid_mask):
            empty = np.empty((0,), dtype=np.intp)
            return empty, empty

        stride = max(1, self.voxel_config.depth_stride_px)
        sampled_mask = np.zeros_like(valid_mask, dtype=bool)
        sampled_mask[::stride, ::stride] = valid_mask[::stride, ::stride]
        rows, cols = np.nonzero(sampled_mask)
        return rows, cols

    def depth_pixels_to_odom_points(
        self,
        depth: NDArray[np.float32],
        rows: NDArray[np.intp],
        cols: NDArray[np.intp],
        cam0_in_odom: SE3,
    ) -> NDArray[np.float32]:
        """Back-project selected depth pixels and transform endpoints into odom frame."""
        if rows.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        z = depth[rows, cols].astype(np.float64, copy=False)
        k_matrix = self.stereo_ctx.stereo_k
        fx = float(k_matrix[0, 0])
        fy = float(k_matrix[1, 1])
        cx = float(k_matrix[0, 2])
        cy = float(k_matrix[1, 2])

        x = (cols.astype(np.float64) - cx) * z / fx
        y = (rows.astype(np.float64) - cy) * z / fy
        points_cam0 = np.column_stack((x, y, z))

        rotation = cam0_in_odom.rotation().as_matrix()
        translation = cam0_in_odom.translation()
        points_odom = points_cam0 @ rotation.T + translation
        return points_odom.astype(np.float32, copy=False)

    @staticmethod
    def pixel_colors_rgb(
        image: NDArray[np.uint8],
        rows: NDArray[np.intp],
        cols: NDArray[np.intp],
    ) -> NDArray[np.float32]:
        """Return RGB colors for sampled pixels, promoting grayscale to RGB."""
        if rows.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        if image.ndim == GRAYSCALE_IMAGE_NDIM:
            gray = image[rows, cols].astype(np.float32, copy=False)
            return np.repeat(gray[:, None], 3, axis=1)

        samples = image[rows, cols].astype(np.float32, copy=False)
        if samples.ndim == 1:
            return np.repeat(samples[:, None], 3, axis=1)
        if samples.shape[1] == 1:
            return np.repeat(samples, 3, axis=1)
        return samples[:, :3]

    def integrate_points(
        self,
        points_odom: NDArray[np.float32],
        timestamp_ns: float,
        colors_rgb: NDArray[np.float32] | None = None,
    ) -> int:
        """Integrate occupied endpoint points into the voxel hash map."""
        if points_odom.size == 0:
            return 0
        if colors_rgb is None:
            colors_rgb = np.full((points_odom.shape[0], 3), 155.0, dtype=np.float32)
        if colors_rgb.shape != (points_odom.shape[0], 3):
            msg = f"colors_rgb shape {colors_rgb.shape} does not match points shape {points_odom.shape}"
            raise ValueError(msg)

        voxel_keys = np.floor(points_odom / self.voxel_config.voxel_size_m).astype(np.int32)
        unique_keys, inverse, counts = np.unique(voxel_keys, axis=0, return_inverse=True, return_counts=True)
        centroid_sums = np.zeros((unique_keys.shape[0], 3), dtype=np.float64)
        np.add.at(centroid_sums, inverse, points_odom.astype(np.float64, copy=False))
        centroids = (centroid_sums / counts[:, None]).astype(np.float32)
        color_sums = np.zeros((unique_keys.shape[0], 3), dtype=np.float64)
        np.add.at(color_sums, inverse, colors_rgb.astype(np.float64, copy=False))
        color_centroids = (color_sums / counts[:, None]).astype(np.float32)

        for key_array, hit_count, centroid, color_rgb in zip(
            unique_keys,
            counts,
            centroids,
            color_centroids,
            strict=True,
        ):
            key = (int(key_array[0]), int(key_array[1]), int(key_array[2]))
            voxel = self.voxels.get(key)
            if voxel is None:
                self.voxels[key] = Voxel(
                    hits=int(hit_count),
                    observations=1,
                    centroid=centroid.copy(),
                    color_rgb=color_rgb.copy(),
                    last_seen_ts=timestamp_ns,
                )
                continue

            total_hits = voxel.hits + int(hit_count)
            voxel.centroid = (
                (voxel.centroid.astype(np.float64) * voxel.hits + centroid.astype(np.float64) * int(hit_count))
                / total_hits
            ).astype(np.float32)
            voxel.color_rgb = (
                (voxel.color_rgb.astype(np.float64) * voxel.hits + color_rgb.astype(np.float64) * int(hit_count))
                / total_hits
            ).astype(np.float32)
            voxel.hits = total_hits
            voxel.observations += 1
            voxel.last_seen_ts = timestamp_ns

        return int(unique_keys.shape[0])

    def confirmed_voxel_count(self) -> int:
        """Count voxels stable enough to be considered occupied obstacles."""
        return sum(
            voxel.hits >= self.voxel_config.min_confirmed_hits
            and voxel.observations >= self.voxel_config.min_confirmed_observations
            for voxel in self.voxels.values()
        )

    def voxel_pointcloud(self, *, confirmed_only: bool) -> NDArray[np.float32]:
        """Return voxel rows shaped [id, x, y, z, hits, r, g, b]."""
        rows: list[list[float]] = []
        for voxel_id, (key, voxel) in enumerate(sorted(self.voxels.items())):
            if confirmed_only and not self.is_confirmed_voxel(voxel):
                continue
            color_rgb = np.clip(voxel.color_rgb, 0.0, 255.0)
            center = self.voxel_center_from_key(key)
            rows.append(
                [
                    float(voxel_id),
                    float(center[0]),
                    float(center[1]),
                    float(center[2]),
                    float(voxel.hits),
                    float(color_rgb[0]),
                    float(color_rgb[1]),
                    float(color_rgb[2]),
                ]
            )
        if not rows:
            return np.empty((0, 8), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    def voxel_center_from_key(self, key: tuple[int, int, int]) -> NDArray[np.float32]:
        """Return the center of a voxel grid cell for visualization and occupancy output."""
        return (np.asarray(key, dtype=np.float32) + 0.5) * self.voxel_config.voxel_size_m

    def is_confirmed_voxel(self, voxel: Voxel) -> bool:
        """Check whether a voxel has enough evidence to be treated as an obstacle."""
        return (
            voxel.hits >= self.voxel_config.min_confirmed_hits
            and voxel.observations >= self.voxel_config.min_confirmed_observations
        )

    def append_voxel_outputs(self, ctx: Ctx) -> Ctx:
        """Attach voxel pointclouds to the mapping output context."""
        mapping_voxels = self.voxel_pointcloud(confirmed_only=False)
        mapping_confirmed_voxels = self.voxel_pointcloud(confirmed_only=True)
        return (
            ctx.set_ndarray("mapping_voxels", mapping_voxels)
            .set_scalar("mapping_voxels_size", mapping_voxels.shape[0])
            .set_ndarray("mapping_confirmed_voxels", mapping_confirmed_voxels)
            .set_scalar("mapping_confirmed_voxels_size", mapping_confirmed_voxels.shape[0])
        )

    def append_depth_output(self, ctx: Ctx, depth: NDArray[np.float32]) -> Ctx:
        """Attach the filtered metric depth image to the mapping output context."""
        return ctx.set_ndarray("mapping_depth", depth.astype(np.float32, copy=False))

    @on_stop
    def shutdown(self) -> None:
        """Shutdown the mapping node."""
        self.logger.info("Mapping node stopped")


if __name__ == "__main__":
    MappingNode(camera_model=MappingNode.create_stereo_camera_model()).run()
