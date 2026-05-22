from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from core.camera_model.stereo_camera_ctx import StereoContext
from core.loop_closure.vpr_frame import VPRDetection, VPRGeometrySchema

type Mask = NDArray[np.bool_]
type Geometry = NDArray[np.float32]
type Image = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class VPRDetectorConfig:
    """Visual Place Recognition detector configuration."""

    stereo_k_matrix: np.ndarray
    resolution: tuple[int, int]
    grid_rows: int
    grid_cols: int
    region_limit: int
    baseline: float
    disparity_min_threshold: float
    depth_min_threshold: float
    depth_max_threshold: float
    vertical_shift_threshold: float


class VPRDetector:
    """Visual Place Recognition detector."""

    def __init__(self, detector: cv2.ORB, config: VPRDetectorConfig) -> None:
        """Initialize the VPR detector."""
        self.detector = detector
        self.config = config
        self.grid = self._spawn_grid()
        self.stereo_k_inv = np.linalg.inv(self.config.stereo_k_matrix)
        self.stereo_klt_params = {
            "winSize": (24, 24),
            "maxLevel": 3,
            "criteria": (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
        }

    @classmethod
    def from_stereo_ctx(
        cls, stereo_ctx: StereoContext, n_features: int = 1000, grid: tuple[int, int] = (1, 1)
    ) -> "VPRDetector":
        """Create a VPR detector from a stereo context."""
        rows_count, cols_count = grid
        n_features = n_features // (rows_count * cols_count)
        detector = cv2.ORB.create(
            patchSize=31,
            edgeThreshold=31,
            fastThreshold=20,
            WTA_K=2,
            scaleFactor=1.2,
            nlevels=8,
            nfeatures=n_features,
            firstLevel=0,
            scoreType=cv2.ORB_HARRIS_SCORE,
        )
        config = VPRDetectorConfig(
            stereo_k_matrix=stereo_ctx.stereo_k,
            resolution=stereo_ctx.resolution,
            grid_rows=rows_count,
            grid_cols=cols_count,
            region_limit=n_features,
            baseline=stereo_ctx.baseline,
            disparity_min_threshold=5.0,
            depth_min_threshold=0.15,
            depth_max_threshold=40.0,
            vertical_shift_threshold=10.0,
        )
        return cls(
            detector,
            config,
        )

    def _spawn_grid(self) -> list[np.ndarray]:
        """Spawn a grid of regions."""
        rows, cols = self.config.grid_rows, self.config.grid_cols
        grid = []
        for row in range(rows):
            for col in range(cols):
                mask = np.zeros((self.config.resolution[1], self.config.resolution[0]), dtype=np.uint8)
                mask[
                    row * self.config.resolution[1] // rows : (row + 1) * self.config.resolution[1] // rows,
                    col * self.config.resolution[0] // cols : (col + 1) * self.config.resolution[0] // cols,
                ] = 1
                grid.append(mask)
        return grid

    def _select_strongest_kps(
        self, kps: Sequence[cv2.KeyPoint], descriptors: np.ndarray
    ) -> tuple[Sequence[cv2.KeyPoint], np.ndarray]:
        """Select the strongest keypoints."""
        if len(kps) == 0 or descriptors is None:
            return [], np.empty((0, 32), dtype=np.uint8)
        count = min(len(kps), descriptors.shape[0])
        keypoints = kps[:count]
        descriptors = descriptors[:count]

        responses = np.array([kp.response for kp in keypoints], dtype=np.float32)
        indices = np.argsort(responses)[::-1][: self.config.region_limit]
        selected_keypoints = [keypoints[i] for i in indices]
        selected_descriptors = np.ascontiguousarray(descriptors[indices], dtype=np.uint8)
        return selected_keypoints, selected_descriptors

    def _match_stereo_klt(self, geometry: Geometry, left: Image, right: Image) -> tuple[Geometry, Mask]:
        """Match the stereo KLT features."""
        if len(geometry) == 0:
            return np.empty((0, VPRGeometrySchema.count()), dtype=np.float32), np.empty((0,), dtype=bool)
        p0 = geometry[:, VPRGeometrySchema.LEFT_U : VPRGeometrySchema.LEFT_V + 1].astype(np.float32)
        points_right, st_left_right, _err_left_right = cv2.calcOpticalFlowPyrLK(
            left, right, p0, None, **self.stereo_klt_params
        )  # ty: ignore
        points_back, st_right_left, _err_right_left = cv2.calcOpticalFlowPyrLK(
            right, left, points_right, None, **self.stereo_klt_params
        )  # ty: ignore

        forward_back_err = np.linalg.norm(p0 - points_back, axis=1).ravel()
        forward_back_mask = forward_back_err < 1.0
        ul, vl = p0[:, 0], p0[:, 1]
        ur, vr = points_right[:, 0], points_right[:, 1]
        disp = ul - ur

        max_disparity = 64
        epipolar_mask = np.abs(vl - vr) < 1.0
        disparity_mask = (disp > 0) & (disp < max_disparity)

        mask = (
            (st_left_right.ravel() == 1)
            & (st_right_left.ravel() == 1)
            & forward_back_mask
            & epipolar_mask
            & disparity_mask
        )
        matched_geometry = geometry.copy()
        matched_geometry[:, VPRGeometrySchema.RIGHT_U : VPRGeometrySchema.RIGHT_V + 1] = np.nan
        matched_geometry[mask, VPRGeometrySchema.RIGHT_U : VPRGeometrySchema.RIGHT_V + 1] = points_right[mask]
        return matched_geometry[mask], mask

    def _triangulate_stereo_geometry(self, geometry: Geometry) -> tuple[Geometry, Mask]:
        """Triangulate the stereo geometry."""
        if len(geometry) == 0:
            return np.empty((0, VPRGeometrySchema.count()), dtype=np.float32), np.empty((0,), dtype=bool)
        left_uv = geometry[:, VPRGeometrySchema.LEFT_U : VPRGeometrySchema.LEFT_V + 1]
        right_uv = geometry[:, VPRGeometrySchema.RIGHT_U : VPRGeometrySchema.RIGHT_V + 1]

        fx = self.config.stereo_k_matrix[0, 0]
        fy = self.config.stereo_k_matrix[1, 1]
        cx = self.config.stereo_k_matrix[0, 2]
        cy = self.config.stereo_k_matrix[1, 2]
        baseline = self.config.baseline

        with np.errstate(divide="ignore", invalid="ignore"):
            disp = left_uv[:, 0] - right_uv[:, 0]
            z = fx * baseline / disp
            x = (left_uv[:, 0] - cx) * z / fx
            y = (left_uv[:, 1] - cy) * z / fy

        bad_parallax = disp < self.config.disparity_min_threshold
        too_close = z < self.config.depth_min_threshold
        too_far = z > self.config.depth_max_threshold
        vertical_shift = np.abs(left_uv[:, 1] - right_uv[:, 1]) > self.config.vertical_shift_threshold
        is_nan = np.isnan(disp) | np.isnan(x) | np.isnan(y) | np.isnan(z)

        bad_feat_mask = bad_parallax | too_close | too_far | vertical_shift | is_nan

        mask = np.invert(bad_feat_mask)

        triang_geometry = geometry.copy()
        triang_geometry[:, VPRGeometrySchema.POINT_X : VPRGeometrySchema.POINT_Z + 1] = np.nan

        triang_geometry[mask, VPRGeometrySchema.POINT_X : VPRGeometrySchema.POINT_Z + 1] = np.column_stack(
            [x[mask], y[mask], z[mask]]
        )
        return triang_geometry[mask], mask

    def detect_stereo(self, left_image: np.ndarray, right_image: np.ndarray) -> VPRDetection:
        """Detect the features in the stereo images."""
        geometry_buffer = np.zeros((500, VPRGeometrySchema.count()), dtype=np.float32)
        descriptor_buffer = np.zeros((500, 32), dtype=np.uint8)
        buffer_index = 0

        for mask in self.grid:
            region_kps, region_desc = self.detector.detectAndCompute(left_image, mask)
            region_kps, region_desc = self._select_strongest_kps(region_kps, region_desc)
            if len(region_kps) == 0 or region_desc is None:
                continue
            uv = np.ascontiguousarray([kp.pt for kp in region_kps], dtype=np.float32).reshape(-1, 2)
            uv_h = np.column_stack([uv, np.ones(uv.shape[0], dtype=np.float32)])
            bearing_vectors = uv_h @ self.stereo_k_inv.T
            bearing_vectors = bearing_vectors / np.linalg.norm(bearing_vectors, axis=1, keepdims=True)
            region_geometry = np.full((uv.shape[0], VPRGeometrySchema.count()), np.nan, dtype=np.float32)
            region_geometry[:, VPRGeometrySchema.LEFT_U : VPRGeometrySchema.LEFT_V + 1] = uv
            region_geometry[:, VPRGeometrySchema.BEARING_X : VPRGeometrySchema.BEARING_Z + 1] = bearing_vectors
            region_descriptors = np.ascontiguousarray(region_desc, dtype=np.uint8)
            if buffer_index + len(region_geometry) >= len(geometry_buffer):
                geometry_buffer = np.concatenate(
                    [geometry_buffer, np.zeros((500, VPRGeometrySchema.count()), dtype=np.float32)]
                )
                descriptor_buffer = np.concatenate([descriptor_buffer, np.zeros((500, 32), dtype=np.uint8)])
            geometry_buffer[buffer_index : buffer_index + len(region_geometry)] = region_geometry
            descriptor_buffer[buffer_index : buffer_index + len(region_descriptors)] = region_descriptors
            buffer_index += len(region_geometry)

        geometry = geometry_buffer[:buffer_index]
        descriptors = descriptor_buffer[:buffer_index]
        stereo_geometry, stereo_mask = self._match_stereo_klt(geometry, left_image, right_image)
        stereo_descriptors = descriptors[stereo_mask]

        triang_geometry, triang_mask = self._triangulate_stereo_geometry(stereo_geometry)
        triang_descriptors = stereo_descriptors[triang_mask]
        return VPRDetection(
            geometry=triang_geometry,
            descriptors=triang_descriptors,
        )
