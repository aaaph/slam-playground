from collections.abc import Sequence
from enum import Enum, auto
from typing import Literal, NamedTuple

import cv2
import numpy as np
from numpy.typing import NDArray

from core.camera_model.stereo_camera_ctx import StereoContext
from core.feature_tracker.feature_frame import FeatureFrame
from core.feature_tracker.feature_schema import FeatureLifecycle, FeatureSchema, StereoMatchSchema
from core.feature_tracker.feature_tensor import FeatureTensor
from core.feature_tracker.feature_tracker_region import FeatureTrackerRegion
from core.feature_tracker.helper import grid_factor
from logger import spawn_logger

MIN_ESSENTIAL_MATRIX_POINTS = 5
RETRACK_MIN_DISTANCE_PX = 20


class FeatureTrackerMode(Enum):
    """Feature tracker mode."""

    STEREO = auto()
    MONOCULAR = auto()


class FeatureTrackerConfig(NamedTuple):
    """Feature tracker configuration."""

    shift_margin: tuple[int, int, int, int] = (16, 16, 16, 16)
    region_amount: int = 8
    optical_flow_klt_win_size: tuple[int, int] = (8, 8)
    stereo_klt_win_size: tuple[int, int] = (24, 24)
    feat_amount_per_region: int = 25
    feat_retrack_threshold: int = 20
    image_shape: tuple[int, int] = (752, 480)
    mode: FeatureTrackerMode = FeatureTrackerMode.MONOCULAR
    temporal_forward_backward_threshold: float = 2.0
    temporal_max_flow_px: float = 100.0
    temporal_flow_mad_multiplier: float = 5.0
    temporal_min_flow_gate_px: float = 30.0


class FeatureTracker:
    """Feature tracker (Stereo Implementation)."""

    def __init__(
        self,
        k_matrix: np.ndarray,
        feature_tracker_config: FeatureTrackerConfig,
    ) -> None:
        """Initialize the feature tracker."""
        self.logger = spawn_logger(app="feature_tracker")
        if feature_tracker_config.region_amount % 2 != 0:
            raise ValueError("Region must be a multiple of two")
        self.SHIFT_MARGIN = (
            feature_tracker_config.shift_margin
        )  # how many pixels are cut off from the all sides of the image
        self.SHIFT_MARGIN_DICT: dict[Literal["left", "right", "top", "bottom"], int] = {
            "left": feature_tracker_config.shift_margin[0],
            "right": feature_tracker_config.shift_margin[1],
            "top": feature_tracker_config.shift_margin[2],
            "bottom": feature_tracker_config.shift_margin[3],
        }
        self.REGION_AMOUNT = feature_tracker_config.region_amount
        self.FEAT_PER_REGION = feature_tracker_config.feat_amount_per_region
        self.FEAT_RETRACK_THRESHOLD = feature_tracker_config.feat_retrack_threshold
        self.TEMPORAL_FORWARD_BACKWARD_THRESHOLD = feature_tracker_config.temporal_forward_backward_threshold
        self.TEMPORAL_MAX_FLOW_PX: int | float = feature_tracker_config.temporal_max_flow_px
        self.TEMPORAL_FLOW_MAD_MULTIPLIER = feature_tracker_config.temporal_flow_mad_multiplier
        self.TEMPORAL_MIN_FLOW_GATE_PX = feature_tracker_config.temporal_min_flow_gate_px
        if self.FEAT_RETRACK_THRESHOLD > self.FEAT_PER_REGION:
            raise ValueError("feat_retrack_threshold > feat_amount_per_region")
        self.optical_flow_klt_params = {
            "winSize": feature_tracker_config.optical_flow_klt_win_size,
            "maxLevel": 3,
            "criteria": (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
        }
        self.stereo_optical_flow_klt_params = {
            "winSize": feature_tracker_config.stereo_klt_win_size,
            "maxLevel": 3,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        }
        self.IMAGE_SHAPE: dict[Literal["w", "h"], int] = {
            "w": feature_tracker_config.image_shape[0],
            "h": feature_tracker_config.image_shape[1],
        }
        self.grid: list[FeatureTrackerRegion] = self._spawn_grid()
        self.grid_mask = self._spawn_grid_mask()
        self.mode = feature_tracker_config.mode

        self.k_matrix = k_matrix
        self.fast = cv2.FastFeatureDetector.create()
        self.tensor: FeatureTensor = FeatureTensor.default_factory(capacity=1000, history_capacity=2)

        self.left_prev: np.ndarray = np.empty(
            (feature_tracker_config.image_shape[0], feature_tracker_config.image_shape[1]), dtype=np.float32
        )
        self.right_prev: np.ndarray = np.empty(
            (feature_tracker_config.image_shape[0], feature_tracker_config.image_shape[1]), dtype=np.float32
        )
        self.ts_prev = -1.0
        self.iterator_count = 0
        self.next_feat_id = 0
        self.median_disparity = 0.0
        self.temporal_pixel_displacement = 0.0

    @classmethod
    def default_factory(
        cls,
        stereo_ctx: StereoContext,
        feat_amount_per_region: int = 25,
        feat_retrack_threshold: int = 10,
        region_amount: int = 8,
        mode: FeatureTrackerMode = FeatureTrackerMode.STEREO,
    ) -> "FeatureTracker":
        """Create a default feature tracker."""
        k_matrix = stereo_ctx.stereo_k if mode == FeatureTrackerMode.STEREO else stereo_ctx.cam0_k

        return cls(
            k_matrix,
            FeatureTrackerConfig(
                feat_amount_per_region=feat_amount_per_region,
                feat_retrack_threshold=feat_retrack_threshold,
                region_amount=region_amount,
                mode=mode,
            ),
        )

    def _spawn_grid(self) -> list[FeatureTrackerRegion]:
        """Spawn a grid of regions."""
        w, h = self.IMAGE_SHAPE["w"], self.IMAGE_SHAPE["h"]
        rows, cols = grid_factor(self.REGION_AMOUNT)
        left_shift, right_shift, top_shift, bottom_shift = self.SHIFT_MARGIN
        shift_mask = np.zeros((h, w), dtype=np.uint8)
        shift_mask[top_shift : h - bottom_shift, left_shift : w - right_shift] = 1

        # create regions
        rows_per_region = h // rows
        cols_per_region = w // cols
        region_masks: list[FeatureTrackerRegion] = []
        index = 0
        for row_index in range(rows):
            for col_index in range(cols):
                row_start = row_index * rows_per_region
                row_end = (row_index + 1) * rows_per_region
                col_start = col_index * cols_per_region
                col_end = (col_index + 1) * cols_per_region

                mask = np.zeros((h, w), dtype=np.uint8)
                mask[row_start:row_end, col_start:col_end] = 1
                mask[shift_mask == 0] = 0
                region = FeatureTrackerRegion(index, mask)
                region_masks.append(region)
                index += 1

        return region_masks

    def _stereo_match_lk(
        self,
        left: NDArray[np.float32],  # (H, W)
        right: NDArray[np.float32],  # (H, W)
        points_left: NDArray[np.float32],  # (N, 3)
    ) -> NDArray[np.float32]:  # (N, 6)
        """KLT Stereo matching between left and right images."""
        if len(points_left) == 0:
            return np.empty((0, StereoMatchSchema.count()), dtype=np.float32)
        p0 = points_left[:, 1:].astype(np.float32)
        points_right, st_left_right, _err_left_right = cv2.calcOpticalFlowPyrLK(
            left, right, p0, None, **self.stereo_optical_flow_klt_params
        )  # ty: ignore
        points_back, st_right_left, _err_right_left = cv2.calcOpticalFlowPyrLK(
            right, left, points_right, None, **self.stereo_optical_flow_klt_params
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
        n_points = points_left.shape[0]
        result = np.full((n_points, StereoMatchSchema.count()), np.nan, dtype=np.float32)
        result[:, StereoMatchSchema.FEAT_ID : StereoMatchSchema.LEFT_V + 1] = points_left
        result[:, StereoMatchSchema.STEREO_OK] = mask.astype(np.float32)
        result[mask, StereoMatchSchema.RIGHT_U : StereoMatchSchema.RIGHT_V + 1] = points_right[mask]
        return result

    # @timeit
    def _optical_flow_lk(self, left_next: np.ndarray, prev_feat_frame: FeatureFrame) -> np.ndarray:
        prev_feat_data = prev_feat_frame.data[prev_feat_frame.active_mask]
        good_feat_mask = prev_feat_data[:, FeatureSchema.LIFECYCLE] != FeatureLifecycle.LOST.value
        prev_good_feat_data = prev_feat_data[good_feat_mask]
        active_points = np.column_stack(
            [
                prev_good_feat_data[:, FeatureSchema.FEAT_ID],
                prev_good_feat_data[:, FeatureSchema.LEFT_U : FeatureSchema.LEFT_V + 1],
            ]
        )
        if active_points.shape[0] == 0:
            self.temporal_pixel_displacement = 0.0
            return np.empty((0, FeatureSchema.count()), dtype=np.float32)
        new_batch = prev_good_feat_data.copy()
        # klt flow
        p0_initial = active_points[:, 1:].astype(np.float32).copy()
        p0 = p0_initial.copy()
        p_next = p0_initial.copy()  # zero motion prediction
        p1, st_fwd, _err = cv2.calcOpticalFlowPyrLK(
            self.left_prev, left_next, p0, p_next, **self.optical_flow_klt_params
        )  # ty: ignore
        p0_back, st_back, _err_back = cv2.calcOpticalFlowPyrLK(
            left_next, self.left_prev, p1, p0_initial.copy(), **self.optical_flow_klt_params
        )  # ty: ignore

        # forward-backward consistency check
        forward_back_err = np.linalg.norm(p0_initial - p0_back, axis=1)
        forward_back_mask = (
            (st_fwd.ravel() == 1)
            & (st_back.ravel() == 1)
            & (forward_back_err < self.TEMPORAL_FORWARD_BACKWARD_THRESHOLD)
        )
        # flow magnitude check
        flow = np.linalg.norm(p1 - p0_initial, axis=1)
        flow_mask = np.zeros(new_batch.shape[0], dtype=bool)
        if np.any(forward_back_mask):
            candidate_flow = flow[forward_back_mask]
            median_flow = float(np.median(candidate_flow))
            mad_flow = float(np.median(np.abs(candidate_flow - median_flow)))
            adaptive_flow_limit = median_flow + self.TEMPORAL_FLOW_MAD_MULTIPLIER * max(mad_flow, 1.0)
            flow_limit = min(
                self.TEMPORAL_MAX_FLOW_PX,
                max(self.TEMPORAL_MIN_FLOW_GATE_PX, adaptive_flow_limit),
            )
            flow_mask = flow < flow_limit

        valid_flow_mask = forward_back_mask & flow_mask
        valid_track_mask = valid_flow_mask.copy()

        new_batch[valid_track_mask, FeatureSchema.LEFT_U : FeatureSchema.LEFT_V + 1] = p1[valid_track_mask]
        new_batch[~valid_track_mask, FeatureSchema.LIFECYCLE] = FeatureLifecycle.LOST.value
        # RANSAC matrix check
        if np.count_nonzero(valid_track_mask) >= MIN_ESSENTIAL_MATRIX_POINTS:
            points1 = new_batch[valid_track_mask, FeatureSchema.LEFT_U : FeatureSchema.LEFT_V + 1]
            points0 = p0_initial[valid_track_mask]
            _, inliners = cv2.findEssentialMat(
                points1,
                points0,
                cameraMatrix=self.k_matrix,
                method=cv2.RANSAC,
                threshold=2.5,
            )
            if inliners is not None:
                inliner_mask = inliners.ravel().astype(bool)
                full_inliner_mask = np.zeros(new_batch.shape[0], dtype=bool)
                full_inliner_mask[valid_track_mask] = inliner_mask
                new_batch[valid_track_mask & ~full_inliner_mask, FeatureSchema.LIFECYCLE] = (
                    FeatureLifecycle.LOST.value
                )
                valid_track_mask &= full_inliner_mask

        # points in bounds check
        if np.any(valid_track_mask):
            good_new_left = new_batch[valid_track_mask, FeatureSchema.LEFT_U : FeatureSchema.LEFT_V + 1]
            in_bounds_mask = self._points_in_bounds(good_new_left[:, 0], good_new_left[:, 1])
            full_in_bounds_mask = np.zeros(new_batch.shape[0], dtype=bool)
            full_in_bounds_mask[valid_track_mask] = in_bounds_mask
            new_batch[valid_track_mask & ~full_in_bounds_mask, FeatureSchema.LIFECYCLE] = (
                FeatureLifecycle.LOST.value
            )
            valid_track_mask &= full_in_bounds_mask

        if np.any(valid_track_mask):
            self.temporal_pixel_displacement = float(np.median(flow[valid_track_mask]))
        else:
            self.temporal_pixel_displacement = 0.0

        new_batch[:, FeatureSchema.AGE] += 1
        return new_batch

    def feed_first(self, timestamp: float, stereo: tuple[np.ndarray, np.ndarray]) -> FeatureFrame:
        """Feed the first frame."""
        left_prev, right_prev = np.asarray(stereo[0]), np.asarray(stereo[1])

        keypoints: list[cv2.KeyPoint] = []
        for region in self.grid:
            kps = self.fast.detect(image=left_prev, mask=np.asarray(region.mask))
            kps = FeatureTracker._select_spread_kps(kps, self.FEAT_PER_REGION, 50)
            keypoints.extend(kps)

        keypoints_mapped = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
        keypoints_array = np.array(keypoints_mapped, dtype=np.float32).reshape(-1, 2)
        first_points = np.column_stack([np.arange(keypoints_array.shape[0]), keypoints_array])
        self.next_feat_id = keypoints_array.shape[0]

        if self.mode == FeatureTrackerMode.STEREO:
            stereo_match = self._stereo_match_lk(left_prev, right_prev, first_points)
            stereo_only_mask = stereo_match[:, StereoMatchSchema.STEREO_OK].astype(bool)
            stereo_match = stereo_match[stereo_only_mask]
            batch = np.full((stereo_match.shape[0], FeatureSchema.count()), np.nan, dtype=np.float32)
            batch[:, FeatureSchema.FEAT_ID] = stereo_match[:, StereoMatchSchema.FEAT_ID]
            batch[:, FeatureSchema.TIMESTAMP] = timestamp
            batch[:, FeatureSchema.LEFT_U : FeatureSchema.RIGHT_V + 1] = stereo_match[
                :,
                StereoMatchSchema.LEFT_U : StereoMatchSchema.RIGHT_V + 1,
            ]
            batch[:, FeatureSchema.LIFECYCLE] = FeatureLifecycle.ACTIVE.value
            batch[:, FeatureSchema.AGE] = 0.0
            batch[:, FeatureSchema.STEREO_SCORE] = 0.0
        else:
            batch = np.full((first_points.shape[0], FeatureSchema.count()), np.nan, dtype=np.float32)
            batch[:, FeatureSchema.FEAT_ID] = first_points[:, 0]
            batch[:, FeatureSchema.TIMESTAMP] = timestamp
            batch[:, FeatureSchema.LEFT_U : FeatureSchema.LEFT_V + 1] = first_points[:, 1:3]
            batch[:, FeatureSchema.LIFECYCLE] = FeatureLifecycle.ACTIVE.value
            batch[:, FeatureSchema.AGE] = 0.0
            batch[:, FeatureSchema.STEREO_SCORE] = 0.0

        self.tensor.add_batch(timestamp, batch)

        self.left_prev = left_prev
        self.right_prev = right_prev
        self.ts_prev = timestamp
        self.iterator_count += 1
        return self.tensor.active_frame

    def feed(self, timestamp: float, stereo: tuple[np.ndarray, np.ndarray]) -> FeatureFrame:
        """Feed the next frame."""
        self.logger.debug(f"Feeding frame {self.iterator_count} in timestamp {timestamp:.0f}")
        if not self.tensor.initiated:
            return self.feed_first(timestamp, stereo)

        prev_feat_frame = self.tensor.active_frame
        self.tensor.step(timestamp)

        left_next, right_next = np.asarray(stereo[0]), np.asarray(stereo[1])
        next_batch = self._optical_flow_lk(left_next, prev_feat_frame)
        next_batch[:, FeatureSchema.TIMESTAMP] = timestamp

        good_feat_mask = next_batch[:, FeatureSchema.LIFECYCLE] != FeatureLifecycle.LOST.value
        if self.mode == FeatureTrackerMode.STEREO:
            good_new_feat = next_batch[good_feat_mask]
            good_new_feat = np.column_stack([good_new_feat[:, 0], good_new_feat[:, 2:4]])
            stereo_match = self._stereo_match_lk(left_next, right_next, good_new_feat)
            next_batch[good_feat_mask, FeatureSchema.RIGHT_U : FeatureSchema.RIGHT_V + 1] = stereo_match[
                :,
                StereoMatchSchema.RIGHT_U : StereoMatchSchema.RIGHT_V + 1,
            ]

            stereo_ok = stereo_match[:, StereoMatchSchema.STEREO_OK].astype(bool)
            prev_score = np.nan_to_num(next_batch[good_feat_mask, FeatureSchema.STEREO_SCORE], nan=0.0)
            next_batch[:, FeatureSchema.STEREO_SCORE] = 0.0
            next_batch[good_feat_mask, FeatureSchema.STEREO_SCORE] = np.where(stereo_ok, prev_score + 1.0, 0.0)

        u, v = next_batch[good_feat_mask, 2].astype(np.int32), next_batch[good_feat_mask, 3].astype(np.int32)
        region_counts = np.bincount(self.grid_mask[v, u], minlength=self.REGION_AMOUNT)
        hungry_regions = region_counts < self.FEAT_RETRACK_THRESHOLD
        if np.any(hungry_regions):
            self.logger.trace(f"{np.sum(hungry_regions)} regions to retrack")
            forbidden_mask = np.ones((self.IMAGE_SHAPE["h"], self.IMAGE_SHAPE["w"]), dtype=np.uint8)
            for i in range(len(u)):
                cv2.circle(forbidden_mask, (u[i], v[i]), 20, 0, -1)
            target_region_id = np.where(hungry_regions)[0]
            target_mask = np.isin(self.grid_mask, target_region_id).astype(np.uint8) * 255
            mask = forbidden_mask & target_mask
            new_keypoints = self.fast.detect(image=left_next, mask=mask)
            if len(new_keypoints) > 0:
                new_keypoints = self._select_retrack_kps(
                    new_keypoints,
                    region_counts=region_counts,
                    target_region_ids=target_region_id,
                    min_distance=RETRACK_MIN_DISTANCE_PX,
                )
                new_batch = self.initiate_new_features(left_next, right_next, new_keypoints, timestamp)
                next_batch = np.concatenate([next_batch, new_batch], axis=0)

        self.tensor.add_batch(timestamp, next_batch)
        self.left_prev = left_next.copy()
        self.right_prev = right_next.copy()
        self.ts_prev = timestamp
        self.iterator_count += 1
        self.hungry_regions = []
        return self.tensor.active_frame

    def initiate_new_features(
        self,
        left_next: np.ndarray,
        right_next: np.ndarray,
        new_keypoints: Sequence[cv2.KeyPoint],
        timestamp: float,
    ) -> np.ndarray:
        """Convert a sequence of keypoints to a batch."""
        if len(new_keypoints) == 0:
            return np.empty((0, FeatureSchema.count()), dtype=np.float32)

        kps = np.array([[kp.pt[0], kp.pt[1], kp.response] for kp in new_keypoints], dtype=np.float32)
        kps = kps[kps[:, 2].argsort()[::-1]]
        kps = np.column_stack(
            [
                np.arange(self.next_feat_id, self.next_feat_id + kps.shape[0]),
                kps[:, :2],
            ]
        )
        self.next_feat_id += kps.shape[0]
        new_batch = np.full((kps.shape[0], FeatureSchema.count()), np.nan, dtype=np.float32)
        new_batch[:, FeatureSchema.FEAT_ID] = kps[:, 0]
        new_batch[:, FeatureSchema.TIMESTAMP] = timestamp
        new_batch[:, FeatureSchema.LEFT_U : FeatureSchema.LEFT_V + 1] = kps[:, 1:3]
        if self.mode == FeatureTrackerMode.STEREO:
            stereo_match = self._stereo_match_lk(left_next, right_next, kps)
            new_batch[:, FeatureSchema.RIGHT_U : FeatureSchema.RIGHT_V + 1] = stereo_match[
                :,
                StereoMatchSchema.RIGHT_U : StereoMatchSchema.RIGHT_V + 1,
            ]
            new_batch[:, FeatureSchema.STEREO_SCORE] = 0.0
        new_batch[:, FeatureSchema.LIFECYCLE] = FeatureLifecycle.ACTIVE.value
        new_batch[:, FeatureSchema.AGE] = 0
        return new_batch

    def _points_in_bounds(self, u: NDArray[np.float32], v: NDArray[np.float32]) -> NDArray[np.bool_]:
        """Check if a points are in bounds."""
        w = self.IMAGE_SHAPE["w"]
        h = self.IMAGE_SHAPE["h"]
        m = self.SHIFT_MARGIN_DICT
        in_frame = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        in_grid = (u >= m["left"]) & (u < w - m["right"]) & (v >= m["top"]) & (v < h - m["bottom"])
        return in_frame & in_grid

    def _spawn_grid_mask(self) -> np.ndarray:
        """Spawn a grid mask."""
        w, h = self.IMAGE_SHAPE["w"], self.IMAGE_SHAPE["h"]
        mask = np.full((h, w), -1, dtype=np.int8)
        for region in self.grid:
            region_id = region.region_id
            mask[region.mask == 1] = region_id
        return mask

    def _select_retrack_kps(
        self,
        keypoints: Sequence[cv2.KeyPoint],
        region_counts: NDArray[np.int64],
        target_region_ids: NDArray[np.int64],
        min_distance: int,
    ) -> list[cv2.KeyPoint]:
        """Select retracking keypoints with per-region quotas after one FAST detect."""
        quotas = {
            int(region_id): max(self.FEAT_PER_REGION - int(region_counts[int(region_id)]), 0)
            for region_id in target_region_ids
        }
        total_quota = sum(quotas.values())
        if total_quota == 0:
            return []

        selected: list[cv2.KeyPoint] = []
        selected_by_region = dict.fromkeys(quotas, 0)
        min_dist2 = min_distance * min_distance
        height, width = self.grid_mask.shape

        for kp in sorted(keypoints, key=lambda x: x.response, reverse=True):
            x, y = kp.pt
            u, v = int(x), int(y)
            if not (0 <= u < width and 0 <= v < height):
                continue

            region_id = int(self.grid_mask[v, u])
            if region_id not in quotas:
                continue
            if selected_by_region[region_id] >= quotas[region_id]:
                continue

            too_close = False
            for selected_kp in selected:
                sx, sy = selected_kp.pt
                if (x - sx) ** 2 + (y - sy) ** 2 < min_dist2:
                    too_close = True
                    break
            if too_close:
                continue

            selected.append(kp)
            selected_by_region[region_id] += 1
            if len(selected) >= total_quota:
                break

        return selected

    @staticmethod
    def _select_spread_kps(
        keypoints: Sequence[cv2.KeyPoint], max_count: int, min_distance: int
    ) -> Sequence[cv2.KeyPoint]:
        """Select spread keypoints."""
        selected: list[cv2.KeyPoint] = []
        min_dist2 = min_distance * min_distance
        for kp in sorted(keypoints, key=lambda x: x.response, reverse=True):
            x, y = kp.pt
            too_close = False
            for selected_kp in selected:
                sx, sy = selected_kp.pt
                if (x - sx) ** 2 + (y - sy) ** 2 < min_dist2:
                    too_close = True
                    break

            if too_close:
                continue

            selected.append(kp)
            if len(selected) >= max_count:
                break
        return selected

    def active_frame(self) -> FeatureFrame:
        """Get the active frame."""
        return self.tensor.active_frame
