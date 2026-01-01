from collections.abc import Iterator
from typing import Literal, NamedTuple

import cv2
import numpy as np

from core.camera_model.stereo_camera_ctx import StereoContext
from core.feature_tracker.feature import Feature
from core.feature_tracker.feature_pool import FeaturePool
from core.feature_tracker.feature_tracker_region import FeatureTrackerRegion
from core.feature_tracker.helper import grid_factor
from core.feature_tracker.my_collections import ResettableDict
from logger import spawn_logger


class FeatureTrackerConfig(NamedTuple):
    """Feature tracker configuration."""

    shift_margin: tuple[int, int, int, int] = (16, 16, 16, 16)
    region_amount: int = 8
    optical_flow_klt_win_size: tuple[int, int] = (8, 8)
    stereo_klt_win_size: tuple[int, int] = (21, 21)
    feat_amount_per_region: int = 25
    feat_retrack_threshold: int = 20
    image_shape: tuple[int, int] = (752, 480)


class FeatureTracker:
    """Feature tracker (Stereo Implementation)."""

    def __init__(
        self,
        stereo_k: np.ndarray,
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

        default_feat_in_region = {-1: set()}
        for region in self.grid:
            default_feat_in_region[region.region_id] = set()
        self.feat_in_region = ResettableDict(default_feat_in_region)

        self.stereo_k = stereo_k
        self.fast = cv2.FastFeatureDetector.create()

        self.pool: FeaturePool = None
        self.left_prev: np.ndarray = None
        self.right_prev: np.ndarray = None
        self.hungry_regions: list[FeatureTrackerRegion] = []
        self.ts_prev = -1.0
        self.iterator_count = 0

    @classmethod
    def default_factory(
        cls, stereo_ctx: StereoContext, feat_amount_per_region: int = 25, feat_retrack_threshold: int = 10
    ) -> "FeatureTracker":
        """Create a default feature tracker."""
        return cls(
            stereo_ctx.stereo_k,
            FeatureTrackerConfig(
                feat_amount_per_region=feat_amount_per_region, feat_retrack_threshold=feat_retrack_threshold
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
                # Apply the global shift mask so that regions respect the configured margins
                mask[shift_mask == 0] = 0
                region = FeatureTrackerRegion(index, mask)
                region_masks.append(region)
                index += 1
        # apply shift margin

        return region_masks

    def _lk_match_right_to_left(
        self, left: np.ndarray, right: np.ndarray, points_left: list[tuple[float, float]]
    ) -> dict[tuple[float, float], tuple[float, float]]:
        """LK matching right to left."""
        if len(points_left) == 0:
            return {}
        p0 = np.array(points_left, dtype=np.float32).reshape(-1, 2)
        points_right, st_left_right, _err_left_right = cv2.calcOpticalFlowPyrLK(
            left, right, p0, None, **self.stereo_optical_flow_klt_params
        )
        points_back, st_right_left, _err_right_left = cv2.calcOpticalFlowPyrLK(
            right, left, points_right, None, **self.stereo_optical_flow_klt_params
        )

        forward_back_err = np.linalg.norm(p0 - points_back, axis=1).ravel()
        forward_back_mask = forward_back_err < 1.0
        ul, vl = p0[:, 0], p0[:, 1]
        ur, vr = points_right[:, 0], points_right[:, 1]
        disp = ul - ur

        max_disparity = 64
        min_disparity = 0.5
        epipolar_mask = np.abs(vl - vr) < 1.0
        disparity_mask = (disp > min_disparity) & (disp < max_disparity)

        mask = (
            (st_left_right.ravel() == 1)
            & (st_right_left.ravel() == 1)
            & forward_back_mask
            & epipolar_mask
            & disparity_mask
        )

        right_to_left_dict = {}
        left_to_right_dict = {}
        for i, lp in enumerate(points_left):
            x, y = lp.ravel()
            lkey = (x, y)
            rkey = (points_right[i, 0], points_right[i, 1])
            if mask[i]:
                right_to_left_dict[rkey] = lkey
                left_to_right_dict[lkey] = rkey
            else:
                left_to_right_dict[lkey] = None

        return right_to_left_dict

    def _lk_match_left_to_right(
        self, left: np.ndarray, right: np.ndarray, points_left: np.ndarray
    ) -> dict[tuple[int, float, float], tuple[int, float, float]]:
        """LK matching left to right."""
        if len(points_left) == 0:
            return {}
        p0 = points_left[:, 1:].astype(np.float32)
        points_right, st_left_right, _err_left_right = cv2.calcOpticalFlowPyrLK(
            left, right, p0, None, **self.stereo_optical_flow_klt_params
        )
        points_back, st_right_left, _err_right_left = cv2.calcOpticalFlowPyrLK(
            right, left, points_right, None, **self.stereo_optical_flow_klt_params
        )

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

        left_to_right_dict = {}
        for i, lp in enumerate(points_left):
            feat_id, x, y = lp.ravel()
            lkey = (feat_id, x, y)
            rkey = (feat_id, points_right[i, 0], points_right[i, 1])
            if mask[i]:
                left_to_right_dict[lkey] = rkey
            else:
                left_to_right_dict[lkey] = None

        return left_to_right_dict

    def _optical_flow_lk(self, left_next: np.ndarray, active_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p0 = active_points[:, 1:].astype(np.float32)
        p_next = np.array(p0, dtype=np.int32)  # zero motion prediction
        p1, st, _err = cv2.calcOpticalFlowPyrLK(
            self.left_prev, left_next, p0, p_next, **self.optical_flow_klt_params
        )
        st = st.ravel().astype(bool)

        good_old = active_points[st]
        good_old_no_id = good_old[:, 1:]
        good_new = p1[st]
        bad_old = active_points[~st]

        _E, inliners = cv2.findEssentialMat(  # noqa: N806
            good_new,
            good_old_no_id,
            cameraMatrix=self.stereo_k,
            method=cv2.RANSAC,
            threshold=0.999,
        )
        inliner_mask = inliners.ravel().astype(bool)
        bad_old = np.concatenate([bad_old, good_old[~inliner_mask]])
        good_new = good_new[inliner_mask]
        good_old = good_old[inliner_mask]

        good_new_id = []
        for _i, (new, old) in enumerate(zip(good_new, good_old, strict=True)):
            a, b = new.ravel()
            feat_id, _c, _d = old.ravel()
            feat_id = int(feat_id)
            good_new_id.append((feat_id, a, b))

        good_not_out_of_bounds = []
        good_out_of_bounds = []
        for good in good_new_id:
            x, y = good[1], good[2]
            if x < self.IMAGE_SHAPE["w"] and x > 0 and y > 0 and y < self.IMAGE_SHAPE["h"]:
                good_not_out_of_bounds.append(good)
            else:
                good_out_of_bounds.append(good)

        good_new_id = np.array(good_not_out_of_bounds, dtype=np.float32).reshape(-1, 3)
        good_out_of_bounds = np.array(good_out_of_bounds, dtype=np.float32).reshape(-1, 3)

        bad_old = np.concatenate([bad_old, good_out_of_bounds])
        return good_new_id, bad_old

    def iterate_through_features(
        self, states: None | list[Literal["new", "tracked", "lost", "stable", "unstable"]] = None
    ) -> Iterator[Feature]:
        """Iterate through the feature pool."""
        if self.pool is None:
            raise ValueError("Feature pool is not initialized")
        if states is None:
            states: list[Literal["new", "tracked", "lost", "stable", "unstable"]] = []
            states.extend(["new", "tracked", "lost", "stable", "unstable"])
        for feat in self.pool.iterate_features():
            if feat.state in states:
                yield feat

    def feat_count(self) -> int:
        """Get the number of features."""
        return len(self.pool.features)

    def feed_first(self, timestamp: float, stereo: tuple[np.ndarray, np.ndarray]) -> dict[int, Feature]:
        """Feed the first frame."""
        left_prev, right_prev = np.array(stereo[0]), np.array(stereo[1])
        # left_prev, right_prev = self.preprocessor.preprocess_stereo(left_prev, right_prev)
        # left_prev, right_prev = np.array(left_prev), np.array(right_prev)

        keypoints: list[cv2.KeyPoint] = []
        for region in self.grid:
            kps = self.fast.detect(image=left_prev, mask=np.array(region.mask))
            kps = sorted(kps, key=lambda x: x.response, reverse=True)
            kps = kps[: self.FEAT_PER_REGION]
            keypoints.extend(kps)

        keypoints = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
        keypoints = np.array(keypoints, dtype=np.float32).reshape(-1, 2)

        right_to_left_map = self._lk_match_right_to_left(left_prev, right_prev, keypoints)
        self.pool = FeaturePool.spawn_from_stereo_map(timestamp, right_to_left_map)
        self.left_prev = left_prev
        self.right_prev = right_prev
        self.ts_prev = timestamp
        self.iterator_count += 1
        return self.active_features_dict()

    def feed(self, timestamp: float, stereo: tuple[np.ndarray, np.ndarray]) -> dict[int, Feature]:
        """Feed the next frame."""
        self.logger.debug(f"Feeding frame {self.iterator_count} in timestamp {timestamp:.0f}")
        if self.pool is None:
            return self.feed_first(timestamp, stereo)
        self.feat_in_region.clear()
        self.pool.clear_lost_features()

        left_next, right_next = np.asarray(stereo[0]), np.asarray(stereo[1])

        active_points = self.pool.get_active_points_ready_for_klt()
        if len(active_points) > 0:
            good_new_id, bad_old = self._optical_flow_lk(left_next, active_points)
            self.pool.mark_features_as_lost(bad_old)
        else:
            good_new_id = np.array([], dtype=np.float32).reshape(-1, 3)
            bad_old = np.array([], dtype=np.float32).reshape(-1, 3)

        left_to_right_map = self._lk_match_left_to_right(left_next, right_next, good_new_id)
        for left_point, right_point in left_to_right_map.items():
            feat_id, lx, ly = left_point
            if right_point is None:
                self.pool.apply_left_point(timestamp, feat_id, lx, ly)
            else:
                feat_id, rx, ry = right_point
                self.pool.apply_stereo_pair(timestamp, feat_id, (lx, ly), (rx, ry))
                """ right_in_bounds = self._point_in_bounds(rx, ry)
                if right_in_bounds:
                    self.pool.apply_stereo_pair(timestamp, feat_id, (lx, ly), (rx, ry))
                else:
                    self.pool.apply_left_point(timestamp, feat_id, lx, ly) """
            region_id = self.grid_mask[int(ly), int(lx)]
            self.feat_in_region[region_id].add((feat_id, lx, ly))

        hungry_regions: list[FeatureTrackerRegion] = []
        new_keypoints: list[cv2.KeyPoint] = []
        for region in self.grid:
            feats_in_region = self.feat_in_region[region.region_id]
            how_many_feat_in_region = len(feats_in_region)
            if how_many_feat_in_region < self.FEAT_RETRACK_THRESHOLD:
                hungry_regions.append(region)
                region_mask = np.array(region.mask.copy())

                mask_arount_features = (
                    np.ones((self.IMAGE_SHAPE["h"], self.IMAGE_SHAPE["w"]), dtype=np.uint8) * 255
                )
                for _, lx, ly in feats_in_region:
                    x, y = int(lx), int(ly)
                    cv2.circle(mask_arount_features, (x, y), 15, 0, -1)

                mask = region_mask & mask_arount_features
                p2 = self.fast.detect(image=left_next, mask=mask)
                p2 = sorted(p2, key=lambda x: x.response, reverse=True)
                p2 = p2[: self.FEAT_RETRACK_THRESHOLD]
                new_keypoints.extend(p2)

        new_keypoints = [(kp.pt[0], kp.pt[1]) for kp in new_keypoints]
        new_keypoints = np.array(new_keypoints, dtype=np.float32).reshape(-1, 2)
        right_to_left_map = self._lk_match_right_to_left(left_next, right_next, new_keypoints)
        self.pool.apply_new_stereo_pair_batch(timestamp, right_to_left_map)

        self.left_prev = left_next.copy()
        self.right_prev = right_next.copy()
        self.ts_prev = timestamp
        self.iterator_count += 1
        self.hungry_regions = hungry_regions

        return self.active_features_dict()

    def _point_in_bounds(self, u: float, v: float) -> bool:
        """Check if a point is in bounds."""
        out_of_frame = False
        if u < 0 or u >= self.IMAGE_SHAPE["w"] or v < 0 or v >= self.IMAGE_SHAPE["h"]:
            out_of_frame = True
        out_of_grid = False
        if (
            u < self.SHIFT_MARGIN_DICT["left"]
            or u >= self.IMAGE_SHAPE["w"] - self.SHIFT_MARGIN_DICT["right"]
            or v < self.SHIFT_MARGIN_DICT["top"]
            or v >= self.IMAGE_SHAPE["h"] - self.SHIFT_MARGIN_DICT["bottom"]
        ):
            out_of_grid = True
        return not out_of_frame and not out_of_grid

    def _spawn_grid_mask(self) -> np.ndarray:
        """Spawn a grid mask."""
        w, h = self.IMAGE_SHAPE["w"], self.IMAGE_SHAPE["h"]
        mask = np.full((h, w), -1, dtype=np.int8)
        for region in self.grid:
            region_id = region.region_id
            mask[region.mask == 1] = region_id
        return mask

    def get_features_spawned_in_timestamp(self, timestamp: float) -> list[Feature]:
        """Get the features spawned in a timestamp."""
        return [feat for feat in self.pool.features.values() if feat.spawned_timestamp == timestamp]

    def get_oldest_timestamp(self) -> float:
        """Get the oldest timestamp."""
        oldest_ts = float("inf")
        for feat in self.iterate_through_features():
            oldest_ts = min(oldest_ts, feat.spawned_timestamp)
        return oldest_ts

    def get_feature_by_id(self, feat_id: int) -> Feature:
        """Get a feature by its ID."""
        feat = self.pool.features.get(feat_id)
        if feat is None:
            msg = f"Feature with ID {feat_id} not found"
            raise ValueError(msg)
        return feat

    def drop_features(self, features: list[Feature]) -> None:
        """Drop features."""
        p0 = np.array([(feat.feat_id, feat.u[0], feat.v[0]) for feat in features], dtype=np.float32).reshape(-1, 3)
        self.pool.remove_features(p0)

    def get_features_grouped_by_status(
        self,
    ) -> dict[Literal["new", "tracked", "lost", "unstable", "stable"], list[Feature]]:
        """Get the features grouped by status."""
        new_features = []
        tracked_features = []
        lost_features = []
        for feat in self.iterate_through_features():
            if feat.state == "new":
                new_features.append(feat)
            elif feat.state == "tracked":
                tracked_features.append(feat)
            elif feat.state == "lost":
                lost_features.append(feat)
        return {"new": new_features, "tracked": tracked_features, "lost": lost_features}

    def get_active_features_colors(self) -> dict[int, tuple[int, int, int]]:
        """Get the colors of the active features."""
        active_features_colors: dict[int, tuple[int, int, int]] = {}
        for feat in self.iterate_through_features():
            feat_id = feat.feat_id
            color = feat.feature_color()
            active_features_colors[feat_id] = color
        return active_features_colors

    def active_features_dict(
        self, states: None | list[Literal["new", "tracked", "lost", "stable", "unstable"]] = None
    ) -> dict[int, Feature]:
        """Get the dictionary of active features."""
        active_features_dict: dict[int, Feature] = {}
        for feat in self.iterate_through_features(states=states):
            active_features_dict[feat.feat_id] = feat
        return active_features_dict

    def active_features_ids(
        self, states: None | list[Literal["new", "tracked", "lost", "stable", "unstable"]] = None
    ) -> set[int]:
        """Get the IDs of the active features."""
        active_features = self.iterate_through_features(states=states)
        return {feat.feat_id for feat in active_features}

    def active_features_list(
        self, states: None | list[Literal["new", "tracked", "lost", "stable", "unstable"]] = None
    ) -> list[Feature]:
        """Get the list of active features."""
        active_features = self.iterate_through_features(states=states)
        return list(active_features)
