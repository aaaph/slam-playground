from collections.abc import Iterator

import numpy as np
from cv2.typing import MatLike

from core.feature_tracker.feature import Feature
from logger import spawn_logger


class FeaturePool:
    """Feature pool."""

    logger = spawn_logger(app="feature_pool")

    def __init__(self, feat_id_counter: int = 0) -> None:
        """Initialize the feature pool."""
        self.features: dict[int, Feature] = {}
        self.feat_id_counter = feat_id_counter
        self.active_track: dict[int, tuple[float, float]] = {}
        self.active_timestamp = -1.0

    def iterate_features(self) -> Iterator[Feature]:
        """Iterate over the features in the feature pool."""
        return iter(self.features.values())

    def add_feature(self, feature: Feature) -> None:
        """Add a feature to the feature pool."""
        if feature.obs_count() == 0:
            msg = f"Feature has no observations, feat_id: {feature.feat_id}"
            raise ValueError(msg)
        self.features[feature.feat_id] = feature
        self.feat_id_counter += 1
        ts, active_left, _ = feature.get_active_stereo_pair()
        u, v = active_left
        if ts > self.active_timestamp:
            self.active_timestamp = ts
            self.active_track.clear()
        self.active_track[feature.feat_id] = (u, v)

    def get_active_points_ready_for_klt(self) -> MatLike:
        """Get the active points ready for KLT."""
        points: list[tuple[int, float, float]] = []
        for feat_id, (u, v) in self.active_track.items():
            points.append((feat_id, u, v))
        return np.array(points, dtype=np.float32).reshape(-1, 3)

    def apply_stereo_pair(
        self, timestamp: float, feat_id: int, left_uv: tuple[float, float], right_uv: tuple[float, float]
    ) -> None:
        """Apply a stereo pair to the feature pool."""
        feature = self.features[feat_id]
        feature.apply_stereo_pair(timestamp, left_uv, right_uv)
        if timestamp > self.active_timestamp:
            self.active_timestamp = timestamp
            self.active_track.clear()
        self.active_track[feat_id] = left_uv

    def apply_new_stereo_pair(
        self, timestamp: float, left_uv: tuple[float, float], right_uv: tuple[float, float]
    ) -> None:
        """Apply a new stereo pair to the feature pool."""
        feat_id = self.feat_id_counter
        feature = Feature.spawn_from_left_and_right(feat_id, timestamp, left_uv, right_uv)
        self.add_feature(feature)

    def apply_new_stereo_pair_batch(
        self, timestamp: float, right_to_left_map: dict[tuple[float, float], tuple[float, float]]
    ) -> None:
        """Apply a new stereo pair batch to the feature pool."""
        for right_point, left_point in right_to_left_map.items():
            self.apply_new_stereo_pair(timestamp, left_point, right_point)

    def apply_left_point(self, timestamp: float, feat_id: int, u: float, v: float) -> None:
        """Apply a new point to the feature pool."""
        feature = self.features[feat_id]
        feature.apply_left_only(timestamp, (u, v))
        if timestamp > self.active_timestamp:
            self.active_timestamp = timestamp
            self.active_track.clear()
        self.active_track[feat_id] = (u, v)

    def remove_features(self, p0: MatLike) -> None:
        """Remove features from the feature pool."""
        for point in p0:
            feat_id, _x, _y = point.ravel()
            feat_id = int(feat_id)
            feature = self.features[feat_id]
            del self.features[feat_id]
            del self.active_track[feat_id]
            self.logger.debug(f"Removed feature: {feat_id}, iterations: {feature.iteration_life}")
            self.feat_id_counter -= 1

    @staticmethod
    def spawn_from_stereo_map(
        timestamp: float,
        right_to_left_map: dict[tuple[float, float], tuple[float, float]],
    ) -> "FeaturePool":
        """Spawn a feature pool from a stereo map."""
        fp = FeaturePool()
        for right_point, left_point in right_to_left_map.items():
            right_x, right_y = right_point
            left_x, left_y = left_point
            feat_id = fp.feat_id_counter
            feature = Feature.spawn_from_left_and_right(feat_id, timestamp, (left_x, left_y), (right_x, right_y))
            fp.add_feature(feature)
            fp.active_track[feat_id] = (left_x, left_y)
        return fp
