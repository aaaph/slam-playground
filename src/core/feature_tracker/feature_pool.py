from collections.abc import Iterator

import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray

from core.feature_tracker.feature import Feature, FeatureStatus
from core.feature_tracker.feature_tensor import FeatureTensor
from logger import spawn_logger


class FeaturePool:
    """Feature pool."""

    logger = spawn_logger(app="feature_pool")

    def __init__(self, feat_id_counter: int = 0, capacity: int = 1000) -> None:
        """Initialize the feature pool."""
        self.features: dict[int, Feature] = {}
        self.feat_id_counter = feat_id_counter
        self.active_track: dict[int, tuple[float, float]] = {}
        self.capacity = capacity
        self.active_u = np.full(capacity, np.nan, np.float32)
        self.active_v = np.full(capacity, np.nan, np.float32)
        self.active_feat_id = np.full(capacity, -1, np.int32)
        self.tensor = FeatureTensor.from_capacity(capacity)

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
        ts, active_left, active_right = feature.get_active_stereo_pair()
        u, v = active_left
        if ts > self.active_timestamp:
            self.active_timestamp = ts
            self.active_track.clear()
        self.active_track[feature.feat_id] = (u, v)
        self.logger.debug(f"Added feature: {feature.feat_id}")
        self.tensor.add(feature.feat_id, ts, active_left, active_right, feature.state)

    def get_active_points_ready_for_klt(self) -> NDArray[np.float32]:
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
        self.tensor.add(feat_id, timestamp, left_uv, right_uv, feature.state)

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
        self.tensor.add(feat_id, timestamp, (u, v), None, feature.state)

    def remove_features(self, p0: MatLike) -> None:
        """Remove features from the feature pool."""
        for point in p0:
            feat_id, _x, _y = point.ravel()
            feat_id = int(feat_id)
            feature = self.features[feat_id]
            del self.features[feat_id]
            is_active = self.active_track.get(feat_id, None) is not None
            if is_active:
                del self.active_track[feat_id]
            self.logger.debug(f"Removed feature: {feat_id}, iterations: {feature.iteration_life}")

    def mark_features_as_lost(self, timestamp: float, p0: MatLike) -> None:
        """Mark features as lost."""
        for point in p0:
            feat_id, _x, _y = point.ravel()
            feat_id = int(feat_id)
            feature = self.features[feat_id]
            feature.state = FeatureStatus.LOST
            self.active_track.pop(feat_id, None)
            self.logger.debug(f"Marked feature as lost: {feat_id}")
            self.tensor.add(feat_id, timestamp, (-1.0, -1.0), (-1.0, -1.0), FeatureStatus.LOST)

    def clear_lost_features(self) -> None:
        """Clear lost features."""
        feat_dict = {}
        for feat_id, feature in self.features.items():
            if feature.state in [FeatureStatus.LOST]:
                feat_dict[feat_id] = feature.state
        for feat_id, state in feat_dict.items():
            self.active_track.pop(feat_id, None)
            self.features.pop(feat_id, None)
            self.logger.debug(f"Cleared {state} feature: {feat_id}")

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
