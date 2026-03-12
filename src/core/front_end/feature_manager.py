from collections import OrderedDict
from enum import IntEnum

import numpy as np

from core.camera_model.stereo_camera_ctx import StereoContext
from core.feature_tracker.feature_frame import FeatureFrame
from core.feature_tracker.feature_schema import FeatureSchema
from core.pose_tracker.feature_triangulation import FeatureTriangulation


class LocalMapStatus(IntEnum):
    """Local map status."""

    CANDIDATE = 0
    DISPARITY_TRIANGULATED = 1
    RAY_TRIANGULATED = 2


class LocalMapSchema(IntEnum):
    """Local map schema."""

    FEAT_ID = 0
    X = 1
    Y = 2
    Z = 3


class FeatureManager:
    """Feature manager."""

    def __init__(self, triangulator: FeatureTriangulation, capacity: int = 1000) -> None:
        """Initialize the local map."""
        self.lru = OrderedDict()
        self.capacity = capacity
        self.triangulator = triangulator
        self.tensor = np.full((capacity, 5), np.nan, dtype=np.float32)

    def add_active_features(self, feature_frame: FeatureFrame) -> np.ndarray:
        """Add the active features to the local map."""
        mask_exists = np.isin(feature_frame.good_features()[:, 0], self.tensor[:, 0])
        new_features = feature_frame.good_features()[~mask_exists]

        candidates = np.full((new_features.shape[0], 5), np.nan, dtype=np.float32)
        candidates[:, 0] = new_features[:, FeatureSchema.FEAT_ID]
        candidates[:, 1] = new_features[:, FeatureSchema.LEFT_U]
        candidates[:, 2] = new_features[:, FeatureSchema.LEFT_V]
        candidates[:, 3] = new_features[:, FeatureSchema.RIGHT_U]
        candidates[:, 4] = new_features[:, FeatureSchema.RIGHT_V]

        stereo_3d_in_camera_frame = self.triangulator.make_initial_guess_by_stereo_batch(candidates)
        good_stereo_mask = stereo_3d_in_camera_frame[:, 4].astype(bool)
        _bad_stereo_mask = ~good_stereo_mask

        return stereo_3d_in_camera_frame[good_stereo_mask]
        # print(stereo_3d_in_camera_frame[good_stereo_mask])

    @classmethod
    def from_stereo_camera_ctx(cls, stereo_ctx: StereoContext, capacity: int = 1000) -> "FeatureManager":
        """Create a feature triangulation module from a StereoCameraCtx."""
        return cls(FeatureTriangulation.from_stereo_camera_ctx(stereo_ctx), capacity)
