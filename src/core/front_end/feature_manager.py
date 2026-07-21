from collections import OrderedDict
from enum import IntEnum

import cv2
import numpy as np
from numpy.typing import NDArray

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

    @classmethod
    def count(cls) -> int:
        """Count the number of schema."""
        return len(cls.__members__)


class TriangulatedTrackSchema(IntEnum):
    """Aligned triangulated track schema."""

    FEAT_ID = 0
    X = 1
    Y = 2
    Z = 3
    STATUS = 4


class FeatureManager:
    """Feature manager."""

    def __init__(self, triangulator: FeatureTriangulation, capacity: int = 1000) -> None:
        """Initialize the local map."""
        self.lru = OrderedDict()
        self.capacity = capacity
        self.triangulator = triangulator
        self.patch_size = 31
        self.orb_detector = cv2.ORB.create(
            patchSize=self.patch_size,
            edgeThreshold=15,
            fastThreshold=15,
            WTA_K=2,
            scaleFactor=1.2,
            nlevels=8,
            nfeatures=1000,
        )

    def get_orb_descriptors(
        self, active_track: NDArray[np.float32], image: NDArray[np.uint8]
    ) -> NDArray[np.uint8]:
        """Get the ORB descriptors for the active track."""
        feature_ids = active_track[:, FeatureSchema.FEAT_ID].astype(np.int32, copy=False)
        left_points = active_track[:, FeatureSchema.LEFT_U : FeatureSchema.LEFT_V + 1].astype(
            np.float32, copy=False
        )
        keypoints = [
            cv2.KeyPoint(float(u), float(v), float(self.patch_size), -1, 0, 0, int(feature_id))
            for (u, v), feature_id in zip(left_points, feature_ids, strict=True)
        ]
        _, descriptors = self.orb_detector.compute(image, keypoints)

        if descriptors is None or descriptors.size == 0:
            raise ValueError("No descriptors found")
        return np.ascontiguousarray(descriptors, dtype=np.uint8)

    def triangulate_active_track(self, active_track: NDArray[np.float32]) -> NDArray[np.float32]:
        """Triangulate the active track and preserve input row order."""
        if active_track.size == 0:
            return np.empty((0, 5), dtype=np.float32)

        candidates = np.full((active_track.shape[0], 5), np.nan, dtype=np.float32)
        candidates[:, 0] = active_track[:, FeatureSchema.FEAT_ID]
        candidates[:, 1] = active_track[:, FeatureSchema.LEFT_U]
        candidates[:, 2] = active_track[:, FeatureSchema.LEFT_V]
        candidates[:, 3] = active_track[:, FeatureSchema.RIGHT_U]
        candidates[:, 4] = active_track[:, FeatureSchema.RIGHT_V]

        return self.triangulator.make_initial_guess_by_stereo_batch(candidates)

    def triangulate_frame(self, frame: FeatureFrame) -> NDArray[np.float32]:
        """Triangulate the frame."""
        return self.triangulate_active_track(frame.good_features())

    def merge_active_track_and_points(self, active_track: NDArray[np.float32]) -> NDArray[np.float32]:
        """Merge the active track with aligned XYZ estimates."""
        aligned_points = self.triangulate_active_track(active_track)
        if aligned_points.size == 0:
            return np.empty((0, active_track.shape[1] + 3), dtype=np.float32)
        return np.column_stack(
            (active_track, aligned_points[:, TriangulatedTrackSchema.X : TriangulatedTrackSchema.Z + 1])
        )

    def add_active_features(self, feature_frame: FeatureFrame) -> np.ndarray:
        """Return only valid triangulated points for visualization."""
        aligned_points = self.triangulate_active_track(feature_frame.good_features())
        good_stereo_mask = aligned_points[:, TriangulatedTrackSchema.STATUS].astype(bool)
        return aligned_points[good_stereo_mask]

    @classmethod
    def from_stereo_camera_ctx(cls, stereo_ctx: StereoContext, capacity: int = 1000) -> "FeatureManager":
        """Create a feature triangulation module from a StereoCameraCtx."""
        return cls(FeatureTriangulation.from_stereo_camera_ctx(stereo_ctx), capacity)
