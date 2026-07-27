from dataclasses import dataclass
from enum import IntEnum

import cv2
import numpy as np

from core.feature_tracker.feature_schema import FeatureLifecycle, FeatureSchema
from core.feature_tracker.feature_tensor import FeatureTensor


class VPRGeometrySchema(IntEnum):
    """Visual Place Recognition geometry schema."""

    LEFT_U = 0
    LEFT_V = 1
    RIGHT_U = 2
    RIGHT_V = 3
    BEARING_X = 4
    BEARING_Y = 5
    BEARING_Z = 6
    POINT_X = 7
    POINT_Y = 8
    POINT_Z = 9

    @classmethod
    def count(cls) -> int:
        """Get the count of the VPR geometry schema."""
        return len(cls)


@dataclass(frozen=True, slots=True)
class VPRDetection:
    """Visual Place Recognition detection payload."""

    geometry: np.ndarray
    descriptors: np.ndarray

    @property
    def keypoints(self) -> list[cv2.KeyPoint]:
        """Get the keypoints."""
        return [
            cv2.KeyPoint(x, y, 1.0)
            for x, y in self.geometry[:, VPRGeometrySchema.LEFT_U : VPRGeometrySchema.LEFT_V + 1]
        ]


@dataclass(frozen=True, slots=True)
class VPRFrame:
    """Visual Place Recognition frame."""

    frame_id: int
    kf_id: int
    timestamp: float
    geometry: np.ndarray
    descriptors: np.ndarray

    @classmethod
    def empty(cls) -> "VPRFrame":
        """Create an empty VPR frame."""
        return cls(
            frame_id=0,
            kf_id=0,
            timestamp=0.0,
            geometry=np.empty((0, 10), dtype=np.float32),
            descriptors=np.empty((0, 128), dtype=np.float32),
        )

    @property
    def left_uv(self) -> np.ndarray:
        """Get the left UV coordinates."""
        return self.geometry[:, VPRGeometrySchema.LEFT_U : VPRGeometrySchema.LEFT_V + 1]

    @property
    def right_uv(self) -> np.ndarray:
        """Get the right UV coordinates."""
        return self.geometry[:, VPRGeometrySchema.RIGHT_U : VPRGeometrySchema.RIGHT_V + 1]

    @property
    def bearings(self) -> np.ndarray:
        """Get the bearings."""
        return self.geometry[:, VPRGeometrySchema.BEARING_X : VPRGeometrySchema.BEARING_Z + 1]

    @property
    def points_xyz(self) -> np.ndarray:
        """Get the stereo-triangulated 3D points."""
        return self.geometry[:, VPRGeometrySchema.POINT_X : VPRGeometrySchema.POINT_Z + 1]

    @property
    def pointcloud(self) -> np.ndarray:
        """Get the stereo-triangulated 3D points as a pointcloud."""
        pointcloud = np.zeros((self.points_xyz.shape[0], 5), dtype=np.float32)
        pointcloud[:, 0] = np.arange(self.points_xyz.shape[0], dtype=np.float32)
        pointcloud[:, 1:4] = self.points_xyz.astype(np.float32)
        return pointcloud

    @property
    def pointcloud_size(self) -> int:
        """Get the size of the pointcloud."""
        return self.points_xyz.shape[0]

    @property
    def active_feat_tensor(self) -> FeatureTensor:
        """Get the active features as a feature tensor."""
        tensor = FeatureTensor.default_factory(capacity=self.geometry.shape[0])
        batch = np.zeros((self.geometry.shape[0], FeatureSchema.count()), dtype=np.float32)
        batch[:, FeatureSchema.FEAT_ID] = np.arange(self.geometry.shape[0])
        batch[:, FeatureSchema.TIMESTAMP] = self.timestamp
        batch[:, FeatureSchema.LEFT_U] = self.left_uv[:, 0]
        batch[:, FeatureSchema.LEFT_V] = self.left_uv[:, 1]
        batch[:, FeatureSchema.LIFECYCLE] = FeatureLifecycle.ACTIVE.value
        tensor.add_batch(self.timestamp, batch)
        return tensor

    @classmethod
    def from_detection(cls, frame_id: int, kf_id: int, timestamp: float, detection: VPRDetection) -> "VPRFrame":
        """Create a VPR frame from a detection."""
        return VPRFrame(
            frame_id=frame_id,
            kf_id=kf_id,
            timestamp=timestamp,
            geometry=detection.geometry,
            descriptors=detection.descriptors,
        )

    def __repr__(self) -> str:
        """Return a string representation of the VPR frame."""
        return (
            f"VPRFrame(frame_id={self.frame_id}, kf_id={self.kf_id}, timestamp={self.timestamp:.0f}ns, "
            f"geometry_shape={self.geometry.shape}, descriptors_shape={self.descriptors.shape})"
        )
