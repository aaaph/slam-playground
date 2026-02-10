from dataclasses import dataclass
from typing import Literal

import numpy as np

from core.feature_tracker.feature import Feature
from core.front_end.keyframe_selector import SelectReason
from core.transformations.special_euclidian_3_dim import SE3

type KeyframeKeys = Literal[
    "keyframe_id",
    "select_reason",
    "timestamp",
    "pose",
    "active_features",
    "active_landmarks",
    "active_features_count",
    "active_landmarks_count",
]


@dataclass
class Keyframe:
    """Front-End keyframe."""

    keyframe_id: int

    select_reason: SelectReason
    timestamp: float
    pose: SE3
    active_features: dict[int, Feature]
    active_landmarks: dict[int, np.ndarray]

    def __repr__(self) -> str:
        """Return a string with keyframe information."""
        size = len(self.active_features)
        return f"Keyframe({self.keyframe_id} at {self.timestamp:.0f}, reason: {self.select_reason.name}, s:{size})"

    def as_soa(self) -> dict[KeyframeKeys, np.ndarray]:
        """Convert the keyframe to a SOA(Struct of Arrays) dictionary."""
        landmark_ids = np.fromiter(self.active_landmarks.keys(), dtype=np.int32)
        landmark_count = len(landmark_ids)
        landmark_coords = np.array(list(self.active_landmarks.values()), dtype=np.float32)
        active_landmarks = np.column_stack([landmark_ids, landmark_coords])
        active_features_count = len(self.active_features)
        active_features = np.zeros((active_features_count, 6), dtype=np.float32)
        for i, (feat_id, feat) in enumerate(self.active_features.items()):
            active_features[i, 0] = feat_id
            stereo_pair = feat.get_active_stereo_pair()
            active_features[i, 1] = stereo_pair[1][0]
            active_features[i, 2] = stereo_pair[1][1]
            if stereo_pair[2] is not None:
                active_features[i, 3] = stereo_pair[2][0]
                active_features[i, 4] = stereo_pair[2][1]
            else:
                active_features[i, 3] = np.nan
                active_features[i, 4] = np.nan
            active_features[i, 5] = feat.state.value
        return {
            "keyframe_id": np.array([self.keyframe_id], dtype=np.int32),
            "select_reason": np.array([self.select_reason.value], dtype=np.int32),
            "timestamp": np.array([self.timestamp], dtype=np.float32),
            "pose": self.pose.as_matrix(),
            "active_landmarks": active_landmarks.astype(np.float32),
            "active_features": active_features.astype(np.float32),
            "active_features_count": np.array([active_features_count], dtype=np.int32),
            "active_landmarks_count": np.array([landmark_count], dtype=np.int32),
        }

    @classmethod
    def from_soa(cls, soa: dict[KeyframeKeys, np.ndarray]) -> "Keyframe":
        """Create a keyframe from a SOA dictionary."""
        keyframe_id = soa["keyframe_id"][0]
        select_reason = SelectReason(soa["select_reason"][0])
        timestamp = float(soa["timestamp"][0])
        pose = SE3.from_matrix(soa["pose"])
        active_landmarks = zip(soa["active_landmarks"][:, 0], soa["active_landmarks"][:, 1:], strict=False)
        # active_features rows are [feat_id, left_u, left_v, right_u, right_v, status]
        active_features = zip(soa["active_features"][:, 0], soa["active_features"][:, 1:], strict=False)
        return cls(
            keyframe_id=int(keyframe_id),
            select_reason=select_reason,
            timestamp=timestamp,
            pose=pose,
            active_features={
                int(feat_id): Feature.spawn_from_ndarray(
                    np.array([feat_id, timestamp, feat[0], feat[1], feat[2], feat[3], feat[4]])
                )
                for feat_id, feat in active_features
            },
            active_landmarks={int(landmark_id): landmark for landmark_id, landmark in active_landmarks},
        )
