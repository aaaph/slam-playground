from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from core.feature_tracker.feature import Feature
from core.front_end.keyframe import Keyframe
from core.transformations.special_euclidian_3_dim import SE3

type SoAKeys = Literal[
    "active_feat",
    "active_feat_count",
    "new_landmarks",
    "new_landmarks_count",
]


@dataclass
class FrontendResult:
    """
    The result of the frame processing by the SLAM frontend.

    Contains all data to send to the optimizer and visualizer.
    """

    result_id: int
    timestamp: float

    camera_in_world_se3: SE3
    new_landmarks: dict[int, np.ndarray]
    active_features: dict[int, Feature]
    lost_features: dict[int, Feature]
    left: np.ndarray
    right: np.ndarray

    keyframe: Keyframe | None = None

    def as_soa(
        self,
    ) -> dict[SoAKeys, np.ndarray]:
        """Convert the frontend result to a SOA(Struct of Arrays) dictionary."""
        active_feat_ids = []
        active_feat_u_left = []
        active_feat_v_left = []
        active_feat_u_right = []
        active_feat_v_right = []
        active_feat_status = []

        new_landmarks_count = len(self.new_landmarks)
        new_landmarks_ids = np.array(list(self.new_landmarks.keys()), dtype=np.int32)
        new_landmarks_pts = np.array(list(self.new_landmarks.values()), dtype=np.float32)
        new_landmarks = np.column_stack((new_landmarks_ids, new_landmarks_pts)).astype(np.float32)

        for feat_id, feat in self.active_features.items():
            active_feat_ids.append(feat_id)
            _, left_uv, right_uv = feat.get_active_stereo_pair()

            active_feat_u_left.append(left_uv[0])
            active_feat_v_left.append(left_uv[1])
            if right_uv is not None:
                active_feat_u_right.append(right_uv[0])
                active_feat_v_right.append(right_uv[1])
            else:
                active_feat_u_right.append(np.nan)
                active_feat_v_right.append(np.nan)
            active_feat_status.append(feat.state.value)
        active_feat_count = len(active_feat_ids)
        active_feat = np.zeros((active_feat_count, 6), dtype=np.float32)
        active_feat[:, 0] = np.array(active_feat_ids, dtype=np.int32)
        active_feat[:, 1] = np.array(active_feat_u_left, dtype=np.float32)
        active_feat[:, 2] = np.array(active_feat_v_left, dtype=np.float32)
        active_feat[:, 3] = np.array(active_feat_u_right, dtype=np.float32)
        active_feat[:, 4] = np.array(active_feat_v_right, dtype=np.float32)
        active_feat[:, 5] = np.array(active_feat_status, dtype=np.int32)

        return {
            "active_feat": active_feat.astype(np.float32),
            "active_feat_count": np.array([active_feat_count], dtype=np.int32),
            "new_landmarks": new_landmarks.astype(np.float32),
            "new_landmarks_count": np.array([new_landmarks_count], dtype=np.int32),
        }

    def lost_features_ndarrays(self) -> tuple[int, NDArray[np.float32]]:
        """Convert the lost features to a NDArray."""
        length = len(self.lost_features)
        lost_features_coords = np.zeros((length, 6), dtype=np.float32)
        for i, (feat_id, feat) in enumerate(self.lost_features.items()):
            stereo_pair = feat.get_active_stereo_pair()
            lost_features_coords[i, 0] = feat_id
            lost_features_coords[i, 1] = stereo_pair[1][0]
            lost_features_coords[i, 2] = stereo_pair[1][1]
            if stereo_pair[2] is not None:
                lost_features_coords[i, 3] = stereo_pair[2][0]
                lost_features_coords[i, 4] = stereo_pair[2][1]
            else:
                lost_features_coords[i, 3] = np.nan
                lost_features_coords[i, 4] = np.nan
            lost_features_coords[i, 5] = feat.state.value

        return length, lost_features_coords.astype(np.float32)
