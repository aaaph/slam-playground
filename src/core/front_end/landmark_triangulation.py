from typing import Protocol, Self

import gtsam
import numpy as np
from numpy.typing import NDArray

from core.camera_model.stereo_camera_ctx import StereoContext


class LandmarkTriangulatorProtocol(Protocol):
    """Landmark triangulator contract."""

    def triangulate_mixed(
        self, left_uvs: NDArray[np.float64], right_uvs: NDArray[np.float64], left_poses: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Triangulate mixed monocular/stereo observations."""


class LandmarkTriangulator(LandmarkTriangulatorProtocol):
    """Landmark triangulation using GTSAM."""

    def __init__(self, stereo_k: NDArray[np.float32], rect0_from_rect1: NDArray[np.float32]) -> None:
        """Initialize the GTSAM landmark triangulator."""
        self.stereo_k = stereo_k
        fx = stereo_k[0, 0]
        fy = stereo_k[1, 1]
        skew = stereo_k[0, 1]
        cx = stereo_k[0, 2]
        cy = stereo_k[1, 2]
        self.stereo_k_gtsam = gtsam.Cal3_S2(fx, fy, skew, cx, cy)
        self.rect0_from_rect1 = rect0_from_rect1
        self.tri_params = gtsam.TriangulationParameters(rankTolerance=1e-6)

    @classmethod
    def default_factory(cls, stereo_ctx: StereoContext) -> Self:
        """Create a default GTSAM landmark triangulator."""
        baseline = stereo_ctx.baseline
        rect0_from_rect1 = np.eye(4, dtype=np.float32)
        rect0_from_rect1[0, 3] = baseline
        return cls(stereo_ctx.stereo_k, rect0_from_rect1)

    def triangulate_mixed(
        self, left_uvs: NDArray[np.float64], right_uvs: NDArray[np.float64], left_poses: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Triangulate mixed observations."""
        obs_num = left_uvs.shape[0]

        cameras = gtsam.CameraSetCal3_S2()
        measurements = []

        for i in range(obs_num):
            camera_left = gtsam.PinholeCameraCal3_S2(gtsam.Pose3(left_poses[i]), self.stereo_k_gtsam)
            cameras.append(camera_left)
            measurements.append(gtsam.Point2(left_uvs[i, 0], left_uvs[i, 1]))

            if np.all(np.isfinite(right_uvs[i])):
                right_pose = left_poses[i] @ self.rect0_from_rect1
                camera_right = gtsam.PinholeCameraCal3_S2(gtsam.Pose3(right_pose), self.stereo_k_gtsam)
                cameras.append(camera_right)
                measurements.append(gtsam.Point2(right_uvs[i, 0], right_uvs[i, 1]))

        tri_result = gtsam.triangulateSafe(
            cameras=cameras,
            measurements=measurements,
            params=self.tri_params,
        )

        if not tri_result.valid():
            return np.full((3,), np.nan, dtype=np.float64)

        point = tri_result.get()
        refined_point = gtsam.triangulateNonlinear(
            cameras=cameras,
            measurements=measurements,
            initialEstimate=point,
        )
        return np.asarray(refined_point, dtype=np.float64)
