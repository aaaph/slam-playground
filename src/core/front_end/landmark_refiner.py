from enum import Enum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from core.camera_model.stereo_camera_ctx import StereoContext


class LandmarkRefineStatus(Enum):
    """Status of the landmark refinement."""

    SUCCESS = 0
    DEPTH_NEGATIVE = 1
    MAX_ITERATIONS_REACHED = 2
    SOLVER_ERROR = 3


class Refiner(Protocol):
    """Refiner contract for landmark refinement."""

    def refine_point_gn(
        self, initial_guess: NDArray[np.float64], uvs: NDArray[np.float64], poses: NDArray[np.float64]
    ) -> tuple[LandmarkRefineStatus, NDArray[np.float64]]:
        """Refine the point via GN optimization."""


class LandmarkRefiner(Refiner):
    """Component for refining the landmarks via GN optimization."""

    def __init__(self, stereo_ctx: StereoContext, max_iterations: int = 3, min_delta: float = 1e-4) -> None:
        """Initialize the landmark refiner."""
        self._stereo_ctx = stereo_ctx
        self._stereo_k = stereo_ctx.stereo_k
        self._max_iterations = max_iterations
        self._min_delta = min_delta

    def refine_point_gn(
        self, initial_guess: NDArray[np.float64], uvs: NDArray[np.float64], poses: NDArray[np.float64]
    ) -> tuple[LandmarkRefineStatus, NDArray[np.float64]]:
        """Refine an anchor-frame point via GN reprojection optimization."""
        fx, fy = self._stereo_k[0, 0], self._stereo_k[1, 1]
        cx, cy = self._stereo_k[0, 2], self._stereo_k[1, 2]

        p = initial_guess.copy()
        uvs_num = uvs.shape[0]

        for _ in range(self._max_iterations):
            h = np.zeros((3, 3), dtype=np.float64)
            g = np.zeros((3,), dtype=np.float64)
            total_error = 0.0

            for i in range(uvs_num):
                anchor_from_cam = poses[i, :, :]
                anchor_from_cam_rot = anchor_from_cam[:3, :3]
                anchor_from_cam_t = anchor_from_cam[:3, 3]
                cam_from_anchor_rot = anchor_from_cam_rot.T

                p_cam = cam_from_anchor_rot @ (p - anchor_from_cam_t)
                x, y, z = p_cam

                if z <= 0:
                    return LandmarkRefineStatus.DEPTH_NEGATIVE, p

                inv_z = 1.0 / z
                inv_z2 = inv_z * inv_z

                u_proj = fx * x * inv_z + cx
                v_proj = fy * y * inv_z + cy

                r = uvs[i, :] - np.array([u_proj, v_proj])
                total_error += np.dot(r, r)

                j_cam = np.array(
                    [
                        [fx * inv_z, 0.0, -fx * x * inv_z2],
                        [0.0, fy * inv_z, -fy * y * inv_z2],
                    ],
                    dtype=np.float64,
                )

                j = j_cam @ cam_from_anchor_rot

                h += j.T @ j
                g += j.T @ r

            try:
                delta = np.linalg.solve(h, g)
                p += delta
                if np.linalg.norm(delta) < self._min_delta:
                    return LandmarkRefineStatus.SUCCESS, p
            except np.linalg.LinAlgError:
                return LandmarkRefineStatus.SOLVER_ERROR, p

        return LandmarkRefineStatus.SUCCESS, p
