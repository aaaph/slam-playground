from enum import IntEnum, IntFlag, auto
from typing import Any, Protocol, Self

import gtsam
import numpy as np
from numpy.typing import NDArray

from core.camera_model.stereo_camera_ctx import StereoContext

P = gtsam.symbol_shorthand.P
X = gtsam.symbol_shorthand.X


class TriangulationStatus(IntEnum):
    """Triangulation status."""

    SUCCESS = 0
    NOT_VALID = 1
    BIG_REPROJECTION_ERROR = 2
    BIG_DEPTH_VARIANCE = 3
    COVARIANCE_NOT_VALID = 4
    INVALID_POINT_DEPTH = 5


class LandmarkTriangulationFlags(IntFlag):
    """Landmark triangulation options."""

    NONE = 0
    POINT_NONLINEAR_REFINE = auto()
    COVARIANCE_CHECK = auto()
    REPROJECT_ERROR_CHECK = auto()

    DEFAULT = POINT_NONLINEAR_REFINE | COVARIANCE_CHECK | REPROJECT_ERROR_CHECK


class LandmarkTriangulatorProtocol(Protocol):
    """Landmark triangulator contract."""

    def triangulate_mixed(
        self, left_uvs: NDArray[np.float64], right_uvs: NDArray[np.float64], left_poses: NDArray[np.float64]
    ) -> tuple[TriangulationStatus, NDArray[np.float64]]:
        """Triangulate mixed monocular/stereo observations."""


class LandmarkTriangulator(LandmarkTriangulatorProtocol):
    """Landmark triangulation using GTSAM."""

    def __init__(
        self,
        stereo_k: NDArray[np.float32],
        rect0_from_rect1: NDArray[np.float32],
        flags: LandmarkTriangulationFlags = LandmarkTriangulationFlags.DEFAULT,
    ) -> None:
        """Initialize the GTSAM landmark triangulator."""
        self.flags = flags
        self.stereo_k = stereo_k
        fx = stereo_k[0, 0]
        fy = stereo_k[1, 1]
        skew = stereo_k[0, 1]
        cx = stereo_k[0, 2]
        cy = stereo_k[1, 2]
        self.stereo_k_gtsam = gtsam.Cal3_S2(fx, fy, skew, cx, cy)
        self.rect0_from_rect1 = rect0_from_rect1
        self.tri_params = gtsam.TriangulationParameters(rankTolerance=1e-6)

        self.projection_error_threshold = 5.0
        self.depth_variance_threshold_m = 5.0
        self.depth_variance_threshold_m2 = self.depth_variance_threshold_m**2

        self.measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 2.0)
        self.pose_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-6)

    @classmethod
    def default_factory(
        cls, stereo_ctx: StereoContext, flags: LandmarkTriangulationFlags = LandmarkTriangulationFlags.DEFAULT
    ) -> Self:
        """Create a default GTSAM landmark triangulator."""
        baseline = stereo_ctx.baseline
        rect0_from_rect1 = np.eye(4, dtype=np.float32)
        rect0_from_rect1[0, 3] = baseline
        return cls(stereo_ctx.stereo_k, rect0_from_rect1, flags)

    def compute_covariance_matrix(
        self, cameras: gtsam.CameraSetCal3_S2, measurements: list[Any], optimized_point: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute the covariance matrix."""
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        values.insert(P(0), optimized_point)

        for i in range(len(cameras)):
            cam = cameras[i]
            meas = measurements[i]

            values.insert(X(i), cam.pose())
            graph.add(gtsam.PriorFactorPose3(X(i), cam.pose(), self.pose_noise))

            factor = gtsam.GenericProjectionFactorCal3_S2(
                meas, self.measurement_noise, X(i), P(0), cam.calibration()
            )
            graph.add(factor)

        try:
            marginals = gtsam.Marginals(graph, values)
            covariance_matrix = marginals.marginalCovariance(P(0))
            covariance_matrix = 0.5 * (covariance_matrix + covariance_matrix.T)
            return np.asarray(covariance_matrix, dtype=np.float64)
        except RuntimeError:
            return np.full((3, 3), np.nan, dtype=np.float64)

    def triangulate_mixed(  # noqa: C901
        self, left_uvs: NDArray[np.float64], right_uvs: NDArray[np.float64], left_poses: NDArray[np.float64]
    ) -> tuple[TriangulationStatus, NDArray[np.float64]]:
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
            return TriangulationStatus.NOT_VALID, np.full((3,), np.nan, dtype=np.float64)

        point = tri_result.get()

        if self.flags & LandmarkTriangulationFlags.POINT_NONLINEAR_REFINE:
            point = gtsam.triangulateNonlinear(
                cameras=cameras,
                measurements=measurements,
                initialEstimate=point,
            )

        for cam in cameras:
            point_in_cam = cam.pose().transformTo(point)
            if point_in_cam[2] <= 0.0:
                return TriangulationStatus.INVALID_POINT_DEPTH, point

        if self.flags & LandmarkTriangulationFlags.REPROJECT_ERROR_CHECK:
            max_reprojection_error = 0.0
            for i in range(len(cameras)):
                cam = cameras[i]
                meas = measurements[i]

                predicted_uv = cam.project(point)
                pixel_error = np.linalg.norm(predicted_uv - meas)

                max_reprojection_error = max(max_reprojection_error, pixel_error)

            if max_reprojection_error > self.projection_error_threshold:
                return TriangulationStatus.BIG_REPROJECTION_ERROR, point

        if self.flags & LandmarkTriangulationFlags.COVARIANCE_CHECK:
            anchor_world_from_cam = left_poses[0]
            anchor_cam_from_world = anchor_world_from_cam[:3, :3].T
            point_covariance = self.compute_covariance_matrix(cameras, measurements, point)
            covariance_anchor_cam0 = anchor_cam_from_world @ point_covariance @ anchor_cam_from_world.T
            if not np.all(np.isfinite(covariance_anchor_cam0)):
                return TriangulationStatus.COVARIANCE_NOT_VALID, point

            anchor_depth_variance = covariance_anchor_cam0[2, 2]

            if anchor_depth_variance > self.depth_variance_threshold_m2:
                return TriangulationStatus.BIG_DEPTH_VARIANCE, point

        return TriangulationStatus.SUCCESS, np.asarray(point, dtype=np.float64)
