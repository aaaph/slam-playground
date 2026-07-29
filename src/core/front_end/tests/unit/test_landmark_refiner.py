import numpy as np

from core.camera_model.stereo_camera_ctx import StereoContext
from core.front_end.landmark_refiner import LandmarkRefiner, LandmarkRefineStatus
from core.transformations.special_euclidian_3_dim import SE3


def _project(point_in_anchor: np.ndarray, anchor_from_cam: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Project an anchor-frame point into one camera."""
    cam_from_anchor_rot = anchor_from_cam[:3, :3].T
    point_in_cam = cam_from_anchor_rot @ (point_in_anchor - anchor_from_cam[:3, 3])
    return np.array(
        [
            k[0, 0] * point_in_cam[0] / point_in_cam[2] + k[0, 2],
            k[1, 1] * point_in_cam[1] / point_in_cam[2] + k[1, 2],
        ],
        dtype=np.float64,
    )


def test_refine_point_gn_uses_anchor_from_camera_poses() -> None:
    """GN refinement should optimize an anchor-frame point against anchor-from-camera poses."""
    k = np.array([[120.0, 0.0, 50.0], [0.0, 120.0, 50.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    stereo_ctx = StereoContext(
        resolution=(100, 100),
        stereo_k=k,
        cam0_k=k,
        cam1_k=k,
        baseline=0.1,
        cam0_in_body_se3=SE3.identity(),
        cam1_in_body_se3=SE3.identity(),
    )
    refiner = LandmarkRefiner(stereo_ctx, max_iterations=10, min_delta=1e-12)

    theta = 0.2
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    anchor_from_camera = np.tile(np.eye(4, dtype=np.float64), (3, 1, 1))
    anchor_from_camera[1, :3, 3] = np.array([0.5, 0.0, 0.0])
    anchor_from_camera[2, :3, :3] = np.array(
        [
            [cos_theta, 0.0, sin_theta],
            [0.0, 1.0, 0.0],
            [-sin_theta, 0.0, cos_theta],
        ],
        dtype=np.float64,
    )
    anchor_from_camera[2, :3, 3] = np.array([0.9, 0.1, 0.0])

    point_in_anchor = np.array([1.0, 0.4, 4.0], dtype=np.float64)
    uvs = np.array([_project(point_in_anchor, pose, k) for pose in anchor_from_camera], dtype=np.float64)
    initial_guess = np.array([1.2, 0.25, 3.5], dtype=np.float64)

    status, refined = refiner.refine_point_gn(initial_guess, uvs, anchor_from_camera)

    assert status == LandmarkRefineStatus.SUCCESS
    np.testing.assert_allclose(refined, point_in_anchor, atol=1e-6)
