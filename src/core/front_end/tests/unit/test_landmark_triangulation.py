import numpy as np
import pytest

from core.front_end import landmark_triangulation
from core.front_end.landmark_triangulation import LandmarkTriangulator, TriangulationStatus

STEREO_K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=np.float32)
BASELINE_M = 0.1
POINT_IN_WORLD = np.array([0.2, 0.1, 2.0], dtype=np.float64)


def _triangulator() -> LandmarkTriangulator:
    """Build a triangulator with a simple rectified stereo rig."""
    rect0_from_rect1 = np.eye(4, dtype=np.float32)
    rect0_from_rect1[0, 3] = BASELINE_M
    return LandmarkTriangulator(STEREO_K, rect0_from_rect1)


def _project_points(world_from_cams: np.ndarray, point_in_world: np.ndarray) -> np.ndarray:
    """Project a world point into each camera pose."""
    point_in_cams = np.einsum(
        "nij,nj->ni",
        world_from_cams[:, :3, :3].transpose(0, 2, 1),
        point_in_world - world_from_cams[:, :3, 3],
    )
    fx = STEREO_K[0, 0]
    fy = STEREO_K[1, 1]
    cx = STEREO_K[0, 2]
    cy = STEREO_K[1, 2]
    return np.column_stack(
        (
            fx * point_in_cams[:, 0] / point_in_cams[:, 2] + cx,
            fy * point_in_cams[:, 1] / point_in_cams[:, 2] + cy,
        )
    )


def _mixed_observations(obs_num: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic left/right image measurements for a fixed world point."""
    left_poses = np.tile(np.eye(4, dtype=np.float64), (obs_num, 1, 1))
    left_poses[:, 0, 3] = np.arange(obs_num, dtype=np.float64) * 0.2

    rect0_from_rect1 = np.eye(4, dtype=np.float64)
    rect0_from_rect1[0, 3] = BASELINE_M
    right_poses = left_poses @ rect0_from_rect1
    return _project_points(left_poses, POINT_IN_WORLD), _project_points(right_poses, POINT_IN_WORLD), left_poses


def test_triangulate_mixed_accepts_stereo_and_monocular_measurements() -> None:
    """Mixed triangulation should use finite right measurements and keep monocular rows."""
    triangulator = _triangulator()
    left_uvs, right_uvs, left_poses = _mixed_observations()
    right_uvs[1] = np.nan

    status, point_in_world = triangulator.triangulate_mixed(left_uvs, right_uvs, left_poses)

    assert status == TriangulationStatus.SUCCESS
    np.testing.assert_allclose(point_in_world, POINT_IN_WORLD, atol=1e-9)


def test_triangulate_mixed_reports_big_reprojection_error_for_inconsistent_measurement() -> None:
    """A single inconsistent measurement should be rejected by reprojection cost."""
    triangulator = _triangulator()
    left_uvs, right_uvs, left_poses = _mixed_observations(obs_num=3)
    left_uvs[-1, 0] += triangulator.projection_error_threshold * 4.0

    status, point_in_world = triangulator.triangulate_mixed(left_uvs, right_uvs, left_poses)

    assert status == TriangulationStatus.BIG_REPROJECTION_ERROR
    assert np.all(np.isfinite(point_in_world))


def test_triangulate_mixed_reports_covariance_not_valid_when_marginals_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Marginals failures should be converted into a covariance status."""
    triangulator = _triangulator()
    left_uvs, right_uvs, left_poses = _mixed_observations()

    def _raise_runtime_error(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("singular covariance")

    monkeypatch.setattr(landmark_triangulation.gtsam, "Marginals", _raise_runtime_error)

    status, point_in_world = triangulator.triangulate_mixed(left_uvs, right_uvs, left_poses)

    assert status == TriangulationStatus.COVARIANCE_NOT_VALID
    np.testing.assert_allclose(point_in_world, POINT_IN_WORLD, atol=1e-9)


def test_triangulate_mixed_reports_big_depth_variance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Depth variance above the configured gate should reject the triangulated point."""
    triangulator = _triangulator()
    left_uvs, right_uvs, left_poses = _mixed_observations()
    depth_variance_m2 = triangulator.depth_variance_threshold_m2 + 1.0

    def _large_depth_covariance(
        _self: LandmarkTriangulator, _cameras: object, _measurements: object, _point: object
    ) -> np.ndarray:
        return np.diag(np.array([1.0, 1.0, depth_variance_m2], dtype=np.float64))

    monkeypatch.setattr(LandmarkTriangulator, "compute_covariance_matrix", _large_depth_covariance)

    status, point_in_world = triangulator.triangulate_mixed(left_uvs, right_uvs, left_poses)

    assert status == TriangulationStatus.BIG_DEPTH_VARIANCE
    np.testing.assert_allclose(point_in_world, POINT_IN_WORLD, atol=1e-9)


def test_triangulate_mixed_reports_invalid_point_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refined point behind the cameras should be rejected before projection."""
    triangulator = _triangulator()
    left_uvs, right_uvs, left_poses = _mixed_observations()

    class _ValidTriangulationResult:
        def valid(self) -> bool:
            return True

        def get(self) -> np.ndarray:
            return POINT_IN_WORLD.copy()

    def _fake_triangulate_safe(**_kwargs: object) -> _ValidTriangulationResult:
        return _ValidTriangulationResult()

    def _fake_triangulate_nonlinear(**_kwargs: object) -> np.ndarray:
        return np.array([0.2, 0.1, -2.0], dtype=np.float64)

    monkeypatch.setattr(landmark_triangulation.gtsam, "triangulateSafe", _fake_triangulate_safe)
    monkeypatch.setattr(landmark_triangulation.gtsam, "triangulateNonlinear", _fake_triangulate_nonlinear)

    status, point_in_world = triangulator.triangulate_mixed(left_uvs, right_uvs, left_poses)

    assert status == TriangulationStatus.INVALID_POINT_DEPTH
    assert np.all(np.isnan(point_in_world))
