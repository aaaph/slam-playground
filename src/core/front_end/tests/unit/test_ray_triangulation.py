import numpy as np

from core.camera_model.stereo_camera_ctx import StereoContext
from core.front_end.ray_triangulation import RayTriangulation, TriangulationStatus
from core.transformations.special_euclidian_3_dim import SE3


def test_triangulate_feature_observations_matches_rectified_stereo_formula() -> None:
    """Ray triangulation should match disparity depth for a rectified stereo pair."""
    stereo_k = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    stereo_ctx = StereoContext(
        resolution=(100, 100),
        stereo_k=stereo_k,
        cam0_k=stereo_k,
        cam1_k=stereo_k,
        baseline=0.1,
        cam0_in_body_se3=SE3.identity(),
        cam1_in_body_se3=SE3.identity(),
    )
    triangulator = RayTriangulation.default_factory(stereo_ctx)
    uvs = np.array([[100.0, 115.0], [95.0, 115.0]], dtype=np.float64)
    poses = np.tile(np.eye(4, dtype=np.float64), (2, 1, 1))
    poses[1, 0, 3] = 0.1

    status, point_in_anchor = triangulator.triangulate_feature_observations(uvs, poses)

    assert status == TriangulationStatus.SUCCESS
    np.testing.assert_allclose(point_in_anchor, np.array([1.0, 1.3, 2.0]), atol=1e-9)
