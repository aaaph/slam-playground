import numpy as np

from core.front_end.landmark_initialization import LandmarkInitializationFrameSchema
from core.pose_tracker.feature_triangulation import StereoTriangulationSchema
from core.transformations.special_euclidian_3_dim import SE3
from pipeline.nodes.vio_frontend import VIOFrontend


class _LandmarkInitializationApplyStub:
    """Minimal landmark initialization stub for observation application tests."""

    def __init__(self) -> None:
        """Initialize the apply-observation stub."""
        self.cam0_in_world: np.ndarray | None = None
        self.tracking_mask: np.ndarray | None = None
        self.stereo_frame: np.ndarray | None = None
        self.success_mask = np.array([True, False, False], dtype=np.bool_)
        self.landmark_frame = np.full((3, LandmarkInitializationFrameSchema.count()), np.nan, dtype=np.float64)

    def apply_observation_frame(
        self,
        cam0_in_world: np.ndarray,
        tracking_mask: np.ndarray,
        stereo_frame: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Record frame extension inputs and return frame-aligned landmark results."""
        self.cam0_in_world = cam0_in_world.copy()
        self.tracking_mask = tracking_mask.copy()
        self.stereo_frame = stereo_frame.copy()
        return self.success_mask.copy(), self.landmark_frame.copy()


class TestVIOFrontend:
    """VIO frontend unit tests."""

    def test_build_stereo_points_for_visualization_uses_triangulated_stereo_rows(self) -> None:
        """Stereo pointcloud rows should come from one-shot stereo XYZ columns."""
        stereo_frame = np.full((3, StereoTriangulationSchema.count()), np.nan, dtype=np.float32)
        stereo_frame[:, StereoTriangulationSchema.FEAT_ID] = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        stereo_frame[:, StereoTriangulationSchema.XYZ] = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=np.float32,
        )
        stereo_mask = np.array([True, False, True], dtype=np.bool_)

        stereo_points = VIOFrontend.build_stereo_points_for_visualization(stereo_mask, stereo_frame)

        np.testing.assert_allclose(
            stereo_points,
            np.array(
                [
                    [10.0, 1.0, 2.0, 3.0],
                    [30.0, 7.0, 8.0, 9.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_build_landmarks_for_visualization_uses_landmark_xyz_in_current_cam0_frame(self) -> None:
        """Landmark pointcloud rows should use cached landmark XYZ, not stereo-frame prefix columns."""
        landmark_frame = np.full((2, LandmarkInitializationFrameSchema.count()), np.nan, dtype=np.float64)
        landmark_frame[:, LandmarkInitializationFrameSchema.FEAT_ID] = np.array([10.0, 20.0])
        landmark_frame[:, LandmarkInitializationFrameSchema.TIMESTAMP] = np.array([1000.0, 2000.0])
        landmark_frame[:, LandmarkInitializationFrameSchema.LEFT_UV] = np.array([[110.0, 111.0], [120.0, 121.0]])
        landmark_frame[:, LandmarkInitializationFrameSchema.LANDMARK_XYZ] = np.array(
            [
                [11.0, 2.0, 3.0],
                [18.0, -1.0, 0.5],
            ],
            dtype=np.float64,
        )
        success_mask = np.array([True, False], dtype=np.bool_)
        cam0_in_world = SE3(t=np.array([10.0, 0.0, 0.0], dtype=np.float64))

        landmarks = VIOFrontend.build_landmarks_for_visualization(
            success_mask,
            landmark_frame,
            cam0_in_world,
        )

        np.testing.assert_allclose(
            landmarks,
            np.array([[10.0, 1.0, 2.0, 3.0]], dtype=np.float32),
        )
