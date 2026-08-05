import numpy as np
from numpy.typing import NDArray

from core.front_end.gyro_bearing_estimation import GyroDelta
from core.front_end.landmark_initialization import LandmarkInitializationFrameSchema
from core.pose_tracker.feature_triangulation import StereoTriangulationSchema
from core.pose_tracker.inertial_integration import ImuBatch, ImuSchema
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


class _GyroBearingEstimationStub:
    """Minimal gyro-bearing estimator stub."""

    def __init__(self) -> None:
        self.frame_id: int | None = None
        self.timestamp_ns: float | None = None
        self.feat_ids: np.ndarray | None = None
        self.left_bearings: np.ndarray | None = None
        self.gyro_delta: GyroDelta | None = None

    def add_observations(
        self,
        frame_id: int,
        timestamp_ns: float,
        feat_ids: NDArray[np.int32],
        left_bearings: NDArray[np.float64],
        gyro_delta: GyroDelta,
    ) -> slice:
        self.frame_id = frame_id
        self.timestamp_ns = timestamp_ns
        self.feat_ids = feat_ids.copy()
        self.left_bearings = left_bearings.copy()
        self.gyro_delta = gyro_delta
        return slice(4, 4 + feat_ids.shape[0])


class TestVIOFrontend:
    """VIO frontend unit tests."""

    def test_add_gyro_bearing_observations_collects_tracked_finite_bearings(self) -> None:
        """Gyro-bearing bootstrap should collect bearing rows and compact gyro delta."""
        frontend = VIOFrontend.__new__(VIOFrontend)
        frontend.state = np.zeros(16, dtype=np.float64)
        estimator = _GyroBearingEstimationStub()
        frontend.gyro_bearing_estimation = estimator
        stereo_frame = np.full((3, StereoTriangulationSchema.count()), np.nan, dtype=np.float32)
        stereo_frame[:, StereoTriangulationSchema.FEAT_ID] = [10, 20, 30]
        stereo_frame[:, StereoTriangulationSchema.LEFT_BEARING] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        tracking_mask = np.array([True, False, True], dtype=np.bool_)
        imu_rows = np.zeros((2, ImuSchema.count()), dtype=np.float64)
        imu_rows[:, ImuSchema.TIMESTAMP] = [0.0, 10_000_000.0]
        imu_rows[1, ImuSchema.GYRO_SLICE] = [0.0, 0.0, 0.2]
        imu_rows[1, ImuSchema.DT] = 0.01

        obs_slice = frontend.add_gyro_bearing_observations(
            frame_id=5,
            timestamp_ns=10_000_000.0,
            stereo_frame=stereo_frame,
            tracking_mask=tracking_mask,
            imu_batch=ImuBatch(imu_rows),
        )

        assert obs_slice == slice(4, 6)
        assert estimator.frame_id == 5
        assert estimator.timestamp_ns == 10_000_000.0
        np.testing.assert_array_equal(estimator.feat_ids, np.array([10, 30], dtype=np.int32))
        left_bearings = estimator.left_bearings
        assert left_bearings is not None
        np.testing.assert_allclose(
            left_bearings,
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        )
        gyro_delta = estimator.gyro_delta
        assert gyro_delta is not None
        assert gyro_delta.dt_sec == 0.01
        assert gyro_delta.bias_jacobian.shape == (3, 3)
        np.testing.assert_allclose(gyro_delta.rotation.as_rotvec(), np.array([0.0, 0.0, 0.002], dtype=np.float64))

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
