import numpy as np

from core.front_end.keyframe import (
    KF,
    keyframe_schema,
)
from core.front_end.keyframe_selector import SelectReason
from core.front_end.landmark_cache import LandmarkCacheStatus
from core.front_end.landmark_initialization import LandmarkInitializationFrameSchema
from core.pose_tracker.feature_triangulation import StereoTriangulationStatus


class TestKeyframeSchema:
    """Unit test for keyframe schema."""

    @staticmethod
    def make_kf(keyframe_id: int, timestamp: float, reason: SelectReason) -> KF:
        """Create a test keyframe."""
        landmark_frame = np.full((2, LandmarkInitializationFrameSchema.count()), np.nan, dtype=np.float64)
        landmark_frame[:, LandmarkInitializationFrameSchema.FEAT_ID] = [1, 2]
        landmark_frame[:, LandmarkInitializationFrameSchema.TIMESTAMP] = [0.0, 0.5]
        stereo_uv = slice(
            LandmarkInitializationFrameSchema.LEFT_U,
            LandmarkInitializationFrameSchema.RIGHT_V + 1,
        )
        landmark_frame[:, stereo_uv] = [[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, np.nan, np.nan]]
        landmark_frame[:, LandmarkInitializationFrameSchema.LIFECYCLE] = [0, 1]
        landmark_frame[:, LandmarkInitializationFrameSchema.AGE] = [3, 5]
        landmark_frame[:, LandmarkInitializationFrameSchema.STEREO_SCORE] = 0.0
        landmark_frame[:, LandmarkInitializationFrameSchema.FRAME_PIXEL_DISPLACEMENT] = [0.5, 0.0]
        left_bearing = slice(
            LandmarkInitializationFrameSchema.LEFT_BEARING_X,
            LandmarkInitializationFrameSchema.LEFT_BEARING_Z + 1,
        )
        landmark_frame[:, left_bearing] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        landmark_frame[:, LandmarkInitializationFrameSchema.STEREO_STATUS] = [
            StereoTriangulationStatus.TRIANGULATED.value,
            StereoTriangulationStatus.BAD_STEREO.value,
        ]
        landmark_frame[:, LandmarkInitializationFrameSchema.LANDMARK_STATUS] = [
            LandmarkCacheStatus.COMPLETED.value,
            LandmarkCacheStatus.OBSERVING.value,
        ]
        landmark_frame[:, LandmarkInitializationFrameSchema.TRACKED] = [1.0, 1.0]
        landmark_frame[:, LandmarkInitializationFrameSchema.LANDMARK_XYZ] = [
            [1.0, 2.0, 3.0],
            [np.nan, np.nan, np.nan],
        ]
        return KF(
            keyframe_id=keyframe_id,
            timestamp=timestamp,
            select_reasons=[reason],
            state=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            imu_batch=np.array(
                [
                    [0.0, 0.1, 0.2, 0.3, 1.1, 1.2, 1.3, 0.0],
                    [1.0, 0.4, 0.5, 0.6, 1.4, 1.5, 1.6, 0.01],
                    [2.0, 0.7, 0.8, 0.9, 1.7, 1.8, 1.9, 0.01],
                ],
                dtype=np.float64,
            ),
            landmark_frame=landmark_frame,
            vibration_detected=False,
            non_zero_velocity_detected=False,
        )

    def test_keyframe_schema(self):
        """Test that the keyframe schema is correct."""
        kf = self.make_kf(1, 0.0, SelectReason.LOW_CONNECTIVITY)
        arrow = kf.as_arrow()
        assert arrow.schema == keyframe_schema
        assert kf.landmark_frame.shape[1] == LandmarkInitializationFrameSchema.count()

        restored_kf = KF.from_arrow(arrow)
        assert restored_kf.keyframe_id == kf.keyframe_id
        assert restored_kf.timestamp == kf.timestamp
        assert restored_kf.select_reasons == kf.select_reasons
        assert restored_kf.vibration_detected is kf.vibration_detected
        assert restored_kf.non_zero_velocity_detected is kf.non_zero_velocity_detected
        np.testing.assert_allclose(restored_kf.state, kf.state.astype(np.float32))
        np.testing.assert_allclose(restored_kf.imu_batch, kf.imu_batch.astype(np.float64))
        np.testing.assert_allclose(
            restored_kf.landmark_frame,
            kf.landmark_frame.astype(np.float64),
            equal_nan=True,
        )

    def test_to_record_batch_and_list_from_arrow(self):
        """Test that multiple keyframes can be packed into one record batch."""
        first_kf = self.make_kf(1, 0.0, SelectReason.LOW_CONNECTIVITY)
        second_kf = self.make_kf(2, 10.0, SelectReason.PARALLAX)

        arrow = KF.to_record_batch([first_kf, second_kf])

        assert arrow.schema == keyframe_schema
        assert arrow.num_rows == 2
        np.testing.assert_array_equal(arrow.column("keyframe_id").to_numpy(), np.array([1, 2]))
        np.testing.assert_allclose(arrow.column("timestamp").to_numpy(), np.array([0.0, 10.0]))

        restored_keyframes = KF.list_from_arrow(arrow)

        assert len(restored_keyframes) == 2
        assert restored_keyframes[0].keyframe_id == first_kf.keyframe_id
        assert restored_keyframes[0].select_reasons == first_kf.select_reasons
        np.testing.assert_allclose(restored_keyframes[0].state, first_kf.state.astype(np.float32))
        np.testing.assert_allclose(
            restored_keyframes[0].landmark_frame,
            first_kf.landmark_frame.astype(np.float64),
            equal_nan=True,
        )
        assert restored_keyframes[1].keyframe_id == second_kf.keyframe_id
        assert restored_keyframes[1].select_reasons == second_kf.select_reasons
        np.testing.assert_allclose(restored_keyframes[1].state, second_kf.state.astype(np.float32))
        np.testing.assert_allclose(
            restored_keyframes[1].landmark_frame,
            second_kf.landmark_frame.astype(np.float64),
            equal_nan=True,
        )
