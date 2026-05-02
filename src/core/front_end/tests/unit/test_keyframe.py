import numpy as np

from core.front_end.keyframe import (
    KF,
    ActiveTrackSchema,
    keyframe_schema,
)
from core.front_end.keyframe_selector import SelectReason


class TestKeyframeSchema:
    """Unit test for keyframe schema."""

    @staticmethod
    def make_kf(keyframe_id: int, timestamp: float, reason: SelectReason) -> KF:
        """Create a test keyframe."""
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
            active_track=np.array(
                [
                    [1, 0.0, 10.0, 11.0, 12.0, 13.0, 0, 3, 0.0, 1.0, 2.0, 3.0],
                    [2, 0.5, 20.0, 21.0, np.nan, np.nan, 1, 5, 0.0, np.nan, np.nan, np.nan],
                ],
                dtype=np.float32,
            ),
            vibration_detected=False,
            non_zero_velocity_detected=False,
        )

    def test_keyframe_schema(self):
        """Test that the keyframe schema is correct."""
        kf = self.make_kf(1, 0.0, SelectReason.LOW_CONNECTIVITY)
        arrow = kf.as_arrow()
        assert arrow.schema == keyframe_schema
        assert kf.active_track.shape[1] == ActiveTrackSchema.count()

        restored_kf = KF.from_arrow(arrow)
        assert restored_kf.keyframe_id == kf.keyframe_id
        assert restored_kf.timestamp == kf.timestamp
        assert restored_kf.select_reasons == kf.select_reasons
        assert restored_kf.vibration_detected is kf.vibration_detected
        assert restored_kf.non_zero_velocity_detected is kf.non_zero_velocity_detected
        np.testing.assert_allclose(restored_kf.state, kf.state.astype(np.float32))
        np.testing.assert_allclose(restored_kf.imu_batch, kf.imu_batch.astype(np.float64))
        np.testing.assert_allclose(restored_kf.active_track, kf.active_track.astype(np.float32), equal_nan=True)

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
            restored_keyframes[0].active_track,
            first_kf.active_track.astype(np.float32),
            equal_nan=True,
        )
        assert restored_keyframes[1].keyframe_id == second_kf.keyframe_id
        assert restored_keyframes[1].select_reasons == second_kf.select_reasons
        np.testing.assert_allclose(restored_keyframes[1].state, second_kf.state.astype(np.float32))
        np.testing.assert_allclose(
            restored_keyframes[1].active_track,
            second_kf.active_track.astype(np.float32),
            equal_nan=True,
        )
