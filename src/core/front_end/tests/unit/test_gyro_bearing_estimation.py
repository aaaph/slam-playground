import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from core.front_end.gyro_bearing_estimation import GyroBearingEstimation, GyroDelta


def test_add_observations_accumulates_and_wraps_mixed_type_arena_columns() -> None:
    estimator = GyroBearingEstimation(observation_capacity=3, frame_capacity=2)
    gyro_delta0 = GyroDelta(
        rotation=Rotation.from_rotvec(np.array([0.1, 0.0, 0.0])),
        bias_jacobian=np.eye(3, dtype=np.float64),
        dt_sec=0.01,
    )
    gyro_delta1 = GyroDelta(
        rotation=Rotation.from_rotvec(np.array([0.0, 0.2, 0.0])),
        bias_jacobian=np.eye(3, dtype=np.float64) * 2.0,
        dt_sec=0.02,
    )
    gyro_delta2 = GyroDelta(
        rotation=Rotation.from_rotvec(np.array([0.0, 0.0, 0.3])),
        bias_jacobian=np.eye(3, dtype=np.float64) * 3.0,
        dt_sec=0.03,
    )

    first_range = estimator.add_observations(
        frame_id=7,
        timestamp_ns=100.0,
        feat_ids=np.array([10, 20], dtype=np.int32),
        left_bearings=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        gyro_delta=gyro_delta0,
    )
    second_range = estimator.add_observations(
        frame_id=8,
        timestamp_ns=150.0,
        feat_ids=np.array([30], dtype=np.int32),
        left_bearings=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        gyro_delta=gyro_delta1,
    )
    third_range = estimator.add_observations(
        frame_id=9,
        timestamp_ns=200.0,
        feat_ids=np.array([40, 50], dtype=np.int32),
        left_bearings=np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float64),
        gyro_delta=gyro_delta2,
    )

    arena = estimator._arena  # noqa: SLF001
    assert first_range == slice(0, 2)
    assert second_range == slice(2, 3)
    assert third_range == slice(0, 2)
    assert arena.frame_size == 1
    assert arena.observation_size == 2
    assert arena.frame_ids.dtype == np.int64
    assert arena.timestamps_ns.dtype == np.float64
    assert arena.feat_ids.dtype == np.int32
    np.testing.assert_array_equal(arena.frame_ids[:1], np.array([9], dtype=np.int64))
    np.testing.assert_allclose(arena.gyro_delta_rotvecs[:1], np.array([[0.0, 0.0, 0.3]], dtype=np.float64))
    np.testing.assert_allclose(arena.gyro_delta_bias_jacobians[:1], np.eye(3)[None, :, :] * 3.0)
    np.testing.assert_allclose(arena.gyro_delta_dt_sec[:1], np.array([0.03], dtype=np.float64))
    np.testing.assert_array_equal(arena.observation_frame_slots[:2], np.array([0, 0], dtype=np.int32))
    np.testing.assert_array_equal(arena.feat_ids[:2], np.array([40, 50], dtype=np.int32))
    np.testing.assert_allclose(
        arena.left_bearings[:2],
        np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float64),
    )


def test_add_observations_rejects_single_batch_larger_than_arena() -> None:
    estimator = GyroBearingEstimation(observation_capacity=1, frame_capacity=1)

    with pytest.raises(ValueError, match="Observation batch does not fit"):
        estimator.add_observations(
            frame_id=1,
            timestamp_ns=1.0,
            feat_ids=np.array([10, 20], dtype=np.int32),
            left_bearings=np.ones((2, 3), dtype=np.float64),
            gyro_delta=GyroDelta(
                rotation=Rotation.identity(),
                bias_jacobian=np.eye(3, dtype=np.float64),
                dt_sec=0.0,
            ),
        )
