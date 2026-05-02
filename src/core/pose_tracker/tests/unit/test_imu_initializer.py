import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from core.pose_tracker.inertial_integration import ImuBuffer


class TestImuBuffer:
    """Unit test for ImuBuffer."""

    @pytest.fixture
    def imu_buffer(self) -> ImuBuffer:
        """Create an ImuBuffer."""
        return ImuBuffer(capacity=1000)

    def test_add_batch(self, imu_buffer: ImuBuffer) -> None:
        """Test the add_batch method of the ImuInitializer."""
        accel_batch = np.array([[0, 0, 9.81], [0, 0, 9.81], [0, 0, 9.81]])
        gyro_batch = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        timestamp_batch = np.array([1.0, 2.0, 3.0])
        imu_buffer.add_batch(accel_batch, gyro_batch, timestamp_batch)
        assert imu_buffer.buffer is not None

    def test_create_initial_state(self, imu_buffer: ImuBuffer) -> None:
        """Test the create_initial_bias method of the ImuInitializer."""
        accel_batch = np.full((100, 3), np.array([0, 0, 9.81]))
        gyro_batch = np.full((100, 3), np.array([0, 0, 0]))
        timestamp_batch = np.linspace(1.0, 100.0, 100)
        imu_buffer.add_batch(accel_batch, gyro_batch, timestamp_batch)
        init_state = imu_buffer.create_initial_state()
        assert init_state is not None
        assert np.allclose(init_state.gyro_bias, np.array([0, 0, 0]))
        assert np.allclose(init_state.accel_bias, np.array([0, 0, 0]))
        assert np.allclose(init_state.rotation.as_quat(), Rotation.from_matrix(np.eye(3)).as_quat())
        assert np.allclose(init_state.gyro_std, np.array([0, 0, 0]))
        assert np.allclose(init_state.accel_std, np.array([0, 0, 0]))
        assert np.allclose(init_state.gyro_mean, np.array([0, 0, 0]))
        assert np.allclose(init_state.accel_mean, np.array([0, 0, 9.81]))

    def test_size(self, imu_buffer: ImuBuffer) -> None:
        """Test the size attribute of the ImuInitializer."""
        assert imu_buffer.size == 0
        accel_batch = np.array([[0, 0, 9.81], [0, 0, 9.81], [0, 0, 9.81]])
        gyro_batch = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        timestamp_batch = np.array([1.0, 2.0, 3.0])
        imu_buffer.add_batch(accel_batch, gyro_batch, timestamp_batch)
        assert imu_buffer.size == 3

        accel_batch, gyro_batch = imu_buffer.get_batch()
        np.testing.assert_array_equal(accel_batch, imu_buffer.buffer[~np.isnan(imu_buffer.buffer[:, 1]), 1:4])
        np.testing.assert_array_equal(gyro_batch, imu_buffer.buffer[~np.isnan(imu_buffer.buffer[:, 4]), 4:7])

    def test_dt_batch(self, imu_buffer: ImuBuffer) -> None:
        """Test the dt_batch attribute of the ImuInitializer."""
        accel_batch = np.array([[0, 0, 9.81], [0, 0, 9.81], [0, 0, 9.81]])
        gyro_batch = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        timestamp_batch = np.array([1.0, 2.0, 3.0])
        imu_buffer.add_batch(accel_batch, gyro_batch, timestamp_batch)
        assert imu_buffer.size == 3
        expected_dt = np.array([0.0, 1.0, 1.0]) * 1e-9
        np.testing.assert_allclose(imu_buffer.buffer[~np.isnan(imu_buffer.buffer[:, 7]), 7], expected_dt)

    def test_iterate_last_batch(self, imu_buffer: ImuBuffer) -> None:
        """Test the iterate_last_batch method of the ImuInitializer."""
        accel_batch = np.array([[0, 0, 9.81], [0, 0, 9.81], [0, 0, 9.81]])
        gyro_batch = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        timestamp_batch = np.array([1.0, 2.0, 3.0])
        imu_buffer.add_batch(accel_batch, gyro_batch, timestamp_batch)
        assert imu_buffer.last_batch_slice == (0, 3)
        for accel, gyro, dt in imu_buffer.iterate_last_batch():
            np.testing.assert_allclose(accel, np.array([0, 0, 9.81]))
            np.testing.assert_allclose(gyro, np.array([0, 0, 0]))
            assert dt == pytest.approx(1.0 * 1e-9)

    def test_iterate_all_batch(self, imu_buffer: ImuBuffer) -> None:
        """Test the iterate_all_batch method of the ImuInitializer."""
        accel_batch = np.array([[0, 0, 9.81], [0, 0, 9.81], [0, 0, 9.81]])
        gyro_batch = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        timestamp_batch = np.array([1.0, 2.0, 3.0])
        imu_buffer.add_batch(accel_batch, gyro_batch, timestamp_batch)
        assert imu_buffer.size == 3
        for accel, gyro, dt in imu_buffer.iterate_full_buffer():
            np.testing.assert_allclose(accel, np.array([0, 0, 9.81]))
            np.testing.assert_allclose(gyro, np.array([0, 0, 0]))
            assert dt == pytest.approx(1.0 * 1e-9)

        imu_buffer.add_batch(accel_batch, gyro_batch, timestamp_batch)
        assert imu_buffer.size == 6
        for accel, gyro, dt in imu_buffer.iterate_full_buffer():
            np.testing.assert_allclose(accel, np.array([0, 0, 9.81]))
            np.testing.assert_allclose(gyro, np.array([0, 0, 0]))
            assert dt == pytest.approx(1.0 * 1e-9)
