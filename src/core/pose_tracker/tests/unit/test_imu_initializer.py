import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from core.pose_tracker.inertial_integration import ImuInitializer


class TestImuInitializer:
    """Unit test for ImuInitializer."""

    @pytest.fixture
    def imu_initializer(self) -> ImuInitializer:
        """Create an ImuInitializer."""
        return ImuInitializer(capacity=1000)

    def test_add_batch(self, imu_initializer: ImuInitializer) -> None:
        """Test the add_batch method of the ImuInitializer."""
        accel_batch = np.array([[0, 0, 9.81], [0, 0, 9.81], [0, 0, 9.81]])
        gyro_batch = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        timestamp_batch = np.array([1.0, 2.0, 3.0])
        imu_initializer.add_batch(accel_batch, gyro_batch, timestamp_batch)
        assert imu_initializer.buffer is not None

    def test_create_initial_state(self, imu_initializer: ImuInitializer) -> None:
        """Test the create_initial_bias method of the ImuInitializer."""
        accel_batch = np.full((100, 3), np.array([0, 0, 9.81]))
        gyro_batch = np.full((100, 3), np.array([0, 0, 0]))
        timestamp_batch = np.linspace(1.0, 100.0, 100)
        imu_initializer.add_batch(accel_batch, gyro_batch, timestamp_batch)
        init_state = imu_initializer.create_initial_state()
        assert init_state is not None
        assert np.allclose(init_state.gyro_bias, np.array([0, 0, 0]))
        assert np.allclose(init_state.accel_bias, np.array([0, 0, 0]))
        assert np.allclose(init_state.rotation.as_quat(), Rotation.from_matrix(np.eye(3)).as_quat())
        assert np.allclose(init_state.gyro_noise, np.array([0, 0, 0]))
        assert np.allclose(init_state.accel_noise, np.array([0, 0, 0]))
