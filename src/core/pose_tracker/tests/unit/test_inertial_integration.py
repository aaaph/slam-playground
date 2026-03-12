import numpy as np
import pytest
from pytest_mock import MockerFixture

from core.pose_tracker.inertial_integration import InertialIntegration


class TestInertialIntegration:
    """Unit test for InertialIntegration."""

    def test_init(self) -> None:
        """Test the initialization of the InertialIntegration."""
        inertial_integration = InertialIntegration(0)
        assert inertial_integration.nav_state is not None

    def test_integrate_should_ignore_older_timestamps(self) -> None:
        """Test that the integration of the InertialIntegration ignores older timestamps."""
        inertial_integration = InertialIntegration(1000)
        accel_batch = np.array([[0, 0, 9.81], [0, 0, 9.81], [0, 0, 9.81]])
        gyro_batch = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        timestamp_batch = np.array([1.0, 2.0, 1001.0])
        count = inertial_integration.integrate_batch(accel_batch, gyro_batch, timestamp_batch)
        assert count == 1

    def test_integrate_should_calculate_dt(self, mocker: MockerFixture) -> None:
        """
        Test that the integration of the InertialIntegration calculates the dt correctly.

        - Should map the batch of timestamps to the batch of dt
        - The first element of the batch should be calculated using a self.timestamp
        - The last element of the batch should be pasted into the self.timestamp after the integration
        - The elements after the first and last should be calculated using the previous element
        """
        inertial_integration = InertialIntegration(1000)
        mock_integrate = mocker.patch.object(inertial_integration, "_integrate")
        accel_batch = np.array(
            [
                [0, 0, 9.81],
                [0, 0, 9.81],
                [0, 0, 9.81],
                [0, 0, 9.81],
                [0, 0, 9.81],
                [0, 0, 9.81],
                [0, 0, 9.81],
                [0, 0, 9.81],
                [0, 0, 9.81],
                [0, 0, 9.81],
            ]
        )
        gyro_batch = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        timestamp_batch = np.array(
            [1001.0, 1002.0, 1003.0, 1004.0, 1005.0, 1006.0, 1007.0, 1008.0, 1009.0, 1010.0]
        )
        count = inertial_integration.integrate_batch(accel_batch, gyro_batch, timestamp_batch)
        assert count == 10
        # assert that method was called with the correct arguments
        assert mock_integrate.call_count == 10
        _, kwargs = mock_integrate.call_args_list[0]
        np.testing.assert_array_equal(kwargs["accel"], accel_batch[0])
        np.testing.assert_array_equal(kwargs["gyro"], gyro_batch[0])
        assert kwargs["dt"] == pytest.approx(1.0 / 1e9)

        assert inertial_integration.timestamp == 1010.0

    def test_integrate_and_predict(self, mocker: MockerFixture) -> None:
        """Test that the integration and prediction of the InertialIntegration works correctly."""
        inertial_integration = InertialIntegration(1000)
        accel_batch = np.array([[0, 0, -9.81], [0, 0, -9.81], [0, 0, -9.81]])
        gyro_batch = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        timestamp_batch = np.array([1001.0, 1002.0, 1003.0])
        new_state = inertial_integration.integrate_and_predict(accel_batch, gyro_batch, timestamp_batch)
        assert inertial_integration.timestamp == 1003.0
        assert new_state is not None
        assert np.allclose(new_state.pose().rotation().matrix(), np.eye(3), atol=1e-6)
        assert np.allclose(new_state.pose().translation(), np.array([0, 0, 0]), atol=1e-6)
        assert np.allclose(new_state.velocity(), np.array([0, 0, 0]), atol=1e-6)
