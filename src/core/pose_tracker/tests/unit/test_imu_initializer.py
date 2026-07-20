import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from core.pose_tracker.inertial_integration import ImuBuffer, ImuSchema


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

    def test_get_last_batch_returns_view_for_last_added_rows(self, imu_buffer: ImuBuffer) -> None:
        """Test that get_last_batch returns a non-owning view over the latest batch."""
        first_accel = np.array([[0.0, 0.0, 9.81], [0.0, 0.1, 9.81]])
        first_gyro = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]])
        first_ts = np.array([1_000_000_000.0, 1_010_000_000.0])
        second_accel = np.array([[1.0, 0.0, 9.81], [2.0, 0.0, 9.81], [3.0, 0.0, 9.81]])
        second_gyro = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]])
        second_ts = np.array([1_020_000_000.0, 1_030_000_000.0, 1_040_000_000.0])

        imu_buffer.add_batch(first_accel, first_gyro, first_ts)
        imu_buffer.add_batch(second_accel, second_gyro, second_ts)

        last_batch = imu_buffer.get_last_batch()

        assert last_batch.rows.shape == (3, ImuSchema.count())
        assert np.shares_memory(last_batch.rows, imu_buffer.buffer)
        assert np.shares_memory(last_batch.timestamps(), imu_buffer.buffer)
        assert np.shares_memory(last_batch.accel(), imu_buffer.buffer)
        assert np.shares_memory(last_batch.gyro(), imu_buffer.buffer)
        assert np.shares_memory(last_batch.dt(), imu_buffer.buffer)
        np.testing.assert_allclose(last_batch.timestamps(), second_ts)
        np.testing.assert_allclose(last_batch.accel(), second_accel)
        np.testing.assert_allclose(last_batch.gyro(), second_gyro)
        np.testing.assert_allclose(last_batch.dt(), np.array([0.01, 0.01, 0.01]))

        last_batch.rows[0, ImuSchema.ACCEL_X] = 42.0
        assert imu_buffer.buffer[2, ImuSchema.ACCEL_X] == 42.0

    def test_get_full_buffer_returns_view_for_all_rows(self, imu_buffer: ImuBuffer) -> None:
        """Test that get_full_buffer returns all valid rows as an ImuBatch view."""
        first_accel = np.array([[0.0, 0.0, 9.81], [0.0, 0.1, 9.81]])
        first_gyro = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]])
        first_ts = np.array([1_000_000_000.0, 1_010_000_000.0])
        second_accel = np.array([[1.0, 0.0, 9.81]])
        second_gyro = np.array([[0.1, 0.0, 0.0]])
        second_ts = np.array([1_020_000_000.0])

        imu_buffer.add_batch(first_accel, first_gyro, first_ts)
        imu_buffer.add_batch(second_accel, second_gyro, second_ts)

        full_batch = imu_buffer.get_full_buffer()

        assert full_batch.rows.shape == (3, ImuSchema.count())
        assert np.shares_memory(full_batch.rows, imu_buffer.buffer)
        assert np.shares_memory(full_batch.timestamps(), imu_buffer.buffer)
        assert np.shares_memory(full_batch.accel(), imu_buffer.buffer)
        assert np.shares_memory(full_batch.gyro(), imu_buffer.buffer)
        assert np.shares_memory(full_batch.dt(), imu_buffer.buffer)
        np.testing.assert_allclose(full_batch.timestamps(), np.array([*first_ts, *second_ts]))
        np.testing.assert_allclose(full_batch.accel(), np.vstack((first_accel, second_accel)))
        np.testing.assert_allclose(full_batch.gyro(), np.vstack((first_gyro, second_gyro)))
        np.testing.assert_allclose(full_batch.dt(), np.array([0.0, 0.01, 0.01]))

    def test_imu_batch_iterate_skips_non_positive_dt(self, imu_buffer: ImuBuffer) -> None:
        """Test that ImuBatch.iterate yields only rows with positive dt."""
        accel_batch = np.array([[0.0, 0.0, 9.81], [1.0, 0.0, 9.81], [2.0, 0.0, 9.81]])
        gyro_batch = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
        timestamp_batch = np.array([1_000_000_000.0, 1_010_000_000.0, 1_020_000_000.0])
        imu_buffer.add_batch(accel_batch, gyro_batch, timestamp_batch)

        measurements = list(imu_buffer.get_last_batch().iterate())

        assert len(measurements) == 2
        for row_idx, (accel, gyro, dt) in enumerate(measurements, start=1):
            np.testing.assert_allclose(accel, accel_batch[row_idx])
            np.testing.assert_allclose(gyro, gyro_batch[row_idx])
            assert dt == pytest.approx(0.01)

    def test_imu_batch_gram_schmidt_returns_identity_for_nominal_gravity(self, imu_buffer: ImuBuffer) -> None:
        """Test Gram-Schmidt rotation for accel already aligned with +Z."""
        accel_batch = np.full((3, 3), np.array([0.0, 0.0, 9.81]))
        gyro_batch = np.zeros((3, 3))
        timestamp_batch = np.array([1_000_000_000.0, 1_010_000_000.0, 1_020_000_000.0])
        imu_buffer.add_batch(accel_batch, gyro_batch, timestamp_batch)

        rotation = imu_buffer.get_last_batch().gram_schmidt()

        np.testing.assert_allclose(rotation.as_matrix(), np.eye(3), atol=1e-12)

    def test_imu_batch_gram_schmidt_aligns_x_axis_accel_with_z(self, imu_buffer: ImuBuffer) -> None:
        """Test Gram-Schmidt remains valid when accel is parallel to the default reference axis."""
        accel_batch = np.full((3, 3), np.array([9.81, 0.0, 0.0]))
        gyro_batch = np.zeros((3, 3))
        timestamp_batch = np.array([1_000_000_000.0, 1_010_000_000.0, 1_020_000_000.0])
        imu_buffer.add_batch(accel_batch, gyro_batch, timestamp_batch)

        rotation = imu_buffer.get_last_batch().gram_schmidt()
        accel_direction = np.array([1.0, 0.0, 0.0])

        np.testing.assert_allclose(rotation.as_matrix() @ accel_direction, np.array([0.0, 0.0, 1.0]), atol=1e-12)
        assert np.linalg.det(rotation.as_matrix()) == pytest.approx(1.0)

    def test_empty_imu_batch_gram_schmidt_raises(self, imu_buffer: ImuBuffer) -> None:
        """Test Gram-Schmidt fails explicitly for an empty batch."""
        with pytest.raises(ValueError, match="No accel measurements"):
            imu_buffer.get_last_batch().gram_schmidt()

    def test_imu_batch_metrics(self, imu_buffer: ImuBuffer) -> None:
        """Test metrics computed from an ImuBatch."""
        accel_batch = np.array([[0.0, 0.0, 9.81], [0.0, 0.0, 9.81], [0.0, 0.0, 9.81]])
        gyro_batch = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        timestamp_batch = np.array([1_000_000_000.0, 1_010_000_000.0, 1_030_000_000.0])
        imu_buffer.add_batch(accel_batch, gyro_batch, timestamp_batch)

        metrics = imu_buffer.get_last_batch().metrics()

        assert metrics.sample_count == 3
        assert metrics.duration_sec == pytest.approx(0.03)
        np.testing.assert_allclose(metrics.gyro_mean, np.array([3.0, 0.0, 0.0]))
        np.testing.assert_allclose(metrics.gyro_std, np.std(gyro_batch, axis=0))
        assert metrics.gyro_norm_mean == pytest.approx(3.0)
        assert metrics.gyro_norm_std == pytest.approx(np.std(np.array([1.0, 3.0, 5.0])))
        np.testing.assert_allclose(metrics.accel_mean, np.array([0.0, 0.0, 9.81]))
        np.testing.assert_allclose(metrics.accel_std, np.zeros(3))
        assert metrics.accel_norm_mean == pytest.approx(9.81)
        assert metrics.accel_norm_std == pytest.approx(0.0)
        assert metrics.accel_direction_std_rad == pytest.approx(0.0)

    def test_empty_imu_batch_metrics(self, imu_buffer: ImuBuffer) -> None:
        """Test metrics for an empty ImuBatch."""
        metrics = imu_buffer.get_last_batch().metrics()

        assert metrics.sample_count == 0
        assert metrics.duration_sec == 0.0
        np.testing.assert_allclose(metrics.gyro_mean, np.zeros(3))
        np.testing.assert_allclose(metrics.gyro_std, np.zeros(3))
        assert metrics.gyro_norm_mean == 0.0
        assert metrics.gyro_norm_std == 0.0
        np.testing.assert_allclose(metrics.accel_mean, np.zeros(3))
        np.testing.assert_allclose(metrics.accel_std, np.zeros(3))
        assert metrics.accel_norm_mean == 0.0
        assert metrics.accel_norm_std == 0.0
        assert metrics.accel_direction_std_rad == 0.0

    def test_imu_batch_metrics_handles_degenerate_acceleration_directions(self, imu_buffer: ImuBuffer) -> None:
        """Test direction metrics stay finite for zero and cancelling accel directions."""
        accel_batch = np.array([[0.0, 0.0, 0.0], [9.81, 0.0, 0.0], [-9.81, 0.0, 0.0]])
        gyro_batch = np.zeros((3, 3))
        timestamp_batch = np.array([1_000_000_000.0, 1_010_000_000.0, 1_020_000_000.0])
        imu_buffer.add_batch(accel_batch, gyro_batch, timestamp_batch)

        metrics = imu_buffer.get_last_batch().metrics()

        assert metrics.accel_direction_std_rad == pytest.approx(np.pi)
