import numpy as np
import pytest

from core.feature_tracker.feature import FeatureStatus
from core.feature_tracker.feature_tensor import FeatureTensor


class TestFeatureTensor:
    """Unit test for feature tensor."""

    def test_get_free_indexes_method(self):
        """Test that the free indexes are returned in the correct order."""
        tensor = FeatureTensor(feat_capacity=10, history_capacity=2)
        np.testing.assert_array_equal(tensor.get_free_indexes(2), np.array([0, 1]))
        np.testing.assert_array_equal(tensor.get_free_indexes(2), np.array([2, 3]))
        np.testing.assert_array_equal(tensor.get_free_indexes(2), np.array([4, 5]))
        np.testing.assert_array_equal(tensor.free_slots, np.array([9, 8, 7, 6]))

    def test_allocate_slots_method(self):
        """Test that the allocate slots method returns the correct indexes."""
        tensor = FeatureTensor(feat_capacity=10, history_capacity=2)
        slots = tensor.allocate_slots(np.array([1]))
        np.testing.assert_array_equal(slots, np.array([0]))

    def test_allocate_same_slots_method(self):
        """Test that the allocate slots method returns the correct indexes."""
        tensor = FeatureTensor(feat_capacity=10, history_capacity=2)
        tensor.add(1, 1, (0, 0), (1, 1), FeatureStatus.NEW)
        index = tensor._id_to_idx[1]  # noqa: SLF001
        np.testing.assert_array_equal(tensor.allocate_slots(np.array([1])), np.array([index]))

    def test_allocate_duplicated_feat_ids(self):
        """Test that the allocate slots method returns the correct indexes."""
        tensor = FeatureTensor(feat_capacity=10, history_capacity=2)
        with pytest.raises(ValueError, match="Duplicate feature IDs"):
            tensor.allocate_slots(np.array([1, 1]))

    def test_find_slots_by_feat_ids(self):
        """Test that the find slots by feat ids method returns the correct slots."""
        tensor = FeatureTensor(feat_capacity=10, history_capacity=2)
        timestamp = 1
        tensor.add(1, timestamp, (0, 0), (1, 1), FeatureStatus.NEW)
        tensor.add(2, timestamp, (0, 0), (1, 1), FeatureStatus.NEW)
        tensor.add(3, timestamp, (0, 0), (1, 1), FeatureStatus.NEW)
        tensor.add(4, timestamp, (0, 0), (1, 1), FeatureStatus.NEW)
        np.testing.assert_array_equal(tensor.find_slots(np.array([1, 2, 3, 4])), np.array([0, 1, 2, 3]))

    def test_find_slots_not_found(self):
        """Test that the find slots method returns the correct slots."""
        tensor = FeatureTensor(feat_capacity=10, history_capacity=2)
        with pytest.raises(ValueError, match="Feature with ID 1 not found"):
            tensor.find_slots(np.array([1]))

    def test_add_method_with_feat_deep(self):
        """Test that the feature tensor has a add_batch method."""
        tensor = FeatureTensor(feat_capacity=4, history_capacity=2)
        timestamp = 1
        tensor.add(1, timestamp, (0, 0), (1, 1), FeatureStatus.NEW)
        tensor.add(2, timestamp, (0, 0), None, FeatureStatus.NEW)
        tensor.add(3, timestamp, (0, 0), (1, 1), FeatureStatus.NEW)
        tensor.add(4, timestamp, (0, 0), (1, 1), FeatureStatus.NEW)
        with pytest.raises(ValueError, match="No free slots available"):
            tensor.add(5, timestamp, (0, 0), (1, 1), FeatureStatus.NEW)
        np.testing.assert_almost_equal(tensor.free_slots, np.array([]))

    def test_add_method_same_feat(self):
        """Test that the feature tensor handles the same feature correctly."""
        tensor = FeatureTensor(feat_capacity=4, history_capacity=2)
        timestamp = 1
        tensor.add(1, timestamp, (0, 0), (1, 1), FeatureStatus.NEW)
        tensor.add(1, timestamp, (0, 0), (1, 1), FeatureStatus.NEW)
        np.testing.assert_almost_equal(tensor.free_slots, np.array([3, 2, 1]))

    def test_add_old_timestamp(self):
        """Test that the feature tensor handles the old timestamp correctly."""
        tensor = FeatureTensor(feat_capacity=4, history_capacity=2)
        timestamp = 2
        tensor.add(1, timestamp, (0, 0), (1, 1), FeatureStatus.NEW)
        with pytest.raises(ValueError, match="Old timestamp"):
            tensor.add(1, timestamp - 1, (0, 0), (1, 1), FeatureStatus.NEW)

    def test_add_new_timestamp(self):
        """Test that the feature tensor handles the new timestamp correctly."""
        tensor = FeatureTensor(feat_capacity=4, history_capacity=2)
        tensor.add(1, 1, (0, 0), (1, 1), FeatureStatus.NEW)
        np.testing.assert_almost_equal(tensor.free_slots, np.array([3, 2, 1]))
        tensor.add(1, 3, (0, 0), (1, 1), FeatureStatus.NEW)
        np.testing.assert_almost_equal(tensor.free_slots, np.array([3, 2, 1]))
        tensor.add(1, 5, (0, 0), (1, 1), FeatureStatus.NEW)
        np.testing.assert_almost_equal(tensor.free_slots, np.array([3, 2, 1]))

    def test_batch_method(self):
        """Test that the feature tensor has a batch method."""
        tensor = FeatureTensor(feat_capacity=4, history_capacity=2)
        assert hasattr(tensor, "add_batch")
        assert callable(tensor.add_batch)
        timestamp = 1

        batch = np.full((4, 8), np.nan, dtype=np.float32)
        batch[0, :] = np.array([1, timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        batch[1, :] = np.array([2, timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        batch[2, :] = np.array([3, timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        batch[3, :] = np.array([4, timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)

        tensor.add_batch(timestamp, batch)
        np.testing.assert_almost_equal(tensor.free_slots, np.array([]))

        with pytest.raises(ValueError, match="Old timestamp"):
            tensor.add_batch(timestamp - 1, batch)

        with pytest.raises(ValueError, match="No free slots available"):
            tensor.add(5, timestamp, (0, 0), (1, 1), FeatureStatus.NEW)

    def test_update_state_method(self):
        """Test that the feature tensor has a update_state method."""
        tensor = FeatureTensor(feat_capacity=4, history_capacity=2)
        assert hasattr(tensor, "update_state")
        assert callable(tensor.update_state)
        timestamp = 1
        batch = np.full((4, 8), np.nan, dtype=np.float32)
        batch[0, :] = np.array([1, timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        batch[1, :] = np.array([2, timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        batch[2, :] = np.array([3, timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        batch[3, :] = np.array([4, timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        tensor.add_batch(timestamp, batch)
        tensor.update_state(np.array([1, 2, 3, 4]), FeatureStatus.TRACKED)
        np.testing.assert_almost_equal(
            tensor.current_data[tensor.find_slots(np.array([1, 2, 3, 4])), 6],
            np.array(
                [
                    FeatureStatus.TRACKED.value,
                    FeatureStatus.TRACKED.value,
                    FeatureStatus.TRACKED.value,
                    FeatureStatus.TRACKED.value,
                ]
            ),
        )

    def test_prev_data_method(self):
        """Test that the feature tensor has a prev_data method."""
        tensor = FeatureTensor(feat_capacity=4, history_capacity=2)
        prev_batch = np.full((4, 8), np.nan, dtype=np.float32)
        prev_timestamp = 1
        prev_batch[0, :] = np.array([1, prev_timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        prev_batch[1, :] = np.array([2, prev_timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        prev_batch[2, :] = np.array([3, prev_timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        prev_batch[3, :] = np.array([4, prev_timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        tensor.add_batch(prev_timestamp, prev_batch)

        next_batch = np.full((4, 8), np.nan, dtype=np.float32)
        next_timestamp = 2
        next_batch[0, :] = np.array([1, next_timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        next_batch[1, :] = np.array([2, next_timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        next_batch[2, :] = np.array([3, next_timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        next_batch[3, :] = np.array([4, next_timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        tensor.add_batch(next_timestamp, next_batch)

        prev = tensor.prev_data
        expected_prev = np.full((4, 8), np.nan, dtype=np.float32)
        expected_prev[:, 0] = prev_batch[:, 0]
        expected_prev[:, 1] = prev_timestamp
        expected_prev[:, 2:] = prev_batch[:, 2:]
        np.testing.assert_almost_equal(prev, expected_prev)

    def test_batch_add_with_new_ids(self):
        """
        Test that the on next batch update there is a issue with new ids.

        If we have a new id that is not in the previous batch, we should raise an error.
        """
        tensor = FeatureTensor(feat_capacity=4, history_capacity=2)
        timestamp = 1
        prev_batch = np.full((4, 8), np.nan, dtype=np.float32)
        prev_batch[0, :] = np.array([1, timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        prev_batch[1, :] = np.array([2, timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        prev_batch[2, :] = np.array([3, timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        prev_batch[3, :] = np.array([4, timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        tensor.add_batch(timestamp, prev_batch)
        np.testing.assert_almost_equal(tensor.free_slots, np.array([]))

        next_timestamp = 2
        next_batch = np.full((4, 8), np.nan, dtype=np.float32)
        next_batch[0, :] = np.array([1, next_timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        next_batch[1, :] = np.array([2, next_timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        next_batch[2, :] = np.array([3, next_timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        next_batch[3, :] = np.array([5, next_timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)

        with pytest.raises(ValueError, match="No free slots available"):
            tensor.add_batch(next_timestamp, next_batch)

    def test_tensor_pruning(self):
        """Test that the feature tensor could prune not actual features."""
        tensor = FeatureTensor(feat_capacity=4, history_capacity=2)
        first_timestamp = 1
        first_batch = np.full((4, 8), np.nan, dtype=np.float32)
        first_batch[0, :] = np.array([1, first_timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        first_batch[1, :] = np.array([2, first_timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        first_batch[2, :] = np.array([3, first_timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        first_batch[3, :] = np.array([4, first_timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        tensor.add_batch(first_timestamp, first_batch)

        second_timestamp = 2
        second_batch = np.full((3, 8), np.nan, dtype=np.float32)
        second_batch[0, :] = np.array([1, second_timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        second_batch[1, :] = np.array([2, second_timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        second_batch[2, :] = np.array([4, second_timestamp, 1, 0, 0, 1, 1, 0], dtype=np.float32)
        tensor.add_batch(second_timestamp, second_batch)

        third_timestamp = 3
        tensor.step(third_timestamp)
        assert tensor.exists(3) is False

    def test_get_slots_by_status_method(self):
        """Test that the feature tensor has a get_slots_by_status method."""
        tensor = FeatureTensor(feat_capacity=4, history_capacity=2)
        timestamp = 1
        batch = np.full((4, 8), np.nan, dtype=np.float32)
        batch[0, :] = np.array([1, timestamp, 1, 0, 0, 1, FeatureStatus.NEW.value, 0], dtype=np.float32)
        batch[1, :] = np.array([2, timestamp, 1, 0, 0, 1, FeatureStatus.TRACKED.value, 0], dtype=np.float32)
        batch[2, :] = np.array([3, timestamp, 1, 0, 0, 1, FeatureStatus.LOST.value, 0], dtype=np.float32)
        batch[3, :] = np.array([4, timestamp, 1, 0, 0, 1, FeatureStatus.NEW.value, 0], dtype=np.float32)
        tensor.add_batch(timestamp, batch)
        np.testing.assert_array_equal(tensor.get_slots_by_status(FeatureStatus.NEW), np.array([0, 3]))
        expected_tensor_data = np.full((2, 8), np.nan, dtype=np.float32)
        expected_tensor_data[0, :] = np.array(
            [1, timestamp, 1, 0, 0, 1, FeatureStatus.NEW.value, 0], dtype=np.float32
        )
        expected_tensor_data[1, :] = np.array(
            [4, timestamp, 1, 0, 0, 1, FeatureStatus.NEW.value, 0], dtype=np.float32
        )
        np.testing.assert_almost_equal(
            tensor.current_data[tensor.get_slots_by_status(FeatureStatus.NEW)], expected_tensor_data
        )

    def test_from_arrow_method(self):
        """Test that the feature tensor has a from_arrow method."""
        tensor = FeatureTensor(feat_capacity=4, history_capacity=2)
        timestamp = 1
        batch = np.full((4, 8), np.nan, dtype=np.float32)
        batch[0, :] = np.array([1, timestamp, 1, 0, 0, 1, FeatureStatus.NEW.value, 0], dtype=np.float32)
        batch[1, :] = np.array([2, timestamp, 1, 0, 0, 1, FeatureStatus.TRACKED.value, 0], dtype=np.float32)
        batch[2, :] = np.array([3, timestamp, 1, 0, 0, 1, FeatureStatus.LOST.value, 0], dtype=np.float32)
        batch[3, :] = np.array([4, timestamp, 1, 0, 0, 1, FeatureStatus.NEW.value, 0], dtype=np.float32)
        tensor.add_batch(timestamp, batch)
        arrow = tensor.as_arrow()
        tensor_from_arrow = FeatureTensor.from_arrow(arrow)
        assert tensor_from_arrow.feat_capacity == 4
        active_data = tensor_from_arrow.active_data
        np.testing.assert_almost_equal(active_data[:, 0], np.array([1, 2, 3, 4], dtype=np.float32))
        np.testing.assert_almost_equal(
            active_data[:, 1], np.array([timestamp, timestamp, timestamp, timestamp], dtype=np.float32)
        )
        np.testing.assert_almost_equal(active_data[:, 2], np.array([1, 1, 1, 1], dtype=np.float32))
        np.testing.assert_almost_equal(active_data[:, 3], np.array([0, 0, 0, 0], dtype=np.float32))
        np.testing.assert_almost_equal(active_data[:, 4], np.array([0, 0, 0, 0], dtype=np.float32))
        np.testing.assert_almost_equal(active_data[:, 5], np.array([1, 1, 1, 1], dtype=np.float32))
        np.testing.assert_almost_equal(
            active_data[:, 6],
            np.array(
                [
                    FeatureStatus.NEW.value,
                    FeatureStatus.TRACKED.value,
                    FeatureStatus.LOST.value,
                    FeatureStatus.NEW.value,
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_almost_equal(active_data[:, 7], np.array([0, 0, 0, 0], dtype=np.float32))

    def test_to_color_array_method(self):
        """Test that the feature tensor has a to_color_array method."""
        tensor = FeatureTensor(feat_capacity=4, history_capacity=2)
        timestamp = 1
        batch = np.full((4, 8), np.nan, dtype=np.float32)
        batch[0, :] = np.array([1, timestamp, 1, 0, 0, 1, FeatureStatus.NEW.value, 0], dtype=np.float32)
        batch[1, :] = np.array([2, timestamp, 1, 0, 0, 1, FeatureStatus.TRACKED.value, 0], dtype=np.float32)
        batch[2, :] = np.array([3, timestamp, 1, 0, 0, 1, FeatureStatus.LOST.value, 0], dtype=np.float32)
        batch[3, :] = np.array([4, timestamp, 1, 0, 0, 1, FeatureStatus.NEW.value, 0], dtype=np.float32)
        tensor.add_batch(timestamp, batch)
        color_array = FeatureTensor.to_color_array(tensor.current_data)
        np.testing.assert_array_equal(
            color_array, np.array([[0, 255, 0], [255, 0, 0], [128, 128, 128], [0, 255, 0]])
        )
