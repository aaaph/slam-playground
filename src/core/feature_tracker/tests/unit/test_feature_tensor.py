import numpy as np
import pytest
from numpy.typing import NDArray

from core.feature_tracker.feature_schema import FeatureLifecycle, FeatureSchema
from core.feature_tracker.feature_tensor import FeatureTensor


class TestFeatureTensor:
    """Unit test for feature tensor."""

    @staticmethod
    def feature_row(  # noqa: PLR0913
        feat_id: int,
        timestamp: float = 1.0,
        left_uv: tuple[float, float] = (1.0, 0.0),
        right_uv: tuple[float, float] | None = (0.0, 1.0),
        lifecycle: FeatureLifecycle = FeatureLifecycle.ACTIVE,
        age: float = 0.0,
        stereo_score: float = 0.0,
    ) -> NDArray[np.float32]:
        """Build a feature row using FeatureSchema indexes."""
        row = np.full(FeatureSchema.count(), np.nan, dtype=np.float32)
        row[FeatureSchema.FEAT_ID] = feat_id
        row[FeatureSchema.TIMESTAMP] = timestamp
        row[FeatureSchema.LEFT_U] = left_uv[0]
        row[FeatureSchema.LEFT_V] = left_uv[1]
        if right_uv is not None:
            row[FeatureSchema.RIGHT_U] = right_uv[0]
            row[FeatureSchema.RIGHT_V] = right_uv[1]
        row[FeatureSchema.LIFECYCLE] = lifecycle.value
        row[FeatureSchema.AGE] = age
        if hasattr(FeatureSchema, "STEREO_SCORE"):
            row[FeatureSchema.STEREO_SCORE] = stereo_score
        return row

    @pytest.fixture
    def batch(self) -> NDArray[np.float32]:
        """Create a batch of features."""
        return np.vstack([self.feature_row(feat_id) for feat_id in range(1, 5)]).astype(np.float32)

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
        tensor.add(1, 1, (0, 0), (1, 1), FeatureLifecycle.ACTIVE)
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
        tensor.add(1, timestamp, (0, 0), (1, 1), FeatureLifecycle.ACTIVE)
        tensor.add(2, timestamp, (0, 0), (1, 1), FeatureLifecycle.ACTIVE)
        tensor.add(3, timestamp, (0, 0), (1, 1), FeatureLifecycle.ACTIVE)
        tensor.add(4, timestamp, (0, 0), (1, 1), FeatureLifecycle.ACTIVE)
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
        tensor.add(1, timestamp, (0, 0), (1, 1), FeatureLifecycle.ACTIVE)
        tensor.add(2, timestamp, (0, 0), None, FeatureLifecycle.ACTIVE)
        tensor.add(3, timestamp, (0, 0), (1, 1), FeatureLifecycle.ACTIVE)
        tensor.add(4, timestamp, (0, 0), (1, 1), FeatureLifecycle.ACTIVE)
        with pytest.raises(ValueError, match="No free slots available"):
            tensor.add(5, timestamp, (0, 0), (1, 1), FeatureLifecycle.ACTIVE)
        np.testing.assert_almost_equal(tensor.free_slots, np.array([]))

    def test_add_method_same_feat(self):
        """Test that the feature tensor handles the same feature correctly."""
        tensor = FeatureTensor(feat_capacity=4, history_capacity=2)
        timestamp = 1
        tensor.add(1, timestamp, (0, 0), (1, 1), FeatureLifecycle.ACTIVE)
        tensor.add(1, timestamp, (0, 0), (1, 1), FeatureLifecycle.ACTIVE)
        np.testing.assert_almost_equal(tensor.free_slots, np.array([3, 2, 1]))

    def test_add_old_timestamp(self):
        """Test that the feature tensor handles the old timestamp correctly."""
        tensor = FeatureTensor(feat_capacity=4, history_capacity=2)
        timestamp = 2
        tensor.add(1, timestamp, (0, 0), (1, 1), FeatureLifecycle.ACTIVE)
        with pytest.raises(ValueError, match="Old timestamp"):
            tensor.add(1, timestamp - 1, (0, 0), (1, 1), FeatureLifecycle.ACTIVE)

    def test_add_new_timestamp(self):
        """Test that the feature tensor handles the new timestamp correctly."""
        tensor = FeatureTensor(feat_capacity=4, history_capacity=2)
        tensor.add(1, 1, (0, 0), (1, 1), FeatureLifecycle.ACTIVE)
        np.testing.assert_almost_equal(tensor.free_slots, np.array([3, 2, 1]))
        tensor.add(1, 3, (0, 0), (1, 1), FeatureLifecycle.ACTIVE)
        np.testing.assert_almost_equal(tensor.free_slots, np.array([3, 2, 1]))
        tensor.add(1, 5, (0, 0), (1, 1), FeatureLifecycle.ACTIVE)
        np.testing.assert_almost_equal(tensor.free_slots, np.array([3, 2, 1]))

    def test_batch_method(self, batch: NDArray[np.float32]):
        """Test that the feature tensor has a batch method."""
        tensor = FeatureTensor(feat_capacity=4, history_capacity=2)
        assert hasattr(tensor, "add_batch")
        assert callable(tensor.add_batch)
        timestamp = 1

        tensor.add_batch(timestamp, batch)
        np.testing.assert_almost_equal(tensor.free_slots, np.array([]))

        with pytest.raises(ValueError, match="Old timestamp"):
            tensor.add_batch(timestamp - 1, batch)

        with pytest.raises(ValueError, match="No free slots available"):
            tensor.add(5, timestamp, (0, 0), (1, 1), FeatureLifecycle.ACTIVE)

    def test_add_empty_batch(self):
        """Test that the feature tensor can add an empty batch."""
        tensor = FeatureTensor(feat_capacity=4, history_capacity=2)
        batch = np.empty((0, FeatureSchema.count()), dtype=np.float32)
        tensor.add_batch(1, batch)
        active_feat = tensor.active_frame
        assert active_feat.ndarray.shape[0] == 0
        assert active_feat.ndarray.shape[1] == FeatureSchema.count()

    def test_update_state_method(self, batch: NDArray[np.float32]):
        """Test that the feature tensor has a update_state method."""
        tensor = FeatureTensor(feat_capacity=4, history_capacity=2)
        assert hasattr(tensor, "update_state")
        assert callable(tensor.update_state)
        timestamp = 1
        batch[:, FeatureSchema.TIMESTAMP] = timestamp
        tensor.add_batch(timestamp, batch)
        tensor.update_state(np.array([1, 2, 3, 4]), FeatureLifecycle.ACTIVE)
        np.testing.assert_almost_equal(
            tensor.current_data[tensor.find_slots(np.array([1, 2, 3, 4])), 6],
            np.array(
                [
                    FeatureLifecycle.ACTIVE.value,
                    FeatureLifecycle.ACTIVE.value,
                    FeatureLifecycle.ACTIVE.value,
                    FeatureLifecycle.ACTIVE.value,
                ]
            ),
        )

    def test_prev_data_method(self, batch: NDArray[np.float32]):
        """Test that the feature tensor has a prev_data method."""
        tensor = FeatureTensor(feat_capacity=4, history_capacity=2)
        prev_batch = batch.copy()
        prev_timestamp = 1
        prev_batch[:, FeatureSchema.TIMESTAMP] = prev_timestamp
        tensor.add_batch(prev_timestamp, prev_batch)

        next_timestamp = 2
        next_batch = batch.copy()
        next_batch[:, FeatureSchema.TIMESTAMP] = next_timestamp
        tensor.add_batch(next_timestamp, next_batch)

        prev = tensor.prev_data
        expected_prev = np.full((4, FeatureSchema.count()), np.nan, dtype=np.float32)
        expected_prev[:, :] = prev_batch
        np.testing.assert_almost_equal(prev, expected_prev)

    def test_get_frame_by_timestamp_returns_frame_specific_slots(self):
        """Historical frame should use slots and count from its own timestamp."""
        tensor = FeatureTensor(feat_capacity=4, history_capacity=3)

        first_timestamp = 1
        first_batch = np.vstack([self.feature_row(feat_id, first_timestamp) for feat_id in (1, 2, 3, 4)])
        tensor.add_batch(first_timestamp, first_batch)

        second_timestamp = 2
        second_batch = np.vstack([self.feature_row(feat_id, second_timestamp) for feat_id in (1, 2, 4)])
        tensor.add_batch(second_timestamp, second_batch)
        tensor.step(3)

        first_frame = tensor.get_frame_by_timestamp(first_timestamp)
        second_frame = tensor.get_frame_by_timestamp(second_timestamp)

        np.testing.assert_array_equal(first_frame.active_indeces, np.array([0, 1, 2, 3], dtype=np.int32))
        np.testing.assert_array_equal(first_frame.ids, np.array([1, 2, 3, 4], dtype=np.int32))
        assert first_frame.count() == 4

        np.testing.assert_array_equal(second_frame.active_indeces, np.array([0, 1, 3], dtype=np.int32))
        np.testing.assert_array_equal(second_frame.ids, np.array([1, 2, 4], dtype=np.int32))
        assert second_frame.count() == 3

    def test_feature_row_adapts_to_feature_schema(self):
        """Feature row helper should follow FeatureSchema indexes and width."""
        row = self.feature_row(
            7,
            timestamp=42,
            left_uv=(10, 11),
            right_uv=None,
            lifecycle=FeatureLifecycle.LOST,
            age=3,
            stereo_score=5,
        )

        assert row.shape == (FeatureSchema.count(),)
        assert row[FeatureSchema.FEAT_ID] == 7
        assert row[FeatureSchema.TIMESTAMP] == 42
        np.testing.assert_array_equal(row[FeatureSchema.LEFT_U : FeatureSchema.LEFT_V + 1], np.array([10, 11]))
        assert np.isnan(row[FeatureSchema.RIGHT_U])
        assert np.isnan(row[FeatureSchema.RIGHT_V])
        assert row[FeatureSchema.LIFECYCLE] == FeatureLifecycle.LOST.value
        assert row[FeatureSchema.AGE] == 3
        if hasattr(FeatureSchema, "STEREO_SCORE"):
            assert row[FeatureSchema.STEREO_SCORE] == 5

    def test_timestamp_index_raises_value_error_for_missing_timestamp(self):
        """Missing timestamp should raise a stable ValueError."""
        tensor = FeatureTensor(feat_capacity=4, history_capacity=2)
        with pytest.raises(ValueError, match="Timestamp 42 not found"):
            tensor.timestamp_index(42)

    def test_batch_add_with_new_ids(self, batch: NDArray[np.float32]):
        """
        Test that the on next batch update there is a issue with new ids.

        If we have a new id that is not in the previous batch, we should raise an error.
        """
        tensor = FeatureTensor(feat_capacity=4, history_capacity=2)
        timestamp = 1
        prev_batch = batch.copy()
        prev_batch[:, FeatureSchema.TIMESTAMP] = timestamp
        tensor.add_batch(timestamp, prev_batch)
        np.testing.assert_almost_equal(tensor.free_slots, np.array([]))

        next_timestamp = 2
        next_batch = batch.copy()
        next_batch[:, FeatureSchema.TIMESTAMP] = next_timestamp
        next_batch[0, FeatureSchema.FEAT_ID] = 5
        with pytest.raises(ValueError, match="No free slots available"):
            tensor.add_batch(next_timestamp, next_batch)

    def test_tensor_pruning(self):
        """Test that the feature tensor could prune not actual features."""
        tensor = FeatureTensor(feat_capacity=4, history_capacity=2)
        first_timestamp = 1
        first_batch = np.vstack([self.feature_row(feat_id, first_timestamp) for feat_id in (1, 2, 3, 4)])
        tensor.add_batch(first_timestamp, first_batch)

        second_timestamp = 2
        second_batch = np.vstack([self.feature_row(feat_id, second_timestamp) for feat_id in (1, 2, 4)])
        tensor.add_batch(second_timestamp, second_batch)

        third_timestamp = 3
        tensor.step(third_timestamp)
        assert tensor.exists(3) is False

    def test_get_slots_by_status_method(self):
        """Test that the feature tensor has a get_slots_by_status method."""
        tensor = FeatureTensor(feat_capacity=4, history_capacity=2)
        timestamp = 1
        batch = np.vstack(
            [
                self.feature_row(1, timestamp),
                self.feature_row(2, timestamp, lifecycle=FeatureLifecycle.LOST),
                self.feature_row(3, timestamp, lifecycle=FeatureLifecycle.LOST),
                self.feature_row(4, timestamp),
            ]
        )
        tensor.add_batch(timestamp, batch)
        np.testing.assert_array_equal(tensor.get_slots_by_status(FeatureLifecycle.ACTIVE), np.array([0, 3]))
        expected_tensor_data = np.vstack(
            [
                self.feature_row(1, timestamp),
                self.feature_row(4, timestamp),
            ]
        )
        np.testing.assert_almost_equal(
            tensor.current_data[tensor.get_slots_by_status(FeatureLifecycle.ACTIVE)], expected_tensor_data
        )

    def test_from_arrow_method(self):
        """Test that the feature tensor has a from_arrow method."""
        tensor = FeatureTensor(feat_capacity=4, history_capacity=2)
        timestamp = 1
        batch = np.vstack(
            [
                self.feature_row(1, timestamp),
                self.feature_row(2, timestamp),
                self.feature_row(3, timestamp, lifecycle=FeatureLifecycle.LOST),
                self.feature_row(4, timestamp),
            ]
        )
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
                    FeatureLifecycle.ACTIVE.value,
                    FeatureLifecycle.ACTIVE.value,
                    FeatureLifecycle.LOST.value,
                    FeatureLifecycle.ACTIVE.value,
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_almost_equal(active_data[:, 7], np.array([0, 0, 0, 0], dtype=np.float32))

    def test_to_color_array_method(self):
        """Test that the feature tensor has a to_color_array method."""
        tensor = FeatureTensor(feat_capacity=4, history_capacity=2)
        timestamp = 1
        batch = np.vstack(
            [
                self.feature_row(1, timestamp),
                self.feature_row(2, timestamp),
                self.feature_row(3, timestamp, lifecycle=FeatureLifecycle.LOST),
                self.feature_row(4, timestamp),
            ]
        )
        tensor.add_batch(timestamp, batch)
        color_array = FeatureTensor.to_color_array(tensor.current_data)
        np.testing.assert_array_equal(
            color_array, np.array([[0, 255, 0], [0, 255, 0], [128, 128, 128], [0, 255, 0]])
        )
