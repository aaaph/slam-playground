import numpy as np
import pytest

from core.pose_tracker.frame_to_frame_pnp_store import (
    FeatureDuplicateError,
    FrameToFramePnpStore,
    NotEnoughSlotsError,
    PnPMapSchema,
)


def make_feature_batch(feat_ids: np.ndarray, *, include_xyz: bool = True) -> np.ndarray:
    """Create a PnP feature batch for store tests."""
    feat_ids = np.asarray(feat_ids, dtype=np.float64)
    feature_batch = np.full((feat_ids.shape[0], PnPMapSchema.count()), np.nan, dtype=np.float64)
    feature_batch[:, PnPMapSchema.FEAT_ID] = feat_ids
    feature_batch[:, PnPMapSchema.LEFT_U] = feat_ids + 10.0
    feature_batch[:, PnPMapSchema.LEFT_V] = feat_ids + 20.0
    feature_batch[:, PnPMapSchema.RIGHT_U] = feat_ids + 30.0
    feature_batch[:, PnPMapSchema.RIGHT_V] = feat_ids + 40.0
    if include_xyz:
        feature_batch[:, PnPMapSchema.X] = feat_ids
        feature_batch[:, PnPMapSchema.Y] = feat_ids + 1.0
        feature_batch[:, PnPMapSchema.Z] = feat_ids + 2.0
    return feature_batch


class TestFrameToFramePnpStore:
    """Unit tests for the FrameToFramePnpStore class."""

    def test_allocate_slot(self):
        """Test the allocate_slot method."""
        tracker = FrameToFramePnpStore(map_capacity=10)
        slot = tracker._get_feature_slots(np.array([1]))  # noqa: SLF001
        np.testing.assert_array_equal(slot, np.array([0]))
        slot = tracker._get_feature_slots(np.array([1]))  # noqa: SLF001
        np.testing.assert_array_equal(slot, np.array([0]))
        slot = tracker._get_feature_slots(np.array([1, 5, 10, 15, 20]))  # noqa: SLF001
        np.testing.assert_array_equal(slot, np.array([0, 1, 2, 3, 4]))
        slot = tracker._get_feature_slots(np.array([20, 5, 1]))  # noqa: SLF001
        np.testing.assert_array_equal(slot, np.array([4, 1, 0]))

        with pytest.raises(FeatureDuplicateError):
            tracker._get_feature_slots(np.array([1, 1]))  # noqa: SLF001

        with pytest.raises(NotEnoughSlotsError):
            tracker._get_feature_slots(np.arange(100, 1000, dtype=np.int32))  # noqa: SLF001

    def test_add_features(self):
        """Test the add_features method."""
        store = FrameToFramePnpStore(map_capacity=10)
        feature_batch = make_feature_batch(np.arange(10))
        store.add_features(feature_batch)

        np.testing.assert_array_equal(store._map[0, :, :], feature_batch)  # noqa: SLF001

    def test_missing_id_must_be_cleared(self):
        """Test that missing IDs must be cleared."""
        store = FrameToFramePnpStore(map_capacity=10)
        store.add_features(make_feature_batch(np.arange(10)))
        store.finish_frame_and_advance()

        store.add_features(make_feature_batch(np.arange(5)))
        store.finish_frame_and_advance()

        assert set(store._feat_to_slot.keys()) == {0, 1, 2, 3, 4}  # noqa: SLF001

    def test_finish_frame_releases_missing_feature_slots_for_reuse(self):
        """Test that features missing in the current frame release their slots."""
        store = FrameToFramePnpStore(map_capacity=4)
        store.add_features(make_feature_batch(np.array([10, 20, 30], dtype=np.int32)))
        store.finish_frame_and_advance()

        store.add_features(make_feature_batch(np.array([10, 30], dtype=np.int32)))
        store.finish_frame_and_advance()

        assert store._feat_to_slot == {10: 0, 30: 2}  # noqa: SLF001
        assert store._free_slots == [1]  # noqa: SLF001

        store.add_features(make_feature_batch(np.array([40], dtype=np.int32)))

        assert store._feat_to_slot[40] == 1  # noqa: SLF001
        assert store._free_slots == []  # noqa: SLF001

    def test_finish_frame_keeps_observed_feature_without_xyz(self):
        """Test that observed image-only features are not pruned by missing XYZ."""
        store = FrameToFramePnpStore(map_capacity=4)
        store.add_features(make_feature_batch(np.array([10, 20], dtype=np.int32)))
        store.finish_frame_and_advance()

        image_only_features = make_feature_batch(np.array([10], dtype=np.int32), include_xyz=False)
        store.add_features(image_only_features)
        store.finish_frame_and_advance()

        assert 10 in store._feat_to_slot  # noqa: SLF001
        assert 20 not in store._feat_to_slot  # noqa: SLF001
        assert np.isnan(store._map[1, store._feat_to_slot[10], PnPMapSchema.X])  # noqa: SLF001
