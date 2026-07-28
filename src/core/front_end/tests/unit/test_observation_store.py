import numpy as np
import pytest

from core.front_end.observation_store import (
    CompressPolicy,
    ObservationSchema,
    ObservationStore,
    ReadyObservationCriteria,
)


class TestObservationStore:
    """Unit tests for observation store slot allocation."""

    @staticmethod
    def make_observation(feat_id: int, left_u: float) -> np.ndarray:
        """Build a single observation row."""
        observation = np.full((1, ObservationSchema.size()), np.nan, dtype=np.float64)
        observation[0, ObservationSchema.FEAT_ID] = feat_id
        observation[0, ObservationSchema.LEFT_U] = left_u
        observation[0, ObservationSchema.LEFT_V] = 0.0
        observation[0, ObservationSchema.RIGHT_U] = left_u - 1.0
        observation[0, ObservationSchema.RIGHT_V] = 0.0
        return observation

    def test_get_feature_slots_allocates_new_features(self) -> None:
        """New feature IDs should allocate fresh dense slots."""
        store = ObservationStore(capacity=4)

        slots = store._get_feature_slots(np.array([10, 20], dtype=np.int32))  # noqa: SLF001

        np.testing.assert_array_equal(slots, np.array([0, 1], dtype=np.int32))
        assert store._feat_ids_to_slot == {10: 0, 20: 1}  # noqa: SLF001
        assert store._feat_ids_to_history_size == {10: 0, 20: 0}  # noqa: SLF001
        np.testing.assert_array_equal(store._slot_to_feat[:2], np.array([10, 20], dtype=np.int32))  # noqa: SLF001

    def test_get_feature_slots_reuses_existing_and_allocates_new_features(self) -> None:
        """Existing feature IDs should keep their slots while new IDs allocate fresh slots."""
        store = ObservationStore(capacity=4)
        store._get_feature_slots(np.array([10, 20], dtype=np.int32))  # noqa: SLF001

        slots = store._get_feature_slots(np.array([20, 30], dtype=np.int32))  # noqa: SLF001

        np.testing.assert_array_equal(slots, np.array([1, 2], dtype=np.int32))
        assert store._feat_ids_to_slot == {10: 0, 20: 1, 30: 2}  # noqa: SLF001
        np.testing.assert_array_equal(store._slot_to_feat[:3], np.array([10, 20, 30], dtype=np.int32))  # noqa: SLF001

    def test_get_feature_slots_reuses_free_slots(self) -> None:
        """Released slots should be reused before growing next_slot."""
        store = ObservationStore(capacity=4)
        store._get_feature_slots(np.array([10, 20], dtype=np.int32))  # noqa: SLF001
        store.remove_features(np.array([10], dtype=np.int32))

        slots = store._get_feature_slots(np.array([30], dtype=np.int32))  # noqa: SLF001

        np.testing.assert_array_equal(slots, np.array([0], dtype=np.int32))
        assert store._feat_ids_to_slot[30] == 0  # noqa: SLF001
        assert store._slot_to_feat[0] == 30  # noqa: SLF001
        assert store._next_slot == 2  # noqa: SLF001

    def test_get_feature_slots_rejects_duplicate_feature_ids(self) -> None:
        """A single allocation batch should not contain duplicate feature IDs."""
        store = ObservationStore(capacity=4)

        with pytest.raises(ValueError, match="unique"):
            store._get_feature_slots(np.array([10, 10], dtype=np.int32))  # noqa: SLF001

    def test_remove_features_releases_slot_and_clears_history(self) -> None:
        """Removed features should release mappings and clear stored observations."""
        store = ObservationStore(capacity=4, history_size=2)
        slots = store._get_feature_slots(np.array([10, 20], dtype=np.int32))  # noqa: SLF001
        removed_slot = int(slots[0])
        kept_slot = int(slots[1])
        store._observations[removed_slot, :, ObservationSchema.FEAT_ID] = 10  # noqa: SLF001
        store._observations[kept_slot, :, ObservationSchema.FEAT_ID] = 20  # noqa: SLF001
        store._feat_ids_to_history_size[10] = 2  # noqa: SLF001
        store._feat_ids_to_history_size[20] = 2  # noqa: SLF001

        store.remove_features(np.array([10], dtype=np.int32))

        assert 10 not in store._feat_ids_to_slot  # noqa: SLF001
        assert 10 not in store._feat_ids_to_history_size  # noqa: SLF001
        assert store._slot_to_feat[removed_slot] == -1  # noqa: SLF001
        assert removed_slot in store._free_slots  # noqa: SLF001
        assert np.all(np.isnan(store._observations[removed_slot]))  # noqa: SLF001
        assert store._feat_ids_to_slot[20] == kept_slot  # noqa: SLF001
        assert store._feat_ids_to_history_size[20] == 2  # noqa: SLF001

    def test_remove_features_ignores_unknown_feature_ids(self) -> None:
        """Removing an unknown feature ID should keep existing slots unchanged."""
        store = ObservationStore(capacity=4)
        store._get_feature_slots(np.array([10], dtype=np.int32))  # noqa: SLF001

        store.remove_features(np.array([99], dtype=np.int32))

        assert store._feat_ids_to_slot == {10: 0}  # noqa: SLF001
        assert store._feat_ids_to_history_size == {10: 0}  # noqa: SLF001
        np.testing.assert_array_equal(store._slot_to_feat[:1], np.array([10], dtype=np.int32))  # noqa: SLF001
        assert store._free_slots == []  # noqa: SLF001

    def test_get_slots_by_criteria_returns_ready_feature_slots(self) -> None:
        """Readiness criteria should return store slots, not feature IDs."""
        store = ObservationStore(capacity=4, history_size=5)
        for ready_left_u, pending_left_u in [(0.0, 0.0), (2.0, 0.2), (3.0, 0.4)]:
            store.add_observations(
                np.vstack(
                    (
                        self.make_observation(10, ready_left_u),
                        self.make_observation(20, pending_left_u),
                    )
                )
            )

        criteria = ReadyObservationCriteria(
            min_history_size=3,
            min_pixel_displacement=1.0,
            min_displacement_observations=2,
        )

        slots = store.get_slots_by_criteria(criteria)

        np.testing.assert_array_equal(slots, np.array([0], dtype=np.int32))
        np.testing.assert_array_equal(store._slot_to_feat[slots], np.array([10], dtype=np.int32))  # noqa: SLF001

    def test_get_ready_feature_slice_returns_fixed_depth_histories(self) -> None:
        """Ready feature slice should preserve full store history depth."""
        store = ObservationStore(capacity=4, history_size=5)
        for left_u in [0.0, 2.0, 3.0]:
            store.add_observations(self.make_observation(10, left_u))
        for left_u in [0.0, 2.0, 3.0, 4.0, 5.0]:
            store.add_observations(self.make_observation(30, left_u))

        criteria = ReadyObservationCriteria(
            min_history_size=3,
            min_pixel_displacement=1.0,
            min_displacement_observations=2,
        )

        feat_ids, histories, history_mask = store.get_ready_feature_slice(criteria)

        np.testing.assert_array_equal(feat_ids, np.array([10, 30], dtype=np.int32))
        assert histories.shape == (2, 5, ObservationSchema.size())
        assert history_mask.shape == (2, 5)
        np.testing.assert_allclose(histories[0, :3, ObservationSchema.LEFT_U], np.array([0.0, 2.0, 3.0]))
        assert np.all(np.isnan(histories[0, 3:]))
        np.testing.assert_array_equal(
            history_mask,
            np.array(
                [
                    [True, True, True, False, False],
                    [True, True, True, True, True],
                ],
                dtype=np.bool_,
            ),
        )

    def test_add_observations_adds_observations_to_the_observation_store(self) -> None:
        """Adding observations should add them to the observation store."""
        store = ObservationStore(capacity=4, history_size=1)
        observations = np.full((4, ObservationSchema.size()), np.nan, dtype=np.float64)
        observations[:, ObservationSchema.FEAT_ID] = np.array([10, 20, 30, 40], dtype=np.int32)
        store.add_observations(observations)

        assert store._feat_ids_to_slot == {10: 0, 20: 1, 30: 2, 40: 3}  # noqa: SLF001
        assert store._feat_ids_to_history_size == {10: 1, 20: 1, 30: 1, 40: 1}  # noqa: SLF001
        expected_observations = observations.copy()
        expected_observations[:, ObservationSchema.ANCHOR_PIXEL_DISPLACEMENT] = 0.0
        np.testing.assert_array_equal(store._observations[0, 0], expected_observations[0])  # noqa: SLF001
        np.testing.assert_array_equal(store._observations[1, 0], expected_observations[1])  # noqa: SLF001

    def test_add_observations_uniform_recent_compresses_full_history_and_appends_current_observation(
        self,
    ) -> None:
        """Uniform-recent compression should keep anchor, sampled history, and current observation."""
        store = ObservationStore(
            capacity=1,
            history_size=5,
            compressed_history_size=3,
            compress_policy=CompressPolicy.UNIFORM_RECENT,
        )
        for left_u in range(6):
            store.add_observations(self.make_observation(10, float(left_u)))

        history = store.get_feat_history(10)

        assert store._feat_ids_to_history_size[10] == 4  # noqa: SLF001
        np.testing.assert_allclose(history[:, ObservationSchema.LEFT_U], np.array([0.0, 2.0, 4.0, 5.0]))
        np.testing.assert_allclose(
            history[:, ObservationSchema.ANCHOR_PIXEL_DISPLACEMENT],
            np.array([0.0, 2.0, 4.0, 5.0]),
        )
        assert np.all(np.isnan(store._observations[0, 4]))  # noqa: SLF001

    def test_add_observations_uniform_recent_recompresses_summary_and_recent_tail(self) -> None:
        """Repeated uniform-recent compression should keep anchor and bias toward the recent tail."""
        store = ObservationStore(
            capacity=1,
            history_size=5,
            compressed_history_size=3,
            compress_policy=CompressPolicy.UNIFORM_RECENT,
        )
        for left_u in range(8):
            store.add_observations(self.make_observation(10, float(left_u)))

        history = store.get_feat_history(10)

        assert store._feat_ids_to_history_size[10] == 4  # noqa: SLF001
        np.testing.assert_allclose(history[:, ObservationSchema.LEFT_U], np.array([0.0, 4.0, 6.0, 7.0]))
        np.testing.assert_allclose(
            history[:, ObservationSchema.ANCHOR_PIXEL_DISPLACEMENT],
            np.array([0.0, 4.0, 6.0, 7.0]),
        )
        assert np.all(np.isnan(store._observations[0, 4]))  # noqa: SLF001

    def test_add_observations_top_displacement_keeps_anchor_top_displacements_latest_and_current(
        self,
    ) -> None:
        """Top-displacement compression should keep motion evidence that uniform sampling can miss."""
        store = ObservationStore(
            capacity=1,
            history_size=6,
            compressed_history_size=5,
            compress_policy=CompressPolicy.TOP_DISPLACEMENT,
        )
        for left_u in [0.0, 1.0, 10.0, 2.0, 9.0, 3.0, 4.0]:
            store.add_observations(self.make_observation(10, left_u))

        history = store.get_feat_history(10)

        assert store._feat_ids_to_history_size[10] == 6  # noqa: SLF001
        np.testing.assert_allclose(
            history[:, ObservationSchema.LEFT_U],
            np.array([0.0, 10.0, 2.0, 9.0, 3.0, 4.0]),
        )
        np.testing.assert_allclose(
            history[:, ObservationSchema.ANCHOR_PIXEL_DISPLACEMENT],
            np.array([0.0, 10.0, 2.0, 9.0, 3.0, 4.0]),
        )

    def test_add_observations_top_displacement_recompresses_with_recent_tail(
        self,
    ) -> None:
        """Repeated top-displacement compression should keep strongest old evidence and latest."""
        store = ObservationStore(
            capacity=1,
            history_size=6,
            compressed_history_size=5,
            compress_policy=CompressPolicy.TOP_DISPLACEMENT,
        )
        for left_u in [0.0, 1.0, 10.0, 2.0, 9.0, 3.0, 4.0, 5.0]:
            store.add_observations(self.make_observation(10, left_u))

        history = store.get_feat_history(10)

        assert store._feat_ids_to_history_size[10] == 6  # noqa: SLF001
        np.testing.assert_allclose(
            history[:, ObservationSchema.LEFT_U],
            np.array([0.0, 10.0, 9.0, 3.0, 4.0, 5.0]),
        )
        np.testing.assert_allclose(
            history[:, ObservationSchema.ANCHOR_PIXEL_DISPLACEMENT],
            np.array([0.0, 10.0, 9.0, 3.0, 4.0, 5.0]),
        )
