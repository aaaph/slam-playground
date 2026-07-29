import numpy as np
import pytest

from core.front_end.observation_store import (
    CompressPolicy,
    ObservationSchema,
    ObservationStore,
    ReadyObservationCriteria,
    SelectPolicy,
)

K_INV = np.eye(3, dtype=np.float64)


class TestObservationStore:
    """Unit tests for observation store slot allocation."""

    @staticmethod
    def make_observation(
        feat_id: int,
        left_u: float,
        left_v: float = 0.0,
        world_from_cam0: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build a single observation row."""
        observation = np.full((1, ObservationSchema.size()), np.nan, dtype=np.float64)
        observation[0, ObservationSchema.FEAT_ID] = feat_id
        observation[0, ObservationSchema.LEFT_U] = left_u
        observation[0, ObservationSchema.LEFT_V] = left_v
        observation[0, ObservationSchema.RIGHT_U] = left_u - 1.0
        observation[0, ObservationSchema.RIGHT_V] = left_v
        pose_matrix = np.eye(4, dtype=np.float64) if world_from_cam0 is None else world_from_cam0
        observation[0, ObservationSchema.CAM0_MATRIX] = pose_matrix.reshape(-1)
        return observation

    def test_get_feature_slots_allocates_new_features(self) -> None:
        """New feature IDs should allocate fresh dense slots."""
        store = ObservationStore(k_inv=K_INV, capacity=4)

        slots = store._get_feature_slots(np.array([10, 20], dtype=np.int32))  # noqa: SLF001

        np.testing.assert_array_equal(slots, np.array([0, 1], dtype=np.int32))
        assert store._feat_ids_to_slot == {10: 0, 20: 1}  # noqa: SLF001
        np.testing.assert_array_equal(store._history_sizes[:2], np.array([0, 0], dtype=np.int32))  # noqa: SLF001
        np.testing.assert_array_equal(store._slot_to_feat[:2], np.array([10, 20], dtype=np.int32))  # noqa: SLF001

    def test_get_feature_slots_reuses_existing_and_allocates_new_features(self) -> None:
        """Existing feature IDs should keep their slots while new IDs allocate fresh slots."""
        store = ObservationStore(k_inv=K_INV, capacity=4)
        store._get_feature_slots(np.array([10, 20], dtype=np.int32))  # noqa: SLF001

        slots = store._get_feature_slots(np.array([20, 30], dtype=np.int32))  # noqa: SLF001

        np.testing.assert_array_equal(slots, np.array([1, 2], dtype=np.int32))
        assert store._feat_ids_to_slot == {10: 0, 20: 1, 30: 2}  # noqa: SLF001
        np.testing.assert_array_equal(store._slot_to_feat[:3], np.array([10, 20, 30], dtype=np.int32))  # noqa: SLF001

    def test_get_feature_slots_reuses_free_slots(self) -> None:
        """Released slots should be reused before growing next_slot."""
        store = ObservationStore(k_inv=K_INV, capacity=4)
        store._get_feature_slots(np.array([10, 20], dtype=np.int32))  # noqa: SLF001
        store.remove_features(np.array([10], dtype=np.int32))

        slots = store._get_feature_slots(np.array([30], dtype=np.int32))  # noqa: SLF001

        np.testing.assert_array_equal(slots, np.array([0], dtype=np.int32))
        assert store._feat_ids_to_slot[30] == 0  # noqa: SLF001
        assert store._slot_to_feat[0] == 30  # noqa: SLF001
        assert store._next_slot == 2  # noqa: SLF001

    def test_get_feature_slots_rejects_duplicate_feature_ids(self) -> None:
        """A single allocation batch should not contain duplicate feature IDs."""
        store = ObservationStore(k_inv=K_INV, capacity=4)

        with pytest.raises(ValueError, match="unique"):
            store._get_feature_slots(np.array([10, 10], dtype=np.int32))  # noqa: SLF001

    def test_remove_features_releases_slot_and_clears_history(self) -> None:
        """Removed features should release mappings and clear stored observations."""
        store = ObservationStore(k_inv=K_INV, capacity=4, history_size=2)
        slots = store._get_feature_slots(np.array([10, 20], dtype=np.int32))  # noqa: SLF001
        removed_slot = int(slots[0])
        kept_slot = int(slots[1])
        store._observations[removed_slot, :, ObservationSchema.FEAT_ID] = 10  # noqa: SLF001
        store._observations[kept_slot, :, ObservationSchema.FEAT_ID] = 20  # noqa: SLF001
        store._history_sizes[removed_slot] = 2  # noqa: SLF001
        store._history_sizes[kept_slot] = 2  # noqa: SLF001

        store.remove_features(np.array([10], dtype=np.int32))

        assert 10 not in store._feat_ids_to_slot  # noqa: SLF001
        assert store._slot_to_feat[removed_slot] == -1  # noqa: SLF001
        assert store._history_sizes[removed_slot] == 0  # noqa: SLF001
        assert removed_slot in store._free_slots  # noqa: SLF001
        assert np.all(np.isnan(store._observations[removed_slot]))  # noqa: SLF001
        assert store._feat_ids_to_slot[20] == kept_slot  # noqa: SLF001
        assert store._history_sizes[kept_slot] == 2  # noqa: SLF001

    def test_remove_features_ignores_unknown_feature_ids(self) -> None:
        """Removing an unknown feature ID should keep existing slots unchanged."""
        store = ObservationStore(k_inv=K_INV, capacity=4)
        store._get_feature_slots(np.array([10], dtype=np.int32))  # noqa: SLF001

        store.remove_features(np.array([99], dtype=np.int32))

        assert store._feat_ids_to_slot == {10: 0}  # noqa: SLF001
        assert store._history_sizes[0] == 0  # noqa: SLF001
        np.testing.assert_array_equal(store._slot_to_feat[:1], np.array([10], dtype=np.int32))  # noqa: SLF001
        assert store._free_slots == []  # noqa: SLF001

    def test_get_slots_by_criteria_returns_ready_feature_slots(self) -> None:
        """Readiness criteria should return store slots, not feature IDs."""
        store = ObservationStore(k_inv=K_INV, capacity=4, history_size=5)
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
            min_parallax_rad=1.0,
            min_parallax_observations=2,
        )

        slots, history_versions, feat_ids = store.ready_slots(criteria, np.array([0, 1], dtype=np.int32))
        pending_only_slots, pending_only_history_versions, pending_only_feat_ids = store.ready_slots(
            criteria, candidate_slots=np.array([1], dtype=np.int32)
        )

        np.testing.assert_array_equal(slots, np.array([0], dtype=np.int32))
        np.testing.assert_array_equal(history_versions, np.array([3], dtype=np.int32))
        np.testing.assert_array_equal(feat_ids, np.array([10], dtype=np.int32))
        np.testing.assert_array_equal(pending_only_slots, np.empty((0,), dtype=np.int32))
        np.testing.assert_array_equal(pending_only_history_versions, np.empty((0,), dtype=np.int32))
        np.testing.assert_array_equal(pending_only_feat_ids, np.empty((0,), dtype=np.int32))
        np.testing.assert_array_equal(
            store.get_slots_by_criteria(criteria, np.array([0, 1], dtype=np.int32)),
            np.array([0], dtype=np.int32),
        )

    def test_add_observations_returns_used_slots_and_history_slots(self) -> None:
        """Adding observations should expose touched store slots and written history slots."""
        store = ObservationStore(k_inv=K_INV, capacity=4, history_size=3)

        first_slots, first_history_slots = store.add_observations(
            np.vstack((self.make_observation(10, 0.0), self.make_observation(20, 0.0)))
        )
        second_slots, second_history_slots = store.add_observations(
            np.vstack((self.make_observation(20, 1.0), self.make_observation(30, 1.0)))
        )

        np.testing.assert_array_equal(first_slots, np.array([0, 1], dtype=np.int32))
        np.testing.assert_array_equal(first_history_slots, np.array([0, 0], dtype=np.int32))
        np.testing.assert_array_equal(second_slots, np.array([1, 2], dtype=np.int32))
        np.testing.assert_array_equal(second_history_slots, np.array([1, 0], dtype=np.int32))
        np.testing.assert_array_equal(store._history_versions[:3], np.array([1, 2, 1], dtype=np.int32))  # noqa: SLF001

    def test_add_observations_stores_left_and_valid_right_bearings(self) -> None:
        """Observation ingest should cache stereo bearings without reshaping invalid right observations."""
        store = ObservationStore(k_inv=K_INV, capacity=2, history_size=1)
        observations = np.vstack((self.make_observation(10, 3.0, 4.0), self.make_observation(20, 6.0, 8.0)))
        observations[1, ObservationSchema.RIGHT_UV] = np.nan

        store.add_observations(observations)

        expected_left = np.array(
            [
                [3.0, 4.0, 1.0],
                [6.0, 8.0, 1.0],
            ],
            dtype=np.float64,
        )
        expected_left /= np.linalg.norm(expected_left, axis=1, keepdims=True)
        expected_right = np.array([2.0, 4.0, 1.0], dtype=np.float64)
        expected_right /= np.linalg.norm(expected_right)

        np.testing.assert_allclose(store._observations[:2, 0, ObservationSchema.LEFT_BEARING], expected_left)  # noqa: SLF001
        np.testing.assert_allclose(store._observations[0, 0, ObservationSchema.RIGHT_BEARING], expected_right)  # noqa: SLF001
        assert np.all(np.isnan(store._observations[1, 0, ObservationSchema.RIGHT_BEARING]))  # noqa: SLF001

    def test_ready_slots_history_versions_keep_growing_after_compression(self) -> None:
        """Ready slot history versions should track observation updates, not capped history size."""
        store = ObservationStore(
            k_inv=K_INV,
            capacity=1,
            history_size=3,
            compressed_history_size=2,
            compress_policy=CompressPolicy.TOP_DISPLACEMENT,
        )
        for left_u in [0.0, 2.0, 3.0, 4.0, 5.0]:
            store.add_observations(self.make_observation(10, left_u))

        criteria = ReadyObservationCriteria(
            min_history_size=3,
            min_parallax_rad=1.0,
            min_parallax_observations=2,
        )

        slots, history_versions, feat_ids = store.ready_slots(criteria, np.array([0], dtype=np.int32))

        np.testing.assert_array_equal(slots, np.array([0], dtype=np.int32))
        np.testing.assert_array_equal(history_versions, np.array([5], dtype=np.int32))
        np.testing.assert_array_equal(feat_ids, np.array([10], dtype=np.int32))

    def test_p90_parallax_select_policy_uses_supported_history_not_only_latest(self) -> None:
        """P90 parallax policy should keep middle motion evidence even when latest returns near anchor."""
        p90_store = ObservationStore(
            k_inv=K_INV,
            capacity=1,
            history_size=4,
            select_policy=SelectPolicy.P90_PARALLAX,
        )
        latest_store = ObservationStore(
            k_inv=K_INV,
            capacity=1,
            history_size=4,
            select_policy=SelectPolicy.ANCHOR_TO_LATEST_PARALLAX,
        )
        for left_u in [0.0, 2.0, 3.0, 0.2]:
            observation = self.make_observation(10, left_u)
            p90_store.add_observations(observation.copy())
            latest_store.add_observations(observation.copy())

        criteria = ReadyObservationCriteria(
            min_history_size=4,
            min_parallax_rad=1.0,
            min_parallax_observations=2,
        )

        p90_slots = p90_store.get_slots_by_criteria(criteria, np.array([0], dtype=np.int32))
        latest_slots = latest_store.get_slots_by_criteria(criteria, np.array([0], dtype=np.int32))

        np.testing.assert_array_equal(p90_slots, np.array([0], dtype=np.int32))
        np.testing.assert_array_equal(latest_slots, np.empty((0,), dtype=np.int32))

    def test_pixel_displacement_select_policy_uses_cached_anchor_pixel_displacement(self) -> None:
        """Pixel displacement policy should use pixel threshold instead of angular parallax threshold."""
        store = ObservationStore(
            k_inv=K_INV,
            capacity=1,
            history_size=3,
            select_policy=SelectPolicy.PIXEL_DISPLACEMENT,
        )
        for left_u in [0.0, 2.0, 0.2]:
            store.add_observations(self.make_observation(10, left_u))

        criteria = ReadyObservationCriteria(
            min_history_size=3,
            min_parallax_rad=100.0,
            min_parallax_observations=1,
            min_pixel_displacement=1.0,
        )

        slots = store.get_slots_by_criteria(criteria, np.array([0], dtype=np.int32))

        np.testing.assert_array_equal(slots, np.array([0], dtype=np.int32))

    def test_get_slots_by_criteria_compensates_camera_rotation(self) -> None:
        """Pure camera rotation should not make a feature ready for ray triangulation."""
        store = ObservationStore(k_inv=K_INV, capacity=1, history_size=3)
        anchor_bearing = np.array([0.2, 0.0, 1.0], dtype=np.float64)
        anchor_bearing /= np.linalg.norm(anchor_bearing)

        for theta in [0.0, 0.2, 0.4]:
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            world_from_cam0 = np.eye(4, dtype=np.float64)
            world_from_cam0[:3, :3] = np.array(
                [
                    [cos_theta, 0.0, sin_theta],
                    [0.0, 1.0, 0.0],
                    [-sin_theta, 0.0, cos_theta],
                ],
                dtype=np.float64,
            )
            bearing_cam0 = world_from_cam0[:3, :3].T @ anchor_bearing
            store.add_observations(
                self.make_observation(
                    10,
                    left_u=float(bearing_cam0[0] / bearing_cam0[2]),
                    left_v=float(bearing_cam0[1] / bearing_cam0[2]),
                    world_from_cam0=world_from_cam0,
                )
            )

        criteria = ReadyObservationCriteria(
            min_history_size=3,
            min_parallax_rad=0.01,
            min_parallax_observations=2,
        )

        slots = store.get_slots_by_criteria(criteria, np.array([0], dtype=np.int32))

        np.testing.assert_array_equal(slots, np.empty((0,), dtype=np.int32))

    def test_get_ready_feature_slice_returns_fixed_depth_histories(self) -> None:
        """Ready feature slice should preserve full store history depth."""
        store = ObservationStore(k_inv=K_INV, capacity=4, history_size=5)
        for left_u in [0.0, 2.0, 3.0]:
            store.add_observations(self.make_observation(10, left_u))
        for left_u in [0.0, 2.0, 3.0, 4.0, 5.0]:
            store.add_observations(self.make_observation(30, left_u))

        criteria = ReadyObservationCriteria(
            min_history_size=3,
            min_parallax_rad=1.0,
            min_parallax_observations=2,
        )

        feat_ids, histories, history_mask = store.get_ready_feature_slice(
            criteria, np.array([0, 1], dtype=np.int32)
        )

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
        store = ObservationStore(k_inv=K_INV, capacity=4, history_size=1)
        observations = np.full((4, ObservationSchema.size()), np.nan, dtype=np.float64)
        observations[:, ObservationSchema.FEAT_ID] = np.array([10, 20, 30, 40], dtype=np.int32)
        used_slots, history_slots = store.add_observations(observations)

        assert store._feat_ids_to_slot == {10: 0, 20: 1, 30: 2, 40: 3}  # noqa: SLF001
        np.testing.assert_array_equal(store._history_sizes[:4], np.array([1, 1, 1, 1], dtype=np.int32))  # noqa: SLF001
        np.testing.assert_array_equal(used_slots, np.array([0, 1, 2, 3], dtype=np.int32))
        np.testing.assert_array_equal(history_slots, np.array([0, 0, 0, 0], dtype=np.int32))
        expected_observations = observations.copy()
        expected_observations[:, ObservationSchema.ANCHOR_PIXEL_DISPLACEMENT] = 0.0
        np.testing.assert_array_equal(store._observations[0, 0], expected_observations[0])  # noqa: SLF001
        np.testing.assert_array_equal(store._observations[1, 0], expected_observations[1])  # noqa: SLF001

    def test_add_observations_uniform_recent_compresses_full_history_and_appends_current_observation(
        self,
    ) -> None:
        """Uniform-recent compression should keep anchor, sampled history, and current observation."""
        store = ObservationStore(
            k_inv=K_INV,
            capacity=1,
            history_size=5,
            compressed_history_size=3,
            compress_policy=CompressPolicy.UNIFORM_RECENT,
        )
        for left_u in range(6):
            store.add_observations(self.make_observation(10, float(left_u)))

        history = store.get_feat_history(10)

        assert store._history_sizes[0] == 4  # noqa: SLF001
        np.testing.assert_allclose(history[:, ObservationSchema.LEFT_U], np.array([0.0, 2.0, 4.0, 5.0]))
        np.testing.assert_allclose(
            history[:, ObservationSchema.ANCHOR_PIXEL_DISPLACEMENT],
            np.array([0.0, 2.0, 4.0, 5.0]),
        )
        assert np.all(np.isnan(store._observations[0, 4]))  # noqa: SLF001

    def test_add_observations_uniform_recent_recompresses_summary_and_recent_tail(self) -> None:
        """Repeated uniform-recent compression should keep anchor and bias toward the recent tail."""
        store = ObservationStore(
            k_inv=K_INV,
            capacity=1,
            history_size=5,
            compressed_history_size=3,
            compress_policy=CompressPolicy.UNIFORM_RECENT,
        )
        for left_u in range(8):
            store.add_observations(self.make_observation(10, float(left_u)))

        history = store.get_feat_history(10)

        assert store._history_sizes[0] == 4  # noqa: SLF001
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
            k_inv=K_INV,
            capacity=1,
            history_size=6,
            compressed_history_size=5,
            compress_policy=CompressPolicy.TOP_DISPLACEMENT,
        )
        for left_u in [0.0, 1.0, 10.0, 2.0, 9.0, 3.0, 4.0]:
            store.add_observations(self.make_observation(10, left_u))

        history = store.get_feat_history(10)

        assert store._history_sizes[0] == 6  # noqa: SLF001
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
            k_inv=K_INV,
            capacity=1,
            history_size=6,
            compressed_history_size=5,
            compress_policy=CompressPolicy.TOP_DISPLACEMENT,
        )
        for left_u in [0.0, 1.0, 10.0, 2.0, 9.0, 3.0, 4.0, 5.0]:
            store.add_observations(self.make_observation(10, left_u))

        history = store.get_feat_history(10)

        assert store._history_sizes[0] == 6  # noqa: SLF001
        np.testing.assert_allclose(
            history[:, ObservationSchema.LEFT_U],
            np.array([0.0, 10.0, 9.0, 3.0, 4.0, 5.0]),
        )
        np.testing.assert_allclose(
            history[:, ObservationSchema.ANCHOR_PIXEL_DISPLACEMENT],
            np.array([0.0, 10.0, 9.0, 3.0, 4.0, 5.0]),
        )
