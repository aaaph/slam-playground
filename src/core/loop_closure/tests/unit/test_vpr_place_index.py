from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest

from core.loop_closure.vpr_cache import InMemoryFrameCache
from core.loop_closure.vpr_frame import VPRFrame, VPRGeometrySchema
from core.loop_closure.vpr_place_index import QueryItemResult, VPRPlaceIndex, VPRPlaceIndexConfig


class FakeDB:
    """Fake DBoW database."""

    def __init__(self, query_result: list[QueryItemResult] | None) -> None:
        """Initialize the fake DB."""
        self.query = Mock(return_value=query_result)
        self.add = Mock(return_value=0)


def make_vpr_frame(frame_id: int) -> VPRFrame:
    """Make a VPR frame with deterministic descriptors."""
    return VPRFrame(
        frame_id=frame_id,
        kf_id=frame_id + 100,
        timestamp=float(frame_id),
        geometry=np.full((2, VPRGeometrySchema.count()), frame_id, dtype=np.float32),
        descriptors=np.full((2, 32), frame_id, dtype=np.uint8),
    )


def make_place_index(
    db: FakeDB,
    config: VPRPlaceIndexConfig,
    cache: InMemoryFrameCache | None = None,
) -> VPRPlaceIndex:
    """Make a place index with fake vocabulary/DB and an in-memory cache."""
    return VPRPlaceIndex(
        vocabulary=cast("object", object()),
        db=cast("object", db),
        cache=InMemoryFrameCache() if cache is None else cache,
        config=config,
    )


class TestVPRPlaceIndex:
    """Unit tests for VPRPlaceIndex."""

    @pytest.fixture
    def config(self) -> VPRPlaceIndexConfig:
        """Create a VPR place index config."""
        return VPRPlaceIndexConfig(
            max_k=5,
            recent_db_window=3,
            alpha=0.1,
            min_nss_factor=0.005,
            island_db_gap=10,
            min_island_size=3,
            max_island_span=100,
            min_island_density=0.1,
        )

    def test_query_loop_candidates_should_query_db_with_max_results(self, config: VPRPlaceIndexConfig):
        """Test that loop candidate query passes descriptor batch and max_k to the DB."""
        db = FakeDB(query_result=[])
        index = make_place_index(db, config)
        descriptors = np.ones((3, 32), dtype=np.uint8)

        result = index._query_loop_candidates(descriptors, current_db_id=0)  # noqa: SLF001

        db.query.assert_called_once_with(descriptors, max_results=config.max_k)
        assert result.matches == []
        assert result.nss_factor == 1.0
        assert result.score_threshold == config.alpha
        assert not result.is_low_nss

    def test_add_should_insert_descriptors_into_db_and_return_db_id(self, config: VPRPlaceIndexConfig):
        """Test that add delegates descriptor insertion to the DBoW database."""
        db = FakeDB(query_result=[])
        db.add.return_value = 7
        index = make_place_index(db, config)
        descriptors = np.ones((3, 32), dtype=np.uint8)

        db_id = index.add(descriptors)

        db.add.assert_called_once_with(descriptors)
        assert db_id == 7

    def test_add_frame_should_insert_descriptors_and_return_matching_db_id(self, config: VPRPlaceIndexConfig):
        """Test that add_frame inserts frame descriptors and returns the DB id."""
        db = FakeDB(query_result=[])
        db.add.return_value = 0
        index = make_place_index(db, config)
        frame = make_vpr_frame(frame_id=0)

        db_id = index.add_frame(frame)

        db.add.assert_called_once_with(frame.descriptors)
        assert db_id == frame.frame_id
        assert index.cache.get(frame.frame_id) is frame
        assert index.kf_id_to_frame_id[frame.kf_id] == frame.frame_id
        assert index.frame_id_to_kf_id[frame.frame_id] == frame.kf_id
        assert index.next_frame_id() == 1

    def test_add_frame_should_reject_mismatched_db_id(self, config: VPRPlaceIndexConfig):
        """Test that add_frame rejects a DB id that does not match frame.frame_id."""
        db = FakeDB(query_result=[])
        db.add.return_value = 4
        index = make_place_index(db, config)
        frame = make_vpr_frame(frame_id=0)

        with pytest.raises(RuntimeError, match="Place index id mismatch: frame_id=0, cache_id=0, db_id=4"):
            index.add_frame(frame)
        db.add.assert_called_once_with(frame.descriptors)

    def test_query_loop_candidates_should_filter_recent_and_low_score_matches(self, config: VPRPlaceIndexConfig):
        """Test that loop candidate query applies recent-window and relative score filters."""
        db = FakeDB(
            query_result=[
                QueryItemResult(Id=9, Score=0.4),  # latest previous frame, used as NSS
                QueryItemResult(Id=8, Score=0.5),  # too recent
                QueryItemResult(Id=7, Score=0.03),  # old enough, below threshold
                QueryItemResult(Id=6, Score=0.041),  # old enough, above threshold
                QueryItemResult(Id=2, Score=0.2),  # old enough, above threshold
            ]
        )
        index = make_place_index(db, config)

        result = index._query_loop_candidates(np.ones((3, 32), dtype=np.uint8), current_db_id=10)  # noqa: SLF001

        assert result.nss_factor == 0.4
        assert result.score_threshold == pytest.approx(0.04)
        assert result.matches == [
            QueryItemResult(Id=6, Score=0.041),
            QueryItemResult(Id=2, Score=0.2),
        ]
        assert not result.is_low_nss

    def test_query_loop_candidates_should_suppress_matches_before_recent_window_elapsed(
        self, config: VPRPlaceIndexConfig
    ):
        """Test that the first recent_db_window frames cannot produce loop candidates."""
        db = FakeDB(
            query_result=[
                QueryItemResult(Id=1, Score=0.9),  # latest previous frame, used as NSS
                QueryItemResult(Id=0, Score=0.8),  # visually strong but still too recent
            ]
        )
        index = make_place_index(db, config)

        result = index._query_loop_candidates(np.ones((3, 32), dtype=np.uint8), current_db_id=2)  # noqa: SLF001

        assert result.nss_factor == 0.9
        assert result.score_threshold == pytest.approx(0.09)
        assert not result.is_low_nss
        assert result.matches == []

    def test_query_loop_candidates_should_drop_matches_when_nss_is_too_low(self, config: VPRPlaceIndexConfig):
        """Test that low NSS suppresses otherwise valid matches."""
        db = FakeDB(
            query_result=[
                QueryItemResult(Id=4, Score=0.004),
                QueryItemResult(Id=1, Score=0.02),
            ]
        )
        index = make_place_index(db, config)

        result = index._query_loop_candidates(np.ones((3, 32), dtype=np.uint8), current_db_id=5)  # noqa: SLF001

        assert result.nss_factor == 0.004
        assert result.is_low_nss
        assert result.matches == []

    def test_query_loop_candidates_should_handle_empty_db_result(self, config: VPRPlaceIndexConfig):
        """Test that a None query result is treated as no matches."""
        db = FakeDB(query_result=None)
        index = make_place_index(db, config)

        result = index._query_loop_candidates(np.ones((3, 32), dtype=np.uint8), current_db_id=3)  # noqa: SLF001

        assert result.nss_factor == 0.0
        assert result.score_threshold == 0.0
        assert result.is_low_nss
        assert result.matches == []
