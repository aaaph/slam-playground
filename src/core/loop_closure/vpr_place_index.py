from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pydbow3  # ty: ignore[unresolved-import]

from core.loop_closure.vpr_cache import InMemoryFrameCache, VPRFrameCache
from core.loop_closure.vpr_frame import VPRFrame


class QueryItemResult(NamedTuple):
    """DBOW3 Query result."""

    Id: int
    Score: float


class LoopCandidateQuery(NamedTuple):
    """Filtered loop candidate query result."""

    matches: list[QueryItemResult]
    nss_factor: float
    score_threshold: float
    is_low_nss: bool

    def __repr__(self) -> str:
        """Return a string representation of the loop candidate query."""
        return (
            f"LoopCandidateQuery(matches={len(self.matches)}, "
            f"nss_factor={self.nss_factor:.6f}, score_threshold={self.score_threshold:.6f}, "
            f"is_low_nss={self.is_low_nss})"
        )


class PlaceReference(NamedTuple):
    """Group of nearby DBOW matches."""

    matches: list[QueryItemResult]
    score: float
    weighted_score: float
    density: float
    best_match: QueryItemResult
    start_db_id: int
    end_db_id: int

    def __repr__(self) -> str:
        """Return a string representation of the match island."""
        return (
            f"MatchIsland(score={self.score:.6f}, weighted_score={self.weighted_score:.6f}, "
            f"density={self.density:.3f}, best_match={self.best_match.Id}, start_db_id={self.start_db_id}, "
            f"end_db_id={self.end_db_id})"
        )

    @classmethod
    def empty(cls) -> "PlaceReference":
        """Create an empty place reference."""
        return cls(
            matches=[],
            score=0.0,
            weighted_score=0.0,
            density=0.0,
            best_match=QueryItemResult(Id=0, Score=0.0),
            start_db_id=0,
            end_db_id=0,
        )


@dataclass(frozen=True, slots=True)
class VPRPlaceIndexConfig:
    """VPR place index configuration."""

    max_k: int
    recent_db_window: int
    alpha: float
    min_nss_factor: float
    island_db_gap: int
    min_island_size: int
    max_island_span: int
    min_island_density: float


class VPRPlaceIndex:
    """VPR place index."""

    def __init__(
        self,
        vocabulary: pydbow3.Vocabulary,
        db: pydbow3.Database,
        cache: VPRFrameCache,
        config: VPRPlaceIndexConfig,
    ) -> None:
        """Initialize the VPR place index."""
        self.vocabulary = vocabulary
        self.db = db
        self.config = config
        self.cache = cache
        self.kf_id_to_frame_id: dict[int, int] = {}
        self.frame_id_to_kf_id: dict[int, int] = {}

    @classmethod
    def from_vocabulary(
        cls, vocabulary: pydbow3.Vocabulary, config: VPRPlaceIndexConfig | None = None
    ) -> "VPRPlaceIndex":
        """Create a VPR place index from a vocabulary and database."""
        database = pydbow3.Database()
        use_direct_index = False
        database.setVocabulary(vocabulary, use_direct_index, 0)

        return cls.default_factory(vocabulary, database, InMemoryFrameCache(), config)

    @classmethod
    def default_factory(
        cls,
        vocabulary: pydbow3.Vocabulary,
        db: pydbow3.Database,
        cache: InMemoryFrameCache | None = None,
        config: VPRPlaceIndexConfig | None = None,
    ) -> "VPRPlaceIndex":
        """Create a default VPR place index."""
        if config is None:
            config = VPRPlaceIndexConfig(
                max_k=5,
                recent_db_window=15,
                alpha=0.1,
                min_nss_factor=0.005,
                island_db_gap=10,
                min_island_size=3,
                max_island_span=100,
                min_island_density=0.1,
            )
        cache = InMemoryFrameCache() if cache is None else cache
        return VPRPlaceIndex(vocabulary, db, cache, config)

    def find_loop_candidate(self, frame: VPRFrame) -> tuple[bool, PlaceReference, VPRFrame]:
        """Find the loop candidate passed recent/NSS/score/groups gates."""
        loop_candidate_query_result = self._query_loop_candidates(frame.descriptors, frame.frame_id)
        grouped_matches = self._group_matches_in_places(loop_candidate_query_result.matches)
        reference_place = self._select_best_place_reference(grouped_matches)
        if reference_place is None:
            return False, PlaceReference.empty(), frame
        reference_frame = self.cache.get(reference_place.best_match.Id)
        if reference_frame is None:
            msg = f"Reference {reference_place.best_match.Id} exists in DB but not in cache"
            raise RuntimeError(msg)
        return True, reference_place, reference_frame

    def add(self, descriptors: np.ndarray) -> int:
        """Add a VPR frame descriptor batch to the DBoW database."""
        return int(self.db.add(descriptors))

    def add_frame(self, frame: VPRFrame) -> int:
        """Add a VPR frame to the DBoW database and cache."""
        cache_id = self.cache.add(frame)
        db_id = self.add(frame.descriptors)
        if frame.frame_id != db_id or frame.frame_id != cache_id:
            msg = f"Place index id mismatch: frame_id={frame.frame_id}, cache_id={cache_id}, db_id={db_id}"
            raise RuntimeError(msg)
        self.kf_id_to_frame_id[frame.kf_id] = frame.frame_id
        self.frame_id_to_kf_id[frame.frame_id] = frame.kf_id
        return frame.frame_id

    def next_frame_id(self) -> int:
        """Get the next frame ID."""
        return len(self.frame_id_to_kf_id)

    @staticmethod
    def _get_neighbor_score(raw_query_result: list[QueryItemResult], current_db_id: int) -> float:
        """Approximate Kimera NSS using the latest DB entry score from raw query results."""
        latest_db_id = current_db_id - 1
        if latest_db_id < 0:
            return 1.0
        for result in raw_query_result:
            if result.Id == latest_db_id:
                return float(result.Score)
        return 0.0

    def _query_loop_candidates(
        self,
        descriptors: np.ndarray,
        current_db_id: int,
    ) -> LoopCandidateQuery:
        """Query DBOW and apply the VPR candidate gates."""
        query_result: list[QueryItemResult] | None = self.db.query(descriptors, max_results=self.config.max_k)
        raw_query_result = [] if query_result is None else list(query_result)

        nss_factor = self._get_neighbor_score(raw_query_result, current_db_id)
        score_threshold = self.config.alpha * nss_factor

        recent_query_result = [
            result for result in raw_query_result if current_db_id - result.Id >= self.config.recent_db_window
        ]
        score_query_result = [result for result in recent_query_result if result.Score >= score_threshold]

        is_low_nss = current_db_id > 0 and nss_factor < self.config.min_nss_factor
        matches = [] if is_low_nss else score_query_result

        return LoopCandidateQuery(
            matches=matches,
            nss_factor=nss_factor,
            score_threshold=score_threshold,
            is_low_nss=is_low_nss,
        )

    def _group_matches_in_places(self, matches: list[QueryItemResult]) -> list[PlaceReference]:
        """Group matches whose DB ids are close to each other."""
        if len(matches) == 0:
            return []

        references: list[PlaceReference] = []
        current_matches: list[QueryItemResult] = []
        for match in sorted(matches, key=lambda result: result.Id):
            if current_matches and match.Id - current_matches[-1].Id > self.config.island_db_gap:
                self._append_match_reference(references, current_matches)
                current_matches = []
            current_matches.append(match)
        self._append_match_reference(references, current_matches)
        return references

    def _select_best_place_reference(self, references: list[PlaceReference]) -> PlaceReference | None:
        """Choose the best island after island-level filtering."""
        if len(references) == 0:
            return None
        return max(
            references,
            key=lambda reference: (
                reference.weighted_score,
                reference.best_match.Score,
                reference.score,
                len(reference.matches),
            ),
        )

    def _append_match_reference(
        self,
        place_references: list[PlaceReference],
        matches: list[QueryItemResult],
    ) -> None:
        """Append a match reference if it passes the minimum island size."""
        if len(matches) < self.config.min_island_size:
            return
        span = matches[-1].Id - matches[0].Id + 1
        if span > self.config.max_island_span:
            return
        density = len(matches) / span
        if density < self.config.min_island_density:
            return
        score = sum(result.Score for result in matches)
        best_match = max(matches, key=lambda result: result.Score)
        place_references.append(
            PlaceReference(
                matches=matches,
                score=score,
                weighted_score=score * density,
                density=density,
                best_match=best_match,
                start_db_id=matches[0].Id,
                end_db_id=matches[-1].Id,
            )
        )
