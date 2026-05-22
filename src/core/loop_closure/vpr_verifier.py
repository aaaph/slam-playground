from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np

from core.camera_model.stereo_camera_ctx import StereoContext
from core.loop_closure.vpr_3d3d_estimator import K3D3DEstimator, K3D3DResult
from core.loop_closure.vpr_frame import VPRFrame
from core.loop_closure.vpr_matcher import VPRMatcher
from core.loop_closure.vpr_place_index import PlaceReference
from core.transformations.special_euclidian_3_dim import SE3
from logger import spawn_logger


class LoopProposal(NamedTuple):
    """Loop proposal."""

    query_id: int
    best_island_id: int
    start_island_id: int
    end_island_id: int
    score: float


@dataclass(frozen=True, slots=True)
class EssentialVerification:
    """Essential matrix verification result."""

    matches: list[cv2.DMatch]
    inlier_mask: np.ndarray

    @property
    def matches_count(self) -> int:
        """Get descriptor matches count."""
        return len(self.matches)

    @property
    def inliers_count(self) -> int:
        """Get essential matrix inliers count."""
        return int(np.count_nonzero(self.inlier_mask))

    @property
    def inliers_ratio(self) -> float:
        """Get essential matrix inliers ratio."""
        if self.matches_count == 0:
            return 0.0
        return self.inliers_count / self.matches_count

    @property
    def inliers_matches(self) -> list[cv2.DMatch]:
        """Get inliner matches."""
        return [match for match, is_inlier in zip(self.matches, self.inlier_mask, strict=False) if is_inlier]

    @classmethod
    def empty(cls) -> "EssentialVerification":
        """Create an empty essential verification result."""
        return cls(matches=[], inlier_mask=np.empty((0,), dtype=bool))

    @classmethod
    def empty_with_matches(cls, matches: list[cv2.DMatch]) -> "EssentialVerification":
        """Create an empty essential verification result with matches."""
        inliner_mask = np.zeros((len(matches),), dtype=bool)
        return cls(matches=matches, inlier_mask=inliner_mask)

    def accepted(self, min_matches: int, min_inliers: int, min_inliers_ratio: float) -> bool:
        """Check if the essential verification is accepted."""
        return (
            self.matches_count >= min_matches
            and self.inliers_count >= min_inliers
            and self.inliers_ratio >= min_inliers_ratio
        )


@dataclass(frozen=True, slots=True)
class VerifyResult:
    """VPR verification result."""

    query_id: int
    reference_id: int
    accepted: bool
    temporal_consistent: bool
    essential_consistent: bool
    geometric_consistent: bool
    history_depth: int
    se3: SE3
    essential_inliners_count: int
    essntial_matches_count: int
    geometric_inliners_count: int
    geometric_matches_count: int
    matches: list[cv2.DMatch]
    inlier_mask: np.ndarray

    @property
    def essential_inliers_ratio(self) -> float:
        """Get the essential inliers ratio."""
        return self.essential_inliners_count / self.essntial_matches_count

    @property
    def essential_inliers_ratio_str(self) -> str:
        """Get the essential inliers ratio as a string."""
        return f"{self.essential_inliners_count}/{self.essntial_matches_count}"

    @property
    def geometric_inliers_ratio(self) -> float:
        """Get the geometric inliers ratio."""
        return self.geometric_inliners_count / self.geometric_matches_count

    @property
    def geometric_inliers_ratio_str(self) -> str:
        """Get the geometric inliers ratio as a string."""
        return f"{self.geometric_inliners_count}/{self.geometric_matches_count}"

    def __repr__(self) -> str:
        """Return a string representation of the verify result."""
        return (
            f"VerifyResult(query_id={self.query_id}, reference_id={self.reference_id}, "
            f"accepted={self.accepted}, temporal_consistent={self.temporal_consistent}, "
            f"geometric_consistent={self.geometric_consistent}, history_depth={self.history_depth}, "
            f"se3={self.se3})"
        )


@dataclass(frozen=True, slots=True)
class VPRFrameVerifierConfig:
    """VPR frame verifier configuration."""

    history_size: int
    min_island_weighted_score: float
    min_votes: int
    temporal_db_tolerance: float
    temporal_min_overlap_ratio: float
    min_essential_matches: int
    min_essential_inliers: int
    min_essential_matches_ratio: float
    min_rigid3d_inliers: int = 12
    min_rigid3d_inliers_ratio: float = 0.7
    max_rigid3d_median_residual_m: float = 0.05
    rigid3d_ransac_threshold_m: float = 0.15
    rigid3d_ransac_iterations: int = 100
    rigid3d_ransac_seed: int = 42


class VPRFrameVerifier:
    """VPR frame verifier."""

    def __init__(
        self,
        matcher: VPRMatcher,
        config: VPRFrameVerifierConfig,
        stereo_ctx: StereoContext,
    ) -> None:
        """Initialize the VPR frame verifier."""
        self.normalized_threshold = 1.0 / stereo_ctx.stereo_k[0, 0]
        self.matcher = matcher
        self.place_deque = deque(maxlen=config.history_size)
        self.config = config
        self.stereo_ctx = stereo_ctx
        self.k3d3d_estimator = self._create_k3d3d_estimator(config)
        self.logger = spawn_logger(app="vpr_frame_verifier")

    @classmethod
    def default_factory(
        cls, stereo_ctx: StereoContext, config: VPRFrameVerifierConfig | None = None
    ) -> "VPRFrameVerifier":
        """Create a default VPR frame verifier."""
        if config is None:
            config = VPRFrameVerifierConfig(
                history_size=5,
                min_island_weighted_score=0.5,
                min_votes=3,
                temporal_db_tolerance=10,
                temporal_min_overlap_ratio=0.3,
                min_essential_matches=12,
                min_essential_inliers=8,
                min_essential_matches_ratio=0.5,
            )
        return cls(matcher=VPRMatcher.default_factory(), config=config, stereo_ctx=stereo_ctx)

    @staticmethod
    def _create_k3d3d_estimator(config: VPRFrameVerifierConfig) -> K3D3DEstimator:
        """Create and configure the K3D3D estimator used by geometric verification."""
        estimator = K3D3DEstimator.default_factory()
        estimator.rigid3d_ransac_threshold_m = config.rigid3d_ransac_threshold_m
        estimator.rigid3d_ransac_iterations = config.rigid3d_ransac_iterations
        estimator.rigid3d_ransac_seed = config.rigid3d_ransac_seed
        return estimator

    def verify(self, query_frame: VPRFrame, place: PlaceReference, reference_frame: VPRFrame) -> VerifyResult:
        """Verify the frames."""
        query_id = query_frame.frame_id
        reference_id = reference_frame.frame_id
        history_depth = self._get_history_similarity(query_frame, place)
        essential_verification = EssentialVerification.empty()
        geometric_verification = K3D3DResult.empty()

        temporal_consistency = history_depth >= self.config.min_votes
        if temporal_consistency:
            essential_verification = self._get_essential_verification(query_frame, reference_frame)

        essential_consistency = temporal_consistency and essential_verification.accepted(
            self.config.min_essential_matches,
            self.config.min_essential_inliers,
            self.config.min_essential_matches_ratio,
        )
        if essential_consistency:
            geometric_verification = self.k3d3d_estimator.estimate_query_pose(
                query_frame, reference_frame, essential_verification.inliers_matches
            )

        geometric_consistency = essential_consistency and geometric_verification.accepted(
            self.config.min_rigid3d_inliers,
            self.config.min_rigid3d_inliers_ratio,
            self.config.max_rigid3d_median_residual_m,
        )
        accepted = temporal_consistency and geometric_consistency

        return VerifyResult(
            query_id=query_id,
            reference_id=reference_id,
            accepted=accepted,
            temporal_consistent=temporal_consistency,
            essential_consistent=essential_consistency,
            geometric_consistent=geometric_consistency,
            history_depth=history_depth,
            se3=geometric_verification.reference_t_query,
            essential_inliners_count=essential_verification.inliers_count,
            essntial_matches_count=essential_verification.matches_count,
            geometric_inliners_count=geometric_verification.inliers_count,
            geometric_matches_count=geometric_verification.matches_count,
            matches=geometric_verification.matches.copy(),
            inlier_mask=geometric_verification.inlier_mask.copy(),
        )

    def _get_history_similarity(self, query_frame: VPRFrame, place: PlaceReference) -> int:
        """Get the history similarity."""
        query_id = query_frame.frame_id

        if place.best_match is None:
            self.place_deque.append(None)
            return 0
        if place.weighted_score < self.config.min_island_weighted_score:
            self.place_deque.append(None)
            return 0

        loop_proposal = LoopProposal(
            query_id=query_id,
            best_island_id=place.best_match.Id,
            start_island_id=place.start_db_id,
            end_island_id=place.end_db_id,
            score=place.weighted_score,
        )
        similar = [
            old for old in self.place_deque if old is not None and self._compare_loop_proposals(old, loop_proposal)
        ]
        self.place_deque.append(loop_proposal)
        return len(similar) + 1

    def _get_essential_verification(
        self,
        query_frame: VPRFrame,
        reference_frame: VPRFrame,
    ) -> EssentialVerification:
        """Run essential matrix verification and return match/inlier statistics."""
        matches = self.matcher.match(query_frame.descriptors, reference_frame.descriptors)
        if len(matches) < self.config.min_essential_matches:
            return EssentialVerification.empty_with_matches(matches)

        ref_bearings = reference_frame.bearings[[m.trainIdx for m in matches]]
        ref_norm = ref_bearings[:, :2] / ref_bearings[:, 2:3]

        query_bearings = query_frame.bearings[[m.queryIdx for m in matches]]
        query_norm = query_bearings[:, :2] / query_bearings[:, 2:3]

        _, inliers = cv2.findEssentialMat(
            ref_norm,
            query_norm,
            np.eye(3),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=self.normalized_threshold,
        )

        return EssentialVerification(
            matches=matches,
            inlier_mask=inliers.ravel().astype(bool),
        )

    def _compare_loop_proposals(self, proposal1: LoopProposal, proposal2: LoopProposal) -> bool:
        """Check if two proposals point to the same historical DB island."""
        previous_center = (proposal1.start_island_id + proposal1.end_island_id) * 0.5
        current_center = (proposal2.start_island_id + proposal2.end_island_id) * 0.5
        if abs(previous_center - current_center) > self.config.temporal_db_tolerance:
            return False

        overlap = (
            min(proposal1.end_island_id, proposal2.end_island_id)
            - max(
                proposal1.start_island_id,
                proposal2.start_island_id,
            )
            + 1
        )
        if overlap <= 0:
            return False

        previous_span = proposal1.end_island_id - proposal1.start_island_id + 1
        current_span = proposal2.end_island_id - proposal2.start_island_id + 1
        overlap_ratio = overlap / min(previous_span, current_span)
        return overlap_ratio >= self.config.temporal_min_overlap_ratio
