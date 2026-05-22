from typing import TYPE_CHECKING, cast

import cv2
import numpy as np
import pytest

from core.camera_model.stereo_camera_ctx import StereoContext
from core.loop_closure.vpr_3d3d_estimator import K3D3DResult
from core.loop_closure.vpr_frame import VPRFrame, VPRGeometrySchema
from core.loop_closure.vpr_place_index import PlaceReference, QueryItemResult
from core.loop_closure.vpr_verifier import (
    EssentialVerification,
    VerifyResult,
    VPRFrameVerifier,
    VPRFrameVerifierConfig,
)
from core.transformations.special_euclidian_3_dim import SE3

if TYPE_CHECKING:
    from core.loop_closure.vpr_3d3d_estimator import K3D3DEstimator
    from core.loop_closure.vpr_matcher import VPRMatcher


def make_vpr_frame(frame_id: int, rows_count: int = 12) -> VPRFrame:
    """Make a minimal VPR frame."""
    geometry = np.zeros((rows_count, VPRGeometrySchema.count()), dtype=np.float32)
    geometry[:, VPRGeometrySchema.BEARING_Z] = 1.0
    geometry[:, VPRGeometrySchema.POINT_Z] = 1.0
    return VPRFrame(
        frame_id=frame_id,
        kf_id=frame_id + 100,
        timestamp=float(frame_id),
        geometry=geometry,
        descriptors=np.zeros((rows_count, 32), dtype=np.uint8),
    )


def make_matches(count: int) -> list[cv2.DMatch]:
    """Make deterministic descriptor matches."""
    return [cv2.DMatch(_queryIdx=index, _trainIdx=index, _distance=0.0) for index in range(count)]


def make_essential_verification(matches_count: int, inliers_count: int) -> EssentialVerification:
    """Make an essential verification payload with deterministic counts."""
    return EssentialVerification(
        matches=make_matches(matches_count),
        inlier_mask=np.arange(matches_count) < inliers_count,
    )


def make_k3d3d_result(
    matches_count: int,
    inliers_count: int,
    *,
    residual: float = 0.0,
    success: bool = True,
) -> K3D3DResult:
    """Make a K3D3D result with deterministic counts."""
    return K3D3DResult(
        success=success,
        reason="Success" if success else "Failed",
        reference_t_query=SE3.identity(),
        matches=make_matches(matches_count),
        inlier_mask=np.arange(matches_count) < inliers_count,
        residuals=np.full((matches_count,), residual, dtype=np.float64),
    )


def make_place_reference(
    start_db_id: int,
    end_db_id: int,
    *,
    weighted_score: float = 0.2,
) -> PlaceReference:
    """Make a place reference for temporal verification."""
    best_db_id = (start_db_id + end_db_id) // 2
    best_match = QueryItemResult(Id=best_db_id, Score=weighted_score)
    return PlaceReference(
        matches=[best_match],
        score=weighted_score,
        weighted_score=weighted_score,
        density=1.0,
        best_match=best_match,
        start_db_id=start_db_id,
        end_db_id=end_db_id,
    )


@pytest.fixture
def config() -> VPRFrameVerifierConfig:
    """Create a verifier config with two-vote temporal confirmation."""
    return VPRFrameVerifierConfig(
        history_size=5,
        min_island_weighted_score=0.1,
        min_votes=2,
        temporal_db_tolerance=8,
        temporal_min_overlap_ratio=0.3,
        min_essential_matches=12,
        min_essential_inliers=8,
        min_essential_matches_ratio=0.5,
    )


def make_stereo_ctx() -> StereoContext:
    """Make a minimal stereo context with the camera matrix used by verifier."""
    return StereoContext(
        resolution=(752, 480),
        stereo_k=np.array(
            [
                [460.0, 0.0, 376.0],
                [0.0, 460.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        cam0_k=np.array(
            [
                [460.0, 0.0, 376.0],
                [0.0, 460.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        cam1_k=np.array(
            [
                [460.0, 0.0, 376.0],
                [0.0, 460.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        baseline=0.1,
        cam0_in_body_se3=SE3.identity(),
        cam1_in_body_se3=SE3.identity(),
    )


class FakeK3D3DEstimator:
    """K3D3D estimator test double."""

    def __init__(self, result: K3D3DResult) -> None:
        """Initialize the fake estimator."""
        self.result = result
        self.calls: list[tuple[VPRFrame, VPRFrame, list[cv2.DMatch]]] = []

    def estimate_query_pose(
        self,
        query_frame: VPRFrame,
        reference_frame: VPRFrame,
        matches: list[cv2.DMatch],
    ) -> K3D3DResult:
        """Record the call and return the configured result."""
        self.calls.append((query_frame, reference_frame, matches))
        return self.result


class GeometricStubVerifier(VPRFrameVerifier):
    """Verifier test double with deterministic essential verification."""

    def __init__(
        self,
        matcher: "VPRMatcher",
        config: VPRFrameVerifierConfig,
        stereo_ctx: StereoContext,
        essential_verification: EssentialVerification,
        k3d3d_result: K3D3DResult,
    ) -> None:
        """Initialize the verifier test double."""
        super().__init__(matcher=matcher, config=config, stereo_ctx=stereo_ctx)
        self.essential_verification = essential_verification
        self.fake_k3d3d_estimator = FakeK3D3DEstimator(k3d3d_result)
        self.k3d3d_estimator = cast("K3D3DEstimator", self.fake_k3d3d_estimator)

    def _get_essential_verification(
        self,
        query_frame: VPRFrame,
        reference_frame: VPRFrame,
    ) -> EssentialVerification:
        """Return a deterministic essential verification result."""
        return self.essential_verification


def make_verifier(
    config: VPRFrameVerifierConfig,
    essential_verification: EssentialVerification | None = None,
    k3d3d_result: K3D3DResult | None = None,
) -> GeometricStubVerifier:
    """Make a verifier with deterministic verification dependencies."""
    if essential_verification is None:
        essential_verification = make_essential_verification(matches_count=12, inliers_count=8)
    if k3d3d_result is None:
        k3d3d_result = make_k3d3d_result(matches_count=12, inliers_count=12)
    return GeometricStubVerifier(
        matcher=cast("VPRMatcher", object()),
        config=config,
        stereo_ctx=make_stereo_ctx(),
        essential_verification=essential_verification,
        k3d3d_result=k3d3d_result,
    )


def verify_place(verifier: VPRFrameVerifier, query_frame_id: int, place: PlaceReference) -> VerifyResult:
    """Verify a query frame against the place reference and matching cached frame."""
    return verifier.verify(
        make_vpr_frame(query_frame_id),
        place,
        make_vpr_frame(place.best_match.Id),
    )


class TestVPRFrameVerifier:
    """Unit tests for VPRFrameVerifier."""

    def test_verify_should_accept_same_place_after_min_votes(self, config: VPRFrameVerifierConfig) -> None:
        """Repeated references to the same place should become temporally consistent."""
        verifier = make_verifier(config)
        place = make_place_reference(10, 12)

        first = verify_place(verifier, 20, place)
        second = verify_place(verifier, 21, place)

        assert not first.accepted
        assert second.accepted

    def test_verify_should_accept_overlapping_islands_after_min_votes(
        self, config: VPRFrameVerifierConfig
    ) -> None:
        """Nearby overlapping islands should count as the same place."""
        verifier = make_verifier(config)

        first = verify_place(verifier, 20, make_place_reference(10, 14))
        second = verify_place(verifier, 21, make_place_reference(12, 16))

        assert not first.accepted
        assert second.accepted

    def test_verify_should_reject_non_overlapping_islands(self, config: VPRFrameVerifierConfig) -> None:
        """Separated islands should not satisfy temporal consistency."""
        verifier = make_verifier(config)

        first = verify_place(verifier, 20, make_place_reference(10, 12))
        second = verify_place(verifier, 21, make_place_reference(20, 22))

        assert not first.accepted
        assert not second.accepted

    def test_verify_should_reject_low_weighted_score(self, config: VPRFrameVerifierConfig) -> None:
        """Low-score candidates should not enter the proposal history."""
        verifier = make_verifier(config)

        result = verify_place(verifier, 20, make_place_reference(10, 12, weighted_score=0.05))

        assert not result.accepted
        assert not result.temporal_consistent
        assert not result.geometric_consistent
        assert list(verifier.place_deque) == [None]

    def test_verify_should_reject_essential_verification_with_too_few_matches(
        self, config: VPRFrameVerifierConfig
    ) -> None:
        """Essential verification should reject weak minimal-match cases."""
        verifier = make_verifier(config, make_essential_verification(matches_count=5, inliers_count=5))
        place = make_place_reference(10, 12)

        verify_place(verifier, 20, place)
        result = verify_place(verifier, 21, place)

        assert not result.accepted
        assert result.temporal_consistent
        assert not result.essential_consistent
        assert not result.geometric_consistent
        assert result.essntial_matches_count == 5
        assert result.essential_inliners_count == 5
        assert verifier.fake_k3d3d_estimator.calls == []

    def test_verify_should_reject_essential_verification_with_too_few_inliers(
        self, config: VPRFrameVerifierConfig
    ) -> None:
        """Essential verification should require enough absolute inliers."""
        verifier = make_verifier(config, make_essential_verification(matches_count=12, inliers_count=7))
        place = make_place_reference(10, 12)

        verify_place(verifier, 20, place)
        result = verify_place(verifier, 21, place)

        assert not result.accepted
        assert result.temporal_consistent
        assert not result.essential_consistent
        assert not result.geometric_consistent
        assert result.essntial_matches_count == 12
        assert result.essential_inliners_count == 7
        assert verifier.fake_k3d3d_estimator.calls == []

    def test_verify_should_reject_essential_verification_with_low_inlier_ratio(
        self, config: VPRFrameVerifierConfig
    ) -> None:
        """Essential verification should require enough inlier ratio."""
        verifier = make_verifier(config, make_essential_verification(matches_count=20, inliers_count=8))
        place = make_place_reference(10, 12)

        verify_place(verifier, 20, place)
        result = verify_place(verifier, 21, place)

        assert not result.accepted
        assert result.temporal_consistent
        assert not result.essential_consistent
        assert not result.geometric_consistent
        assert result.essntial_matches_count == 20
        assert result.essential_inliners_count == 8
        assert verifier.fake_k3d3d_estimator.calls == []

    def test_verify_should_pass_only_essential_inlier_matches_to_k3d3d(
        self, config: VPRFrameVerifierConfig
    ) -> None:
        """K3D3D should receive only matches that passed the essential gate."""
        verifier = make_verifier(config, make_essential_verification(matches_count=12, inliers_count=8))
        place = make_place_reference(10, 12)

        verify_place(verifier, 20, place)
        result = verify_place(verifier, 21, place)

        assert result.accepted
        assert len(verifier.fake_k3d3d_estimator.calls) == 1
        _query_frame, _reference_frame, matches = verifier.fake_k3d3d_estimator.calls[0]
        assert len(matches) == 8
        assert [match.queryIdx for match in matches] == list(range(8))

    def test_verify_should_reject_k3d3d_verification_with_too_few_inliers(
        self, config: VPRFrameVerifierConfig
    ) -> None:
        """K3D3D verification should require enough absolute inliers."""
        verifier = make_verifier(
            config,
            make_essential_verification(matches_count=12, inliers_count=8),
            make_k3d3d_result(matches_count=12, inliers_count=11),
        )
        place = make_place_reference(10, 12)

        verify_place(verifier, 20, place)
        result = verify_place(verifier, 21, place)

        assert not result.accepted
        assert result.temporal_consistent
        assert result.essential_consistent
        assert not result.geometric_consistent
        assert result.essntial_matches_count == 12
        assert result.geometric_matches_count == 12
        assert result.geometric_inliners_count == 11

    def test_verify_should_reject_k3d3d_verification_with_low_inlier_ratio(
        self, config: VPRFrameVerifierConfig
    ) -> None:
        """K3D3D verification should require enough inlier ratio."""
        verifier = make_verifier(
            config,
            make_essential_verification(matches_count=20, inliers_count=20),
            make_k3d3d_result(matches_count=20, inliers_count=13),
        )
        place = make_place_reference(10, 12)

        verify_place(verifier, 20, place)
        result = verify_place(verifier, 21, place)

        assert not result.accepted
        assert result.temporal_consistent
        assert result.essential_consistent
        assert not result.geometric_consistent
        assert result.essntial_matches_count == 20
        assert result.geometric_matches_count == 20
        assert result.geometric_inliners_count == 13

    def test_verify_should_reject_k3d3d_verification_with_high_residual(
        self, config: VPRFrameVerifierConfig
    ) -> None:
        """K3D3D verification should reject high residual transforms."""
        verifier = make_verifier(
            config,
            make_essential_verification(matches_count=12, inliers_count=8),
            make_k3d3d_result(matches_count=12, inliers_count=12, residual=0.06),
        )
        place = make_place_reference(10, 12)

        verify_place(verifier, 20, place)
        result = verify_place(verifier, 21, place)

        assert not result.accepted
        assert result.temporal_consistent
        assert result.essential_consistent
        assert not result.geometric_consistent
        assert result.essntial_matches_count == 12
        assert result.geometric_matches_count == 12

    def test_verify_should_return_matches_and_inlier_mask_for_visualization(
        self, config: VPRFrameVerifierConfig
    ) -> None:
        """Verify result should keep geometric inlier match payload for later visualization."""
        essential = make_essential_verification(matches_count=12, inliers_count=8)
        k3d3d = make_k3d3d_result(matches_count=8, inliers_count=7)
        verifier = make_verifier(config, essential, k3d3d)
        place = make_place_reference(10, 12)

        verify_place(verifier, 20, place)
        result = verify_place(verifier, 21, place)

        assert isinstance(result, VerifyResult)
        assert not result.accepted
        assert result.matches == k3d3d.matches
        assert result.matches is not k3d3d.matches
        np.testing.assert_array_equal(result.inlier_mask, k3d3d.inlier_mask)
        assert result.essntial_matches_count == 12
        assert result.essential_inliners_count == 8
        assert result.essential_inliers_ratio == pytest.approx(8 / 12)
        assert result.geometric_matches_count == 8
        assert result.geometric_inliners_count == 7
        assert result.geometric_inliers_ratio == pytest.approx(7 / 8)

    def test_verify_result_should_expose_k3d3d_pose_as_matrix_and_se3(
        self, config: VPRFrameVerifierConfig
    ) -> None:
        """Verify result should expose accepted K3D3D pose through the legacy pose properties."""
        reference_t_query = SE3.from_quat_and_translation(
            quat=np.array([0.0, 0.0, 0.0, 1.0]),
            translation=np.array([0.1, -0.2, 0.3]),
        )
        k3d3d = K3D3DResult(
            success=True,
            reason="Success",
            reference_t_query=reference_t_query,
            matches=make_matches(12),
            inlier_mask=np.ones((12,), dtype=bool),
            residuals=np.zeros((12,), dtype=np.float64),
        )
        verifier = make_verifier(config, k3d3d_result=k3d3d)
        place = make_place_reference(10, 12)

        verify_place(verifier, 20, place)
        result = verify_place(verifier, 21, place)

        np.testing.assert_allclose(result.se3.as_matrix(), reference_t_query.as_matrix())
        assert result.se3 == reference_t_query
