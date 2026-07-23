import numpy as np
import pytest

from core.pose_tracker.hypothesis_estimator import (
    CandidateHypothesisStatus,
    HypothesisEstimator,
)
from core.pose_tracker.local_map import CandidateHistorySchema


def make_candidate_history_row(
    feat_id: int,
    timestamp_ns: float,
    xyz: list[float],
    covariance_diag: tuple[float, float, float] = (0.01, 0.01, 0.04),
    cam_xyz: list[float] | None = None,
) -> np.ndarray:
    row = np.full(CandidateHistorySchema.count(), np.nan, dtype=np.float64)
    row[CandidateHistorySchema.FEAT_ID] = feat_id
    row[CandidateHistorySchema.TIMESTAMP_NS] = timestamp_ns
    row[CandidateHistorySchema.XYZ] = xyz
    row[CandidateHistorySchema.COV] = np.diag(covariance_diag).reshape(9)
    row[CandidateHistorySchema.DEPTH_SIGMA] = np.sqrt(covariance_diag[2])
    row[CandidateHistorySchema.CAM_XYZ] = xyz if cam_xyz is None else cam_xyz
    return row


def make_history(points: list[list[float]]) -> np.ndarray:
    return np.array(
        [
            make_candidate_history_row(
                feat_id=7,
                timestamp_ns=float(timestamp_idx),
                xyz=point,
            )
            for timestamp_idx, point in enumerate(points)
        ],
        dtype=np.float64,
    )


class TestHypothesisEstimator:
    """Unit tests for frontend candidate hypothesis estimation."""

    @pytest.fixture
    def estimator(self) -> HypothesisEstimator:
        """Create a candidate-history hypothesis estimator."""
        return HypothesisEstimator(CandidateHistorySchema)

    def test_stable_history_promotes_candidate(self, estimator: HypothesisEstimator) -> None:
        """Test that a consistent five-observation history becomes mature."""
        points = [
            [1.00, 2.00, 3.00],
            [1.01, 1.99, 3.02],
            [0.99, 2.01, 2.98],
            [1.00, 2.02, 3.01],
            [1.02, 2.00, 2.99],
        ]

        hypothesis = estimator.estimate(make_history(points))

        assert hypothesis.status == CandidateHypothesisStatus.STABLE
        assert hypothesis.inlier_count == 5
        assert hypothesis.observation_count == 5
        assert hypothesis.pnp_eligible
        assert hypothesis.promote_to_mature
        assert hypothesis.health_delta > 0.0
        np.testing.assert_array_equal(hypothesis.inlier_mask, np.array([True, True, True, True, True]))
        np.testing.assert_allclose(hypothesis.xyz, np.mean(points, axis=0), atol=0.02)
        assert hypothesis.covariance[2, 2] < 0.04

    def test_weak_history_uses_dominant_cluster_without_promoting(
        self,
        estimator: HypothesisEstimator,
    ) -> None:
        """Test that a dominant three-observation cluster remains weak but usable."""
        history = make_history(
            [
                [1.00, 2.00, 3.00],
                [1.01, 2.00, 3.01],
                [0.99, 2.01, 2.99],
                [1.00, 2.00, 6.00],
                [1.00, 2.00, 8.00],
            ]
        )

        hypothesis = estimator.estimate(history)

        assert hypothesis.status == CandidateHypothesisStatus.WEAK_STABLE
        assert hypothesis.inlier_count == 3
        assert hypothesis.pnp_eligible
        assert not hypothesis.promote_to_mature
        np.testing.assert_array_equal(hypothesis.inlier_mask, np.array([True, True, True, False, False]))
        np.testing.assert_allclose(hypothesis.xyz, np.array([1.0, 2.0, 3.0]), atol=0.02)

    def test_equal_competing_clusters_are_ambiguous(self, estimator: HypothesisEstimator) -> None:
        """Test that two similarly strong separated clusters are marked ambiguous."""
        history = make_history(
            [
                [1.00, 2.00, 3.00],
                [1.01, 2.00, 3.01],
                [1.00, 2.00, 6.00],
                [1.01, 2.00, 6.01],
                [1.00, 2.00, 12.00],
            ]
        )

        hypothesis = estimator.estimate(history)

        assert hypothesis.status == CandidateHypothesisStatus.AMBIGUOUS
        assert hypothesis.inlier_count == 2
        assert not hypothesis.pnp_eligible
        assert not hypothesis.promote_to_mature
        assert hypothesis.health_delta < 0.0

    def test_scattered_history_is_not_pnp_eligible(self, estimator: HypothesisEstimator) -> None:
        """Test that unrelated observations are treated as a scattered history."""
        history = make_history(
            [
                [1.0, 2.0, 1.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 5.0],
                [1.0, 2.0, 7.0],
                [1.0, 2.0, 9.0],
            ]
        )

        hypothesis = estimator.estimate(history)

        assert hypothesis.status == CandidateHypothesisStatus.SCATTERED
        assert hypothesis.inlier_count == 1
        assert not hypothesis.pnp_eligible
        assert not hypothesis.promote_to_mature
        assert hypothesis.health_delta < 0.0
        np.testing.assert_array_equal(hypothesis.inlier_mask, np.array([True, False, False, False, False]))

    def test_single_observation_is_provisional_but_pnp_eligible(self, estimator: HypothesisEstimator) -> None:
        """Test that the first observation can seed early PnP without maturity."""
        history = make_history([[1.0, 2.0, 3.0]])

        hypothesis = estimator.estimate(history)

        assert hypothesis.status == CandidateHypothesisStatus.PROVISIONAL
        assert hypothesis.inlier_count == 1
        assert hypothesis.pnp_eligible
        assert not hypothesis.promote_to_mature
        np.testing.assert_array_equal(hypothesis.inlier_mask, np.array([True]))
