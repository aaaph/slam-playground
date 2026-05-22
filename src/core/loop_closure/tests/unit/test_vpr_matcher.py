from typing import cast
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from core.loop_closure.vpr_matcher import VPRMatcher


class FakeBFMatcher:
    """Fake OpenCV BF matcher."""

    def __init__(self, knn_result: list[list[cv2.DMatch]]) -> None:
        """Initialize the fake matcher."""
        self.knnMatch = Mock(return_value=knn_result)


def make_match(query_idx: int, train_idx: int, distance: float) -> cv2.DMatch:
    """Make an OpenCV descriptor match."""
    return cv2.DMatch(_queryIdx=query_idx, _trainIdx=train_idx, _distance=distance)


class TestVPRMatcher:
    """Unit tests for VPRMatcher."""

    @pytest.fixture
    def query_descriptors(self) -> np.ndarray:
        """Create query descriptors."""
        return np.ones((3, 32), dtype=np.uint8)

    @pytest.fixture
    def train_descriptors(self) -> np.ndarray:
        """Create train descriptors."""
        return np.zeros((4, 32), dtype=np.uint8)

    def test_match_should_call_knn_match_with_configured_neighbor_count(
        self,
        query_descriptors: np.ndarray,
        train_descriptors: np.ndarray,
    ):
        """Test that VPRMatcher delegates KNN matching with the configured k."""
        fake_cv_matcher = FakeBFMatcher(knn_result=[])
        matcher = VPRMatcher(
            matcher=cast("cv2.BFMatcher", fake_cv_matcher),
            knn_neighbors=2,
            lowe_ratio=0.75,
        )

        matches = matcher.match(query_descriptors, train_descriptors)

        fake_cv_matcher.knnMatch.assert_called_once_with(
            query_descriptors,
            train_descriptors,
            k=2,
        )
        assert matches == []

    def test_default_factory_should_use_two_neighbors_for_lowe_ratio(self):
        """Test that default matching follows standard KNN-2 Lowe ratio behavior."""
        matcher = VPRMatcher.default_factory()

        assert matcher.knn_neighbors == 2
        assert matcher.lowe_ratio == pytest.approx(0.75)

    def test_match_should_apply_lowe_ratio_and_skip_incomplete_pairs(
        self,
        query_descriptors: np.ndarray,
        train_descriptors: np.ndarray,
    ):
        """Test that VPRMatcher keeps only complete pairs passing Lowe ratio."""
        accepted = make_match(query_idx=0, train_idx=2, distance=10.0)
        accepted_second = make_match(query_idx=0, train_idx=1, distance=20.0)
        rejected_by_ratio = make_match(query_idx=1, train_idx=0, distance=16.0)
        rejected_second = make_match(query_idx=1, train_idx=3, distance=20.0)
        incomplete = make_match(query_idx=2, train_idx=1, distance=1.0)
        fake_cv_matcher = FakeBFMatcher(
            knn_result=[
                [accepted, accepted_second],
                [rejected_by_ratio, rejected_second],
                [incomplete],
            ]
        )
        matcher = VPRMatcher(
            matcher=cast("cv2.BFMatcher", fake_cv_matcher),
            knn_neighbors=2,
            lowe_ratio=0.75,
        )

        matches = matcher.match(query_descriptors, train_descriptors)

        assert matches == [accepted]

    def test_match_should_only_require_two_neighbors_for_ratio_test(
        self,
        query_descriptors: np.ndarray,
        train_descriptors: np.ndarray,
    ):
        """Test that Lowe ratio accepts valid pairs even if fewer than configured neighbors return."""
        accepted = make_match(query_idx=0, train_idx=2, distance=10.0)
        accepted_second = make_match(query_idx=0, train_idx=1, distance=20.0)
        fake_cv_matcher = FakeBFMatcher(knn_result=[[accepted, accepted_second]])
        matcher = VPRMatcher(
            matcher=cast("cv2.BFMatcher", fake_cv_matcher),
            knn_neighbors=10,
            lowe_ratio=0.75,
        )

        matches = matcher.match(query_descriptors, train_descriptors)

        assert matches == [accepted]
