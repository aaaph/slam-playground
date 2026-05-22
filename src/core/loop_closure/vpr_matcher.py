import cv2
import numpy as np

LOWE_RATIO_NEIGHBOR_COUNT = 2


class VPRMatcher:
    """VPR matcher."""

    def __init__(self, matcher: cv2.BFMatcher, knn_neighbors: int, lowe_ratio: float) -> None:
        """Initialize the VPR matcher."""
        self.matcher = matcher
        self.knn_neighbors = knn_neighbors
        self.lowe_ratio = lowe_ratio

    @classmethod
    def default_factory(cls, knn_neighbors: int = 2, lowe_ratio: float = 0.75) -> "VPRMatcher":
        """Create a default VPR matcher."""
        return cls(
            matcher=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False),
            knn_neighbors=knn_neighbors,
            lowe_ratio=lowe_ratio,
        )

    def match(self, query_descriptors: np.ndarray, reference_descriptors: np.ndarray) -> list[cv2.DMatch]:
        """Match the descriptors."""
        knn_matches = self.matcher.knnMatch(
            query_descriptors,
            reference_descriptors,
            k=self.knn_neighbors,
        )
        return [
            pair[0]
            for pair in knn_matches
            if len(pair) >= LOWE_RATIO_NEIGHBOR_COUNT and pair[0].distance < self.lowe_ratio * pair[1].distance
        ]
