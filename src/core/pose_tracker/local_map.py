from collections import OrderedDict

import numpy as np
from numpy.typing import NDArray

FeatureId = int
Vector3d = NDArray[np.float32]


class LocalMap:
    """Local map."""

    def __init__(self, capacity: int = 1000) -> None:
        """Initialize the local map."""
        self.landmarks: OrderedDict[FeatureId, Vector3d] = OrderedDict()
        self.capacity = capacity

    def add_point(self, feat_id: FeatureId, point_3d: Vector3d) -> None:
        """Add a point to the local map and prune the oldest point if the capacity is exceeded."""
        self.landmarks[feat_id] = point_3d
        if len(self.landmarks) > self.capacity:
            self.landmarks.popitem(last=False)

    def add_points(self, new_points: dict[FeatureId, Vector3d]) -> None:
        """Add a point to the local map and prune the oldest point if the capacity is exceeded."""
        for feat_id, point_3d in new_points.items():
            self.add_point(feat_id, point_3d)

    def get_point(self, feat_id: FeatureId) -> Vector3d | None:
        """Get a point from the local map."""
        point = self.landmarks.get(feat_id, None)
        if point is not None:
            self.landmarks.move_to_end(feat_id)
        return point

    def exists(self, feat_id: FeatureId) -> bool:
        """Check if a point exists in the local map."""
        return feat_id in self.landmarks

    def empty(self) -> bool:
        """Check if the local map is empty."""
        return len(self.landmarks) == 0

    def clear(self) -> None:
        """Clear the local map."""
        self.landmarks.clear()

    @staticmethod
    def from_capacity(capacity: int) -> "LocalMap":
        """Create a local map from a capacity."""
        return LocalMap(capacity)
