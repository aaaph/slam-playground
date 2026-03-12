import numpy as np
from numpy.typing import NDArray

from core.feature_tracker.feature import FeatureStatus


class FeatureFrame:
    """Feature frame wrapper over the feature slice."""

    def __init__(
        self,
        data: NDArray[np.float32],
        active_indeces: NDArray[np.int32],
        active_mask: NDArray[np.bool_],
        timestamp: float,
    ) -> None:
        """Initialize the active features."""
        self.data = data
        self.active_indeces = active_indeces
        self.active_mask = active_mask
        self.timestamp = timestamp

    @property
    def ndarray(self) -> NDArray[np.float32]:
        """Get the active features as a numpy array."""
        if self.active_indeces.size == 0:
            return np.empty((0, 8), dtype=np.float32)
        return self.data[self.active_mask]

    @property
    def left_points(self) -> NDArray[np.float32]:
        """Get the left points."""
        return self.ndarray[:, 2:4]

    @property
    def right_points(self) -> NDArray[np.float32]:
        """Get the right points."""
        return self.ndarray[:, 4:6]

    @property
    def ids(self) -> NDArray[np.int32]:
        """Get the ids of the active features."""
        return self.ndarray[:, 0]

    def good_features(self) -> NDArray[np.float32]:
        """Get the good features."""
        return self.ndarray[self.ndarray[:, 6] != FeatureStatus.LOST.value]

    def lost_features(self) -> NDArray[np.float32]:
        """Get the lost features."""
        return self.ndarray[self.ndarray[:, 6] == FeatureStatus.LOST.value]

    def count(self) -> int:
        """Get the count of the active features."""
        return self.active_indeces.size

    def __repr__(self) -> str:
        """Return the string representation of the feature frame."""
        ts = self.timestamp
        data_shape = self.data.shape
        indeces_shape = self.active_indeces.shape
        mask_shape = self.active_mask.shape
        return f"FeatureFrame(ts={ts}, data_shape={data_shape}, indeces_shape={indeces_shape}, mask_shape={mask_shape})"  # noqa: E501

    def copy(self) -> "FeatureFrame":
        """Copy the feature frame."""
        return FeatureFrame(
            data=self.data.copy(),
            active_indeces=self.active_indeces.copy(),
            active_mask=self.active_mask.copy(),
            timestamp=self.timestamp,
        )
