import cv2
import numpy as np
from numpy.typing import NDArray

from dataset.dataset_config import StereoConfig

Matrix = NDArray[np.float32]


class StereoImagePreprocess:
    """Stereo image preprocessing."""

    def __init__(self, stereo_config: StereoConfig) -> None:
        """Initialize the stereo image preprocessing."""
        self.stereo_config = stereo_config

    def preprocess_stereo(self, left_image: Matrix, right_image: Matrix) -> tuple[Matrix, Matrix]:
        """Preprocess the stereo images."""
        left_image = np.array(left_image)
        right_image = np.array(right_image)

        left_image = cv2.equalizeHist(left_image)
        right_image = cv2.equalizeHist(right_image)

        left_image = cv2.remap(
            left_image, self.stereo_config.left_map_x, self.stereo_config.left_map_y, cv2.INTER_LINEAR
        )
        right_image = cv2.remap(
            right_image, self.stereo_config.right_map_x, self.stereo_config.right_map_y, cv2.INTER_LINEAR
        )
        return left_image, right_image
