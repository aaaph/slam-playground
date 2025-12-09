from unittest.mock import Mock

import numpy as np
import pytest

from core.feature_tracker.image_preprocess import StereoImagePreprocess
from dataset.dataset_config import StereoConfig


class TestStereoImagePreprocess:
    """Unit test for stereo image preprocessing."""

    @pytest.fixture
    def image_preprocess(self) -> StereoImagePreprocess:
        """Create a stereo image preprocessing."""
        mock_stereo_config = Mock(spec=StereoConfig)
        return StereoImagePreprocess(mock_stereo_config)

    def test_image_preprocess_creation_and_properties(self, image_preprocess: StereoImagePreprocess):
        """Test that the image preprocessing can be created."""
        assert image_preprocess is not None
        assert hasattr(image_preprocess, "preprocess_stereo")
        assert callable(image_preprocess.preprocess_stereo)

    def test_image_preprocess_should_return_same_shape(self, mocker, image_preprocess: StereoImagePreprocess):
        """Test that the image preprocessing returns the same shape."""
        mock_remap = mocker.patch("cv2.remap")
        mock_remap.return_value = np.array([[5, 6], [7, 8]])
        mock_equalizehist = mocker.patch("cv2.equalizeHist")
        mock_equalizehist.return_value = np.array([[5, 6], [7, 8]])

        left_image = np.array([[1, 2], [3, 4]])
        right_image = np.array([[5, 6], [7, 8]])
        result_left, result_right = image_preprocess.preprocess_stereo(left_image, right_image)

        assert result_left.shape == left_image.shape
        assert result_right.shape == right_image.shape
        assert isinstance(result_left, np.ndarray)
