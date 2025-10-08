from unittest.mock import Mock

import jax.numpy as jnp
import numpy as np

from core.feature_tracker.image_preprocess import StereoImagePreprocess
from dataset.dataset_config import StereoConfig


class TestUnitStereoImagePreprocess:
    """Unit test for stereo image preprocessing."""

    def test_should_be_possible_to_create(self):
        """Test that the image preprocessing can be created."""
        mock_stereo_config = Mock(spec=StereoConfig)

        image_preprocess = StereoImagePreprocess(mock_stereo_config)
        assert image_preprocess is not None

    def test_should_have_preprocess_stereo_method(self):
        """Test that the image preprocessing has a preprocess_stereo method."""
        mock_stereo_config = Mock(spec=StereoConfig)
        image_preprocess = StereoImagePreprocess(mock_stereo_config)
        assert hasattr(image_preprocess, "preprocess_stereo")

    def test_should_return_same_shape(self, mocker):
        """Test that the image preprocessing returns the same shape."""
        mock_stereo_config = Mock(spec=StereoConfig)
        image_preprocess = StereoImagePreprocess(mock_stereo_config)

        mock_remap = mocker.patch("cv2.remap")
        mock_remap.return_value = np.array([[5, 6], [7, 8]])
        mock_equalizehist = mocker.patch("cv2.equalizeHist")
        mock_equalizehist.return_value = np.array([[5, 6], [7, 8]])

        left_image = jnp.array([[1, 2], [3, 4]])
        right_image = jnp.array([[5, 6], [7, 8]])
        result_left, result_right = image_preprocess.preprocess_stereo(left_image, right_image)

        assert result_left.shape == left_image.shape
        assert result_right.shape == right_image.shape
