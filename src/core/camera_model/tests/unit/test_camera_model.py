import numpy as np
import pytest

from core.camera_model.stereo_camera_model import StereoCameraModel
from dataset.sensor_config import CameraSensor


class TestCameraModel:
    """Unit test for camera model."""

    @pytest.fixture
    def camera_model(self, cam_config_0: CameraSensor, cam_config_1: CameraSensor) -> StereoCameraModel:
        """Create a camera model."""
        return StereoCameraModel.from_cameras_config(cam_config_0, cam_config_1)

    def test_should_be_possible_to_create(self, camera_model: StereoCameraModel):
        """Test that the camera model can be created."""
        assert camera_model is not None
        assert camera_model.stereo_k is not None
        assert camera_model.baseline is not None

    def test_stereo_rectify(self, camera_model: StereoCameraModel):
        """Test that the stereo rectify method works."""
        resolution = camera_model.resolution
        width, height = resolution
        left_image = np.random.default_rng().random((height, width)).astype(np.uint8)
        right_image = np.random.default_rng().random((height, width)).astype(np.uint8)
        rectified_left, rectified_right = camera_model.process_stereo(left_image, right_image)
        assert rectified_left is not None
        assert rectified_right is not None
        assert rectified_left.shape == (height, width)
        assert rectified_right.shape == (height, width)
