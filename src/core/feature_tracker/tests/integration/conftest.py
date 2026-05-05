from pathlib import Path

import cv2
import numpy as np
import pytest

from core.camera_model.stereo_camera_ctx import StereoContext
from core.camera_model.stereo_camera_model import StereoCameraModel
from core.feature_tracker.feature_tracker import FeatureTracker, FeatureTrackerMode
from core.pose_tracker.feature_triangulation import FeatureTriangulation

from .config_helper import CAMERA_CONFIG_0, CAMERA_CONFIG_1


def _read_grayscale_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)

    return np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))


@pytest.fixture
def test_data_dir() -> Path:
    """Fixture for test data directory."""
    return Path(__file__).resolve().parent / "test_data"


@pytest.fixture(scope="session")
def camera_model() -> StereoCameraModel:
    """Fixture for camera model."""
    return StereoCameraModel.from_cameras_config(CAMERA_CONFIG_0, CAMERA_CONFIG_1)


@pytest.fixture(scope="session")
def stereo_ctx(camera_model: StereoCameraModel) -> StereoContext:
    """Fixture for stereo context."""
    return camera_model.as_stereo_ctx()


@pytest.fixture
def feature_tracker(stereo_ctx: StereoContext) -> FeatureTracker:
    """Fixture for feature tracker."""
    return FeatureTracker.default_factory(
        stereo_ctx, feat_amount_per_region=20, feat_retrack_threshold=5, mode=FeatureTrackerMode.STEREO
    )


@pytest.fixture
def feature_triangulator(stereo_ctx: StereoContext) -> FeatureTriangulation:
    """Fixture for feature triangulator."""
    return FeatureTriangulation.from_stereo_camera_ctx(stereo_ctx)


@pytest.fixture
def stereo_frame(test_data_dir: Path, camera_model: StereoCameraModel) -> tuple[np.ndarray, np.ndarray]:
    """Fixture for stereo frame."""
    testing_image_left = _read_grayscale_image(test_data_dir / "testing_image_left.png")
    testing_image_right = _read_grayscale_image(test_data_dir / "testing_image_right.png")
    left, right = camera_model.process_stereo(testing_image_left, testing_image_right)
    return left, right
