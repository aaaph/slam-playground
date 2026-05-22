from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from core.loop_closure.vpr_detector import VPRDetector, VPRDetectorConfig
from core.loop_closure.vpr_frame import VPRGeometrySchema

if TYPE_CHECKING:
    from core.camera_model.stereo_camera_ctx import StereoContext


class FakeORB:
    """Fake ORB detector."""

    def __init__(self) -> None:
        """Initialize the fake ORB detector."""
        self.detectAndCompute = Mock(
            return_value=(
                [cv2.KeyPoint(x=10.0, y=10.0, size=31.0)],
                np.ones((1, 32), dtype=np.uint8),
            )
        )


def make_region_features(region_idx: int, count: int) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """Make deterministic region features with descriptor ids tied to keypoint responses."""
    keypoints = [
        cv2.KeyPoint(
            float(i),
            float(region_idx),
            31.0,
            -1.0,
            float(i),
        )
        for i in range(count)
    ]
    ids = np.array([region_idx * 10 + i for i in range(count)], dtype=np.uint8)
    descriptors = np.repeat(ids[:, None], 32, axis=1)
    return keypoints, descriptors


def make_detector_config() -> VPRDetectorConfig:
    """Make a compact VPR detector config for unit tests."""
    return VPRDetectorConfig(
        stereo_k_matrix=np.eye(3, dtype=np.float32),
        resolution=(100, 100),
        grid_rows=2,
        grid_cols=2,
        region_limit=2,
        baseline=1.0,
        disparity_min_threshold=5.0,
        depth_min_threshold=0.15,
        depth_max_threshold=40.0,
        vertical_shift_threshold=10.0,
    )


class TestVPRDetector:
    """Unit tests for VPRDetector."""

    @pytest.fixture(autouse=True)
    def mock_stereo_klt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mock stereo KLT to return valid left-right correspondences."""
        call_index = 0

        def fake_calc_optical_flow_pyr_lk(
            _prev_img: np.ndarray,
            _next_img: np.ndarray,
            prev_points: np.ndarray,
            _next_points: np.ndarray | None = None,
            **_kwargs: object,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            nonlocal call_index
            points = np.ascontiguousarray(prev_points, dtype=np.float32)
            if call_index % 2 == 0:
                next_points = points - np.array([5.0, 0.0], dtype=np.float32)
            else:
                next_points = points + np.array([5.0, 0.0], dtype=np.float32)
            call_index += 1
            status = np.ones((points.shape[0], 1), dtype=np.uint8)
            error = np.zeros((points.shape[0], 1), dtype=np.float32)
            return next_points, status, error

        monkeypatch.setattr(
            "core.loop_closure.vpr_detector.cv2.calcOpticalFlowPyrLK",
            fake_calc_optical_flow_pyr_lk,
        )

    @pytest.fixture
    def cv2_detector(self) -> FakeORB:
        """Create a CV2 detector."""
        return FakeORB()

    @pytest.fixture
    def vpr_detector_config(self) -> VPRDetectorConfig:
        """Create a VPR detector configuration."""
        stereo_k_matrix = np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1]])
        resolution = (100, 100)
        return VPRDetectorConfig(
            stereo_k_matrix=stereo_k_matrix,
            resolution=resolution,
            grid_rows=2,
            grid_cols=2,
            region_limit=10,
            baseline=1.0,
            disparity_min_threshold=5.0,
            depth_min_threshold=0.15,
            depth_max_threshold=40.0,
            vertical_shift_threshold=10.0,
        )

    @pytest.fixture
    def detector(self, cv2_detector: FakeORB, vpr_detector_config: VPRDetectorConfig) -> VPRDetector:
        """Create a VPR detector."""
        return VPRDetector(cast("cv2.ORB", cv2_detector), vpr_detector_config)

    def test_should_be_possible_to_create(self, detector: VPRDetector):
        """Test that the VPR detector can be created."""
        assert detector is not None
        assert detector.detector is not None
        assert hasattr(detector, "detect_stereo")

    def test_from_stereo_ctx_should_split_feature_budget_over_grid(self, monkeypatch: pytest.MonkeyPatch):
        """Test that the factory applies the requested feature budget and grid."""
        create = Mock(return_value=cast("cv2.ORB", FakeORB()))
        monkeypatch.setattr("core.loop_closure.vpr_detector.cv2.ORB.create", create)
        stereo_ctx = SimpleNamespace(
            stereo_k=np.eye(3, dtype=np.float32),
            resolution=(100, 100),
            baseline=1.0,
        )

        detector = VPRDetector.from_stereo_ctx(
            cast("StereoContext", stereo_ctx),
            n_features=500,
            grid=(5, 4),
        )

        create.assert_called_once()
        assert create.call_args.kwargs["nfeatures"] == 25
        assert detector.config.region_limit == 25
        assert detector.config.grid_rows == 5
        assert detector.config.grid_cols == 4

    def test_should_create_grid_on_bootstrap(self, detector: VPRDetector, vpr_detector_config: VPRDetectorConfig):
        """Test that the VPR detector can create a grid on bootstrap."""
        assert detector.grid is not None
        assert len(detector.grid) == detector.config.grid_rows * detector.config.grid_cols
        for mask in detector.grid:
            assert mask is not None
            assert mask.shape == (detector.config.resolution[1], detector.config.resolution[0])
        shape = (vpr_detector_config.resolution[1], vpr_detector_config.resolution[0])
        all_mask_sum = np.ones(shape, dtype=np.uint8)
        for mask in detector.grid:
            all_mask_sum[mask == 1] = 0
        assert np.sum(all_mask_sum) == 0

    def test_detect_should_be_called_by_grid(self, detector: VPRDetector, cv2_detector: FakeORB):
        """Test that the VPR detector should call detect for each grid region."""
        grid_count = len(detector.grid)
        left_image = np.zeros((detector.config.resolution[1], detector.config.resolution[0]), dtype=np.uint8)
        right_image = np.zeros((detector.config.resolution[1], detector.config.resolution[0]), dtype=np.uint8)

        detector.detect_stereo(left_image, right_image)
        assert cv2_detector.detectAndCompute.call_count == grid_count

    def test_detect_should_limit_kps_per_region(self, cv2_detector: FakeORB):
        """Test that the VPR detector should limit the number of keypoints per region."""
        config = make_detector_config()
        detector = VPRDetector(cast("cv2.ORB", cv2_detector), config)

        cv2_detector.detectAndCompute.side_effect = [
            make_region_features(0, 5),
            make_region_features(1, 5),
            make_region_features(2, 5),
            make_region_features(3, 5),
        ]

        left_image = np.zeros((config.resolution[1], config.resolution[0]), dtype=np.uint8)
        right_image = np.zeros((config.resolution[1], config.resolution[0]), dtype=np.uint8)

        frame = detector.detect_stereo(left_image, right_image)

        assert frame.descriptors.shape == (8, 32)
        np.testing.assert_array_equal(
            frame.descriptors[:, 0],
            np.array([4, 3, 14, 13, 24, 23, 34, 33], dtype=np.uint8),
        )
        np.testing.assert_allclose(
            frame.geometry[:, VPRGeometrySchema.RIGHT_U],
            frame.geometry[:, VPRGeometrySchema.LEFT_U] - 5.0,
        )
        np.testing.assert_allclose(
            frame.geometry[:, VPRGeometrySchema.RIGHT_V],
            frame.geometry[:, VPRGeometrySchema.LEFT_V],
        )
        bearing = frame.geometry[:, VPRGeometrySchema.BEARING_X : VPRGeometrySchema.BEARING_Z + 1]
        np.testing.assert_allclose(np.linalg.norm(bearing, axis=1), 1.0, atol=1e-6)
        assert np.isfinite(frame.geometry[:, VPRGeometrySchema.POINT_X : VPRGeometrySchema.POINT_Z + 1]).all()

    def test_detect_with_empty_region(self, cv2_detector: FakeORB):
        """Test that the VPR detector should return an empty frame if the region is empty."""
        config = make_detector_config()
        detector = VPRDetector(cast("cv2.ORB", cv2_detector), config)

        cv2_detector.detectAndCompute.side_effect = [
            make_region_features(0, 5),
            ([], None),
            make_region_features(2, 1),
            make_region_features(3, 5),
        ]

        left_image = np.zeros((config.resolution[1], config.resolution[0]), dtype=np.uint8)
        right_image = np.zeros((config.resolution[1], config.resolution[0]), dtype=np.uint8)

        frame = detector.detect_stereo(left_image, right_image)

        assert frame.descriptors.shape == (5, 32)

    def test_triangulate_empty_geometry_should_keep_schema_width(self, detector: VPRDetector):
        """Test that empty triangulation preserves the geometry schema width."""
        geometry, mask = detector._triangulate_stereo_geometry(  # noqa: SLF001
            np.empty((0, VPRGeometrySchema.count()), dtype=np.float32)
        )

        assert geometry.shape == (0, VPRGeometrySchema.count())
        assert mask.shape == (0,)

    def test_triangulate_should_filter_invalid_stereo_geometry(self, cv2_detector: FakeORB):
        """Test that triangulation keeps only geometrically valid stereo points."""
        config = replace(
            make_detector_config(),
            disparity_min_threshold=0.5,
            depth_min_threshold=0.15,
            depth_max_threshold=40.0,
        )
        detector = VPRDetector(cast("cv2.ORB", cv2_detector), config)
        geometry = np.full((5, VPRGeometrySchema.count()), np.nan, dtype=np.float32)
        geometry[:, VPRGeometrySchema.LEFT_U] = np.array([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)
        geometry[:, VPRGeometrySchema.LEFT_V] = np.array([5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float32)
        geometry[:, VPRGeometrySchema.RIGHT_U] = np.array([5.0, 9.9, 0.0, 5.0, np.nan], dtype=np.float32)
        geometry[:, VPRGeometrySchema.RIGHT_V] = np.array([5.0, 5.0, 5.0, 20.1, 5.0], dtype=np.float32)

        triangulated, mask = detector._triangulate_stereo_geometry(geometry)  # noqa: SLF001

        np.testing.assert_array_equal(mask, np.array([True, False, False, False, False]))
        assert triangulated.shape == (1, VPRGeometrySchema.count())
        np.testing.assert_allclose(triangulated[0, VPRGeometrySchema.POINT_X], 2.0)
        np.testing.assert_allclose(triangulated[0, VPRGeometrySchema.POINT_Y], 1.0)
        np.testing.assert_allclose(triangulated[0, VPRGeometrySchema.POINT_Z], 0.2)
