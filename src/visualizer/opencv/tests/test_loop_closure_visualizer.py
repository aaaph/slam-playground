import cv2
import numpy as np
import pytest

from core.loop_closure.vpr_frame import VPRFrame, VPRGeometrySchema
from core.loop_closure.vpr_verifier import VerifyResult
from core.transformations.special_euclidian_3_dim import SE3
from visualizer.opencv.loop_closure import LoopClosureOpenCVVisualizer, LoopClosureVisualizationConfig


def make_frame(frame_id: int, kf_id: int, left_uv: list[tuple[float, float]]) -> VPRFrame:
    """Create a minimal VPR frame with left image coordinates."""
    geometry = np.zeros((len(left_uv), VPRGeometrySchema.count()), dtype=np.float32)
    for index, (u, v) in enumerate(left_uv):
        geometry[index, VPRGeometrySchema.LEFT_U] = u
        geometry[index, VPRGeometrySchema.LEFT_V] = v
    return VPRFrame(
        frame_id=frame_id,
        kf_id=kf_id,
        timestamp=1.0,
        geometry=geometry,
        descriptors=np.zeros((len(left_uv), 32), dtype=np.uint8),
    )


def make_verify_result(match: cv2.DMatch) -> VerifyResult:
    """Create an accepted verification result with one inlier match."""
    return VerifyResult(
        query_id=2,
        reference_id=1,
        accepted=True,
        temporal_consistent=True,
        essential_consistent=True,
        geometric_consistent=True,
        history_depth=3,
        se3=SE3.identity(),
        essential_inliners_count=1,
        essntial_matches_count=1,
        geometric_inliners_count=1,
        geometric_matches_count=1,
        matches=[match],
        inlier_mask=np.array([True]),
    )


def test_draw_loop_image_should_render_verified_matches() -> None:
    """Loop visualizer should compose images and draw verified correspondences."""
    config = LoopClosureVisualizationConfig(match_point_radius=1)
    visualizer = LoopClosureOpenCVVisualizer(config)
    query_image = np.zeros((80, 100), dtype=np.uint8)
    reference_image = np.zeros((80, 100), dtype=np.uint8)
    query_frame = make_frame(frame_id=2, kf_id=20, left_uv=[(15.0, 20.0)])
    reference_frame = make_frame(frame_id=1, kf_id=10, left_uv=[(25.0, 20.0)])
    verify_result = make_verify_result(cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=0.0))

    loop_image = visualizer.draw_loop_image(
        query_image,
        reference_image,
        query_frame,
        reference_frame,
        verify_result,
    )

    assert loop_image.shape == (80, 200, 3)
    assert loop_image.dtype == np.uint8
    np.testing.assert_array_equal(loop_image[20, 15], np.array(config.match_inlier_color, dtype=np.uint8))
    np.testing.assert_array_equal(loop_image[20, 125], np.array(config.match_inlier_color, dtype=np.uint8))


def test_draw_loop_image_should_reject_unsupported_image_shape() -> None:
    """Loop visualizer should fail fast on unsupported image layout."""
    visualizer = LoopClosureOpenCVVisualizer()
    query_frame = make_frame(frame_id=2, kf_id=20, left_uv=[(15.0, 20.0)])
    reference_frame = make_frame(frame_id=1, kf_id=10, left_uv=[(25.0, 20.0)])
    verify_result = make_verify_result(cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=0.0))

    with pytest.raises(ValueError, match="Unsupported image shape"):
        visualizer.draw_loop_image(
            np.zeros((80, 100, 2), dtype=np.uint8),
            np.zeros((80, 100), dtype=np.uint8),
            query_frame,
            reference_frame,
            verify_result,
        )
