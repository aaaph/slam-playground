from typing import cast

import gtsam
import numpy as np
import pytest

from core.camera_model.vio_context import VioContext
from core.feature_tracker.feature_schema import FeatureLifecycle
from core.front_end.keyframe_selector import SelectReason
from core.graph_optimizer import smart_factor_vio_optimizer as smart_factor_module
from core.graph_optimizer.optimizer_types import (
    PredictionMode,
    SmartStereoProjectionPoseFactor,
    VioKeyframe,
)
from core.graph_optimizer.smart_factor_vio_optimizer import SmartFactorVIOOptimizer
from core.pose_tracker.feature_triangulation import StereoTriangulationSchema, StereoTriangulationStatus
from core.transformations.special_euclidian_3_dim import SE3

X = gtsam.symbol_shorthand.X


def make_keyframe(keyframe_id: int, timestamp: float, feat_id: int) -> VioKeyframe:
    """Create one static keyframe observing the same stereo feature."""
    stereo_frame = np.full((1, StereoTriangulationSchema.count()), np.nan, dtype=np.float32)
    stereo_frame[0, StereoTriangulationSchema.FEAT_ID] = feat_id
    stereo_frame[0, StereoTriangulationSchema.LEFT_U] = 100.0
    stereo_frame[0, StereoTriangulationSchema.LEFT_V] = 50.0
    stereo_frame[0, StereoTriangulationSchema.RIGHT_U] = 50.0
    stereo_frame[0, StereoTriangulationSchema.RIGHT_V] = 50.0
    stereo_frame[0, StereoTriangulationSchema.LIFECYCLE] = FeatureLifecycle.ACTIVE.value
    stereo_frame[0, StereoTriangulationSchema.STEREO_STATUS] = StereoTriangulationStatus.TRIANGULATED.value

    imu_batch = np.empty((0, 8), dtype=np.float64)
    if keyframe_id > 0:
        imu_batch = np.array([[timestamp, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0, 0.01]], dtype=np.float64)

    return VioKeyframe(
        keyframe_id=keyframe_id,
        select_reason=[SelectReason.STATIC_INITIALIZATION],
        timestamp=timestamp,
        stereo_frame=stereo_frame,
        imu_batch=imu_batch,
        prediction_mode=PredictionMode.PNP,
        pose_guess=SE3.identity(),
        velocity_guess=np.zeros(3, dtype=np.float32),
        bias_guess=np.zeros(6, dtype=np.float32),
    )


@pytest.fixture
def track_crossing_sliding_window(
    vio_ctx: VioContext,
) -> tuple[SmartFactorVIOOptimizer, int, int]:
    """Run X0 -> X1 -> X2 with X0 outside the fixed-lag window."""
    optimizer = SmartFactorVIOOptimizer.from_vio_ctx(vio_ctx, lag=10.0)
    feat_id = 42

    for keyframe_id, timestamp in ((0, 0.0), (1, 5.0)):
        keyframe = make_keyframe(keyframe_id, timestamp, feat_id)
        optimizer.apply_subgraph(optimizer.keyframe_to_subgraph(keyframe))

    old_factor_slot = optimizer.smart_factors[feat_id]
    keyframe = make_keyframe(2, 11.0, feat_id)
    optimizer.apply_subgraph(optimizer.keyframe_to_subgraph(keyframe))
    return optimizer, feat_id, old_factor_slot


def test_optimizer_exposes_optimized_inertial_state(vio_ctx: VioContext) -> None:
    """The backend output must come from the smoother instead of a zero mock."""
    optimizer = SmartFactorVIOOptimizer.from_vio_ctx(vio_ctx)
    pose = SE3.from_quat_and_translation(np.array([0.0, 0.0, 0.0, 1.0]), np.array([1.0, 2.0, 3.0]))
    keyframe = VioKeyframe(
        keyframe_id=7,
        select_reason=[SelectReason.STATIC_INITIALIZATION],
        timestamp=10.0,
        stereo_frame=np.empty((0, StereoTriangulationSchema.count()), dtype=np.float32),
        imu_batch=np.empty((0, 8), dtype=np.float64),
        prediction_mode=PredictionMode.PNP,
        pose_guess=pose,
        velocity_guess=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        bias_guess=np.zeros(6, dtype=np.float32),
    )

    optimizer.apply_subgraph(optimizer.keyframe_to_subgraph(keyframe))

    np.testing.assert_allclose(optimizer.get_nav_state().pose().matrix(), pose.as_matrix())
    np.testing.assert_allclose(optimizer.get_nav_state().velocity(), keyframe.velocity_guess)
    np.testing.assert_allclose(optimizer.get_actual_bias_ndarray(), keyframe.bias_guess)
    assert optimizer.post_fit_avg_error() == 0.0


def test_smart_factor_disappears_when_x0_leaves_sliding_window(
    track_crossing_sliding_window: tuple[SmartFactorVIOOptimizer, int, int],
) -> None:
    """The factor containing X0 must be marginalized with X0."""
    optimizer, _, old_factor_slot = track_crossing_sliding_window

    assert not optimizer.result.exists(X(0))
    assert optimizer.smoother.getFactors().at(old_factor_slot) is None


def test_feature_starts_new_smart_factor_segment_at_x2(
    track_crossing_sliding_window: tuple[SmartFactorVIOOptimizer, int, int],
) -> None:
    """A still-visible feature must restart as a pending observation at X2."""
    optimizer, feat_id, _ = track_crossing_sliding_window

    assert feat_id not in optimizer.smart_factors
    assert [measurement.pose_key for measurement in optimizer.measurement_history[feat_id]] == [X(2)]


def test_smart_factor_is_added_on_second_measurement(vio_ctx: VioContext) -> None:
    """The first observation stays pending until another pose observes the feature."""
    optimizer = SmartFactorVIOOptimizer.from_vio_ctx(vio_ctx, lag=10.0)
    feat_id = 42

    optimizer.apply_subgraph(optimizer.keyframe_to_subgraph(make_keyframe(0, 0.0, feat_id)))

    assert feat_id not in optimizer.smart_factors
    assert [measurement.pose_key for measurement in optimizer.measurement_history[feat_id]] == [X(0)]

    optimizer.apply_subgraph(optimizer.keyframe_to_subgraph(make_keyframe(1, 5.0, feat_id)))

    factor = cast(
        "SmartStereoProjectionPoseFactor",
        optimizer.smoother.getFactors().at(optimizer.smart_factors[feat_id]),
    )
    assert factor.keys() == [X(0), X(1)]


def test_reprojection_gate_keeps_common_shift_and_rejects_isolated_outlier(vio_ctx: VioContext) -> None:
    """A frame-relative gate must preserve common pose error and reject only its gross tail."""
    optimizer = SmartFactorVIOOptimizer.from_vio_ctx(vio_ctx, lag=20.0)
    feat_ids = np.array([42, 43, 44])

    for keyframe_id in range(3):
        keyframe = make_keyframe(keyframe_id, float(keyframe_id), int(feat_ids[0]))
        stereo_frame = np.repeat(keyframe.stereo_frame, feat_ids.size, axis=0)
        stereo_frame[:, StereoTriangulationSchema.FEAT_ID] = feat_ids
        optimizer.apply_subgraph(optimizer.keyframe_to_subgraph(keyframe._replace(stereo_frame=stereo_frame)))

    slots = optimizer.smart_factors.copy()
    keyframe = make_keyframe(3, 3.0, int(feat_ids[0]))
    stereo_frame = np.repeat(keyframe.stereo_frame, feat_ids.size, axis=0)
    stereo_frame[:, StereoTriangulationSchema.FEAT_ID] = feat_ids
    stereo_frame[:, [StereoTriangulationSchema.LEFT_U, StereoTriangulationSchema.RIGHT_U]] += 25.0
    stereo_frame[-1, [StereoTriangulationSchema.LEFT_U, StereoTriangulationSchema.RIGHT_U]] += 300.0
    optimizer.apply_subgraph(optimizer.keyframe_to_subgraph(keyframe._replace(stereo_frame=stereo_frame)))

    assert optimizer.smart_factors[42] != slots[42]
    assert optimizer.smart_factors[43] != slots[43]
    assert optimizer.smart_factors[44] == slots[44]
    assert [measurement.pose_key for measurement in optimizer.measurement_history[44]] == [X(0), X(1), X(2)]


def test_post_fit_quarantine_removes_factor_on_next_update(
    vio_ctx: VioContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A quarantined smart factor must leave both the smoother and measurement history."""
    monkeypatch.setattr(smart_factor_module, "SMART_POST_FIT_QUARANTINE_RMSE", 0.0)
    optimizer = SmartFactorVIOOptimizer.from_vio_ctx(vio_ctx, lag=20.0)
    feat_id = 42
    optimizer.apply_subgraph(optimizer.keyframe_to_subgraph(make_keyframe(0, 0.0, feat_id)))

    inconsistent = make_keyframe(1, 1.0, feat_id)
    inconsistent.stereo_frame[0, StereoTriangulationSchema.LEFT_U] += 5.0
    optimizer.apply_subgraph(optimizer.keyframe_to_subgraph(inconsistent))
    slot = optimizer.smart_factors[feat_id]

    assert feat_id in optimizer.quarantined_feat_ids

    optimizer.apply_subgraph(optimizer.keyframe_to_subgraph(make_keyframe(2, 2.0, feat_id)))

    assert optimizer.smoother.getFactors().at(slot) is None
    assert feat_id not in optimizer.smart_factors
    assert feat_id not in optimizer.measurement_history


def test_lost_feature_drops_pending_measurement(vio_ctx: VioContext) -> None:
    """A lost feature with one observation must leave no pending graph state."""
    optimizer = SmartFactorVIOOptimizer.from_vio_ctx(vio_ctx, lag=10.0)
    feat_id = 42
    optimizer.apply_subgraph(optimizer.keyframe_to_subgraph(make_keyframe(0, 0.0, feat_id)))

    lost_keyframe = make_keyframe(1, 5.0, feat_id)
    lost_keyframe.stereo_frame[0, StereoTriangulationSchema.LIFECYCLE] = FeatureLifecycle.LOST.value
    optimizer.apply_subgraph(optimizer.keyframe_to_subgraph(lost_keyframe))

    assert feat_id not in optimizer.measurement_history
    assert feat_id not in optimizer.smart_factors


def test_missing_active_feature_drops_pending_measurement(vio_ctx: VioContext) -> None:
    """A pending observation must be dropped when the next keyframe omits its feature."""
    optimizer = SmartFactorVIOOptimizer.from_vio_ctx(vio_ctx, lag=10.0)
    feat_id = 42
    optimizer.apply_subgraph(optimizer.keyframe_to_subgraph(make_keyframe(0, 0.0, feat_id)))

    keyframe = make_keyframe(1, 5.0, feat_id)._replace(
        stereo_frame=np.empty((0, StereoTriangulationSchema.count()), dtype=np.float32)
    )
    optimizer.apply_subgraph(optimizer.keyframe_to_subgraph(keyframe))

    assert feat_id not in optimizer.measurement_history
    assert feat_id not in optimizer.smart_factors


def test_smart_factor_fit_metric_is_normalized_by_track_dof(vio_ctx: VioContext) -> None:
    """Smart-factor RMSE must normalize its whitened error by 3N - 3."""
    optimizer = SmartFactorVIOOptimizer.from_vio_ctx(vio_ctx, lag=10.0)
    feat_id = 42
    first_keyframe = make_keyframe(0, 0.0, feat_id)
    optimizer.apply_subgraph(optimizer.keyframe_to_subgraph(first_keyframe))

    second_keyframe = make_keyframe(1, 5.0, feat_id)
    second_keyframe.stereo_frame[0, StereoTriangulationSchema.LEFT_U] += 5.0
    optimizer.apply_subgraph(optimizer.keyframe_to_subgraph(second_keyframe))

    factor = optimizer.smoother.getFactors().at(optimizer.smart_factors[feat_id])
    expected_rmse = np.sqrt(2.0 * factor.error(optimizer.result) / 3.0)

    assert optimizer.smart_factor_whitened_rmse == pytest.approx(expected_rmse)
    assert optimizer.smart_factor_max_whitened_rmse == pytest.approx(expected_rmse)
    assert optimizer.smart_factor_error_ratio == pytest.approx(
        factor.error(optimizer.result) / optimizer.post_fit_error
    )
