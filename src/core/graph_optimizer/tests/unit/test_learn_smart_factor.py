from collections import deque
from typing import TYPE_CHECKING, Any, NamedTuple, SupportsInt, cast

import gtsam
import gtsam_unstable
import numpy as np
import pytest
from numpy.typing import NDArray

from core.camera_model.stereo_camera_model import StereoCameraModel
from core.front_end.keyframe import ActiveTrackSchema

if TYPE_CHECKING:
    from collections.abc import Sequence

SmartStereoProjectionPoseFactor: Any = getattr(gtsam_unstable, "SmartStereoProjectionPoseFactor")  # noqa: B009
X = gtsam.symbol_shorthand.X
L = gtsam.symbol_shorthand.L


class TestSmartVioHypothesis:
    """Learning how to work with gtsam smart factors."""

    def test_find_factors_in_graph(
        self, first_active_track: NDArray[np.float32], camera_model: StereoCameraModel
    ) -> None:
        """Learning how to find smart factor in the graph."""
        smoother = gtsam.IncrementalFixedLagSmoother(10.0)

        stereo_ctx = camera_model.as_stereo_ctx()
        smart_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
        smart_params = gtsam.SmartProjectionParams()
        prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-4)
        prior_pose = gtsam.Pose3(gtsam.Rot3.Identity(), gtsam.Point3(0, 0, 0))
        prior_factor = gtsam.PriorFactorPose3(X(0), prior_pose, prior_noise)

        new_factors = gtsam.NonlinearFactorGraph()
        new_factors.add(prior_factor)

        for track_item in first_active_track:
            right_u = track_item[ActiveTrackSchema.RIGHT_U]
            left_u = track_item[ActiveTrackSchema.LEFT_U]
            left_v = track_item[ActiveTrackSchema.LEFT_V]
            if np.isnan(right_u):
                continue
            stereo_point = gtsam.StereoPoint2(left_u, right_u, left_v)
            smart_stereo_factor = SmartStereoProjectionPoseFactor(
                sharedNoiseModel=smart_noise,
                params=smart_params,
                body_P_sensor=stereo_ctx.cam0_in_body_se3.as_gtsam_pose(),
            )
            smart_stereo_factor.add(stereo_point, X(0), stereo_ctx.stereo_k_gtsam)
            new_factors.add(smart_stereo_factor)

        if new_factors.empty():
            pytest.fail("No factors added to the graph")
        new_values = gtsam.Values()
        new_timestamps = gtsam.FixedLagSmootherKeyTimestampMap()

        new_values.insert(X(0), gtsam.Pose3(gtsam.Rot3.Identity(), gtsam.Point3(0, 0, 0)))
        new_timestamps.insert((X(0), 0.0))

        smoother.update(new_factors, new_values, new_timestamps)

        result = smoother.calculateEstimate()
        assert result.exists(X(0))
        optimized_pose = result.atPose3(X(0))
        assert optimized_pose.equals(prior_pose, 1e-4)
        # factors exists in slots -> in 0 index we have prior factor, the 0 smart factor lives in index 1
        # need to have a control dictionary to know the place of smart factors in actual state
        feat_0_factor = cast("SmartStereoProjectionPoseFactor", smoother.getFactors().at(1))
        point_result = feat_0_factor.point(result)

        point_valid = point_result.valid()
        point_value = point_result.get()
        assert point_valid
        assert point_value.shape == (3,)

    def test_replace_factors_in_graph(  # noqa: PLR0915
        self, first_active_track: NDArray[np.float32], camera_model: StereoCameraModel
    ) -> None:
        """Learning how to replace smart factors in the graph."""
        smoother = gtsam.IncrementalFixedLagSmoother(10.0)

        stereo_ctx = camera_model.as_stereo_ctx()
        smart_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
        smart_params = gtsam.SmartProjectionParams()
        prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-4)
        prior_pose = gtsam.Pose3(gtsam.Rot3.Identity(), gtsam.Point3(0, 0, 0))
        prior_factor = gtsam.PriorFactorPose3(X(0), prior_pose, prior_noise)

        factors_feat_id_to_index = {}
        factors_index_shift = 0
        new_factors = gtsam.NonlinearFactorGraph()
        new_factors.add(prior_factor)
        factors_index_shift += 1
        for track_item in first_active_track:
            right_u = track_item[ActiveTrackSchema.RIGHT_U]
            left_u = track_item[ActiveTrackSchema.LEFT_U]
            left_v = track_item[ActiveTrackSchema.LEFT_V]
            feat_id = int(track_item[ActiveTrackSchema.FEAT_ID])
            if np.isnan(right_u):
                continue
            stereo_point = gtsam.StereoPoint2(left_u, right_u, left_v)
            smart_stereo_factor = SmartStereoProjectionPoseFactor(
                sharedNoiseModel=smart_noise,
                params=smart_params,
                body_P_sensor=stereo_ctx.cam0_in_body_se3.as_gtsam_pose(),
            )
            smart_stereo_factor.add(stereo_point, X(0), stereo_ctx.stereo_k_gtsam)
            new_factors.add(smart_stereo_factor)
            factors_feat_id_to_index[int(feat_id)] = factors_index_shift
            factors_index_shift += 1

        if new_factors.empty():
            pytest.fail("No factors added to the graph")
        new_values = gtsam.Values()
        new_timestamps = gtsam.FixedLagSmootherKeyTimestampMap()

        new_values.insert(X(0), gtsam.Pose3(gtsam.Rot3.Identity(), gtsam.Point3(0, 0, 0)))
        new_timestamps.insert((X(0), 0.0))

        smoother.update(new_factors, new_values, new_timestamps)
        _ = smoother.calculateEstimate()
        next_active_track = first_active_track.copy()
        feat_zero_row = next_active_track[0].copy()
        feat_zero_row[ActiveTrackSchema.LEFT_U] = feat_zero_row[ActiveTrackSchema.LEFT_U] - 0.5
        feat_zero_row[ActiveTrackSchema.LEFT_V] = feat_zero_row[ActiveTrackSchema.LEFT_V] + 0.5
        feat_zero_row[ActiveTrackSchema.RIGHT_U] = feat_zero_row[ActiveTrackSchema.RIGHT_U] - 0.5
        feat_zero_row[ActiveTrackSchema.RIGHT_V] = feat_zero_row[ActiveTrackSchema.RIGHT_V] + 0.5
        next_active_track[0] = feat_zero_row

        next_factors = gtsam.NonlinearFactorGraph()
        # we know that factors are from index 1 to
        delete_slots: Sequence[SupportsInt] = []
        between_keyframe_factor = gtsam.BetweenFactorPose3(
            X(0), X(1), gtsam.Pose3(gtsam.Rot3.Identity(), gtsam.Point3(0, 0, 0)), prior_noise
        )
        factors_index_shift += 1

        next_factors.add(between_keyframe_factor)

        feat_zero_id = int(feat_zero_row[ActiveTrackSchema.FEAT_ID])
        feat_zero_slot = factors_feat_id_to_index[feat_zero_id]
        delete_slots.append(feat_zero_slot)
        # Need a sidecar to recreate a smart factor
        prev_stereo_point = gtsam.StereoPoint2(
            first_active_track[0][ActiveTrackSchema.LEFT_U],
            first_active_track[0][ActiveTrackSchema.RIGHT_U],
            first_active_track[0][ActiveTrackSchema.LEFT_V],
        )
        next_stereo_point = gtsam.StereoPoint2(
            feat_zero_row[ActiveTrackSchema.LEFT_U],
            feat_zero_row[ActiveTrackSchema.RIGHT_U],
            feat_zero_row[ActiveTrackSchema.LEFT_V],
        )
        feat_zero_factor = SmartStereoProjectionPoseFactor(
            sharedNoiseModel=smart_noise,
            params=smart_params,
            body_P_sensor=stereo_ctx.cam0_in_body_se3.as_gtsam_pose(),
        )
        feat_zero_factor.add(prev_stereo_point, X(0), stereo_ctx.stereo_k_gtsam)
        feat_zero_factor.add(next_stereo_point, X(1), stereo_ctx.stereo_k_gtsam)
        next_factors.add(feat_zero_factor)
        factors_feat_id_to_index[feat_zero_id] = factors_index_shift
        factors_index_shift += 1

        next_timestamps = gtsam.FixedLagSmootherKeyTimestampMap()
        next_values = gtsam.Values()
        next_values.insert(X(1), gtsam.Pose3(gtsam.Rot3.Identity(), gtsam.Point3(0, 0, 0)))
        next_timestamps.insert((X(1), 1.0))
        smoother.update(next_factors, next_values, next_timestamps, delete_slots)
        result = smoother.calculateEstimate()
        prev_zero_factor = smoother.getFactors().at(1)
        assert prev_zero_factor is None
        assert factors_feat_id_to_index[feat_zero_id] != 1
        assert factors_feat_id_to_index[feat_zero_id] == factors_index_shift - 1

        zero_feat_slot = factors_feat_id_to_index[feat_zero_id]
        stored_smart_factor = smoother.getFactors().at(zero_feat_slot)
        assert stored_smart_factor is not None
        assert stored_smart_factor.keys() == [X(0), X(1)]

        triang_result = stored_smart_factor.point(result)
        assert triang_result.valid()

    def test_deque_sidecar_of_smart_factor(  # noqa: PLR0915
        self, first_active_track: NDArray[np.float32], camera_model: StereoCameraModel
    ) -> None:
        """Testing of dequeing sidecar of smart factor."""

        class SmartObs(NamedTuple):
            """Smart observation."""

            pose_key: int
            ul: float
            ur: float
            v: float

        smoother = gtsam.IncrementalFixedLagSmoother(10.0)

        stereo_ctx = camera_model.as_stereo_ctx()
        smart_noise = gtsam.noiseModel.Isotropic.Sigma(3, 3.0)
        smart_params = gtsam.SmartProjectionParams()
        smart_params.setDegeneracyMode(gtsam.DegeneracyMode.ZERO_ON_DEGENERACY)
        smart_params.setRankTolerance(1.0)
        smart_params.setLinearizationMode(gtsam.LinearizationMode.HESSIAN)
        prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-4)
        prior_pose = gtsam.Pose3(gtsam.Rot3.Identity(), gtsam.Point3(0, 0, 0))
        prior_factor = gtsam.PriorFactorPose3(X(0), prior_pose, prior_noise)

        smart_factor_measurements = {}
        factors_feat_id_to_index = {}
        factors_index_shift = 0
        new_factors = gtsam.NonlinearFactorGraph()
        new_factors.add(prior_factor)
        factors_index_shift += 1
        for track_item in first_active_track:
            right_u = track_item[ActiveTrackSchema.RIGHT_U]
            left_u = track_item[ActiveTrackSchema.LEFT_U]
            left_v = track_item[ActiveTrackSchema.LEFT_V]
            feat_id = int(track_item[ActiveTrackSchema.FEAT_ID])
            if np.isnan(right_u):
                continue

            stereo_point = gtsam.StereoPoint2(left_u, right_u, left_v)
            smart_stereo_factor = SmartStereoProjectionPoseFactor(
                sharedNoiseModel=smart_noise,
                params=smart_params,
                body_P_sensor=stereo_ctx.cam0_in_body_se3.as_gtsam_pose(),
            )
            pose_key = X(0)
            smart_stereo_factor.add(stereo_point, X(0), stereo_ctx.stereo_k_gtsam)
            new_factors.add(smart_stereo_factor)
            if smart_factor_measurements.get(feat_id) is None:
                smart_factor_measurements[feat_id] = deque(maxlen=25)
            smart_factor_measurements[feat_id].append(SmartObs(pose_key, left_u, right_u, left_v))
            factors_feat_id_to_index[feat_id] = factors_index_shift
            factors_index_shift += 1

        if new_factors.empty():
            pytest.fail("No factors added to the graph")
        new_values = gtsam.Values()
        new_timestamps = gtsam.FixedLagSmootherKeyTimestampMap()

        new_values.insert(X(0), gtsam.Pose3(gtsam.Rot3.Identity(), gtsam.Point3(0, 0, 0)))
        new_timestamps.insert((X(0), 0.0))

        smoother.update(new_factors, new_values, new_timestamps)
        result = smoother.calculateEstimate()

        feat_zero_slot = factors_feat_id_to_index[0]
        stored_smart_factor = smoother.getFactors().at(feat_zero_slot)
        stored_smart_factor = cast("SmartStereoProjectionPoseFactor", stored_smart_factor)
        triang_result = stored_smart_factor.point(result)
        assert triang_result.valid()

        next_active_track = first_active_track.copy()
        new_factors = gtsam.NonlinearFactorGraph()
        odometry_factor = gtsam.BetweenFactorPose3(
            X(0), X(1), gtsam.Pose3(gtsam.Rot3.Identity(), gtsam.Point3(0, 0, 0)), prior_noise
        )
        new_factors.add(odometry_factor)
        factors_index_shift += 1

        to_delete_slots: Sequence[SupportsInt] = []

        for track_item in next_active_track:
            right_u = track_item[ActiveTrackSchema.RIGHT_U]
            left_u = track_item[ActiveTrackSchema.LEFT_U]
            left_v = track_item[ActiveTrackSchema.LEFT_V]
            feat_id = int(track_item[ActiveTrackSchema.FEAT_ID])
            if np.isnan(right_u):
                continue

            smart_stereo_factor = SmartStereoProjectionPoseFactor(
                sharedNoiseModel=smart_noise,
                params=smart_params,
                body_P_sensor=stereo_ctx.cam0_in_body_se3.as_gtsam_pose(),
            )
            prev_slot = factors_feat_id_to_index[feat_id]

            to_delete_slots.append(prev_slot)

            smart_factor_measurements[feat_id].append(SmartObs(X(1), left_u, right_u, left_v))
            for meas in smart_factor_measurements[feat_id]:
                stereo_point = gtsam.StereoPoint2(meas.ul, meas.ur, meas.v)
                smart_stereo_factor.add(stereo_point, meas.pose_key, stereo_ctx.stereo_k_gtsam)
            new_factors.add(smart_stereo_factor)
            factors_feat_id_to_index[feat_id] = factors_index_shift
            factors_index_shift += 1

        new_values = gtsam.Values()
        new_timestamps = gtsam.FixedLagSmootherKeyTimestampMap()
        new_values.insert(X(1), gtsam.Pose3(gtsam.Rot3.Identity(), gtsam.Point3(0, 0, 0)))
        new_timestamps.insert((X(1), 1.0))
        update_result = smoother.update(new_factors, new_values, new_timestamps, to_delete_slots)
        assert update_result is not None
        result = smoother.calculateEstimate()
        assert result.exists(X(1))
        zero_feat_slot = factors_feat_id_to_index[0]
        stored_smart_factor = smoother.getFactors().at(zero_feat_slot)
        assert stored_smart_factor is not None
        stored_smart_factor = cast("SmartStereoProjectionPoseFactor", stored_smart_factor)
        triang_result = stored_smart_factor.point(result)
        assert triang_result.valid()
