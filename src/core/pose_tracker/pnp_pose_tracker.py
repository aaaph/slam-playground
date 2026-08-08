import cv2
import gtsam
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from core.camera_model.stereo_camera_ctx import StereoContext
from core.feature_tracker.feature_schema import FeatureSchema
from core.pose_tracker.local_map import LocalMap, LocalMapSchema
from core.transformations.special_euclidian_3_dim import SE3
from logger import spawn_logger

type TrackedFeatureFrame = NDArray[np.float32]  # rows follow `FeatureSchema` (e.g. `FeatureFrame.good_features()`)

L = gtsam.symbol_shorthand.L
X = gtsam.symbol_shorthand.X


class PnpPoseTracker:
    """Pnp pose tracker."""

    def __init__(self, stereo_ctx: StereoContext, *, motion_only_ba_enabled: bool = True) -> None:
        """Initialize the pnp pose tracker."""
        self.stereo_ctx = stereo_ctx
        self.pnp_points_threshold = 4
        self.motion_only_ba_enabled = motion_only_ba_enabled

        self.opt_params = gtsam.LevenbergMarquardtParams()
        self.opt_params.setMaxIterations(5)
        self.opt_params.setRelativeErrorTol(1e-3)
        self.opt_params.setAbsoluteErrorTol(1e-3)
        self.opt_params.setlambdaInitial(1e-3)
        self.opt_params.setlambdaFactor(10.0)
        self.opt_params.setlambdaLowerBound(1e-6)
        self.opt_params.setlambdaUpperBound(1e3)
        self.opt_params.setVerbosity("SILENT")

        self.ba_noise_model_3 = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
        self.ba_noise_model_2 = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        self.ba_robust_noise_3 = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.345), self.ba_noise_model_3
        )
        self.ba_robust_noise_2 = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.345), self.ba_noise_model_2
        )
        self.logger = spawn_logger(app=PnpPoseTracker.__name__)

    @classmethod
    def default_factory(
        cls, stereo_ctx: StereoContext, *, motion_only_ba_enabled: bool = True
    ) -> "PnpPoseTracker":
        """Create a default `PnpPoseTracker`."""
        return cls(stereo_ctx, motion_only_ba_enabled=motion_only_ba_enabled)

    def find_pose(self, active_frame: TrackedFeatureFrame, local_map: LocalMap) -> tuple[bool, str, SE3]:
        """
        Find the pose using PnP + Motion-Only Bundle Adjustment.

        The method returns body in world pose.
        """
        tracked_measurements = np.full((active_frame.shape[0], 5), np.nan, dtype=np.float64)
        tracked_measurements[:, 0] = active_frame[:, FeatureSchema.FEAT_ID]
        tracked_measurements[:, 1:5] = active_frame[:, FeatureSchema.LEFT_U : FeatureSchema.RIGHT_V + 1]
        feat_ids = tracked_measurements[:, 0].astype(np.int32, copy=False)
        mask, points = local_map.get_stable_batch(feat_ids)
        if mask.sum() < self.pnp_points_threshold:
            return False, "Not enough map correspondences for PnP", SE3.identity()

        object_points = points[mask, LocalMapSchema.X : LocalMapSchema.Z + 1].astype(np.float64, copy=False)
        image_points = tracked_measurements[mask, 1:3].astype(np.float64, copy=False)
        valid_feat_ids = feat_ids[mask]
        try:
            cam0_in_world_se3, good_ids, bad_ids = self._resolve_pnp_pose(
                object_points,
                image_points,
                valid_feat_ids,
            )
            # visual_feat schema (feat_id, left_u, left_v, right_u, right_v, x, y, z) - shape = 8
            tracked_measurements_with_map = tracked_measurements[mask]
            good_mask = np.isin(valid_feat_ids, good_ids)
            visual_features = np.full((good_ids.shape[0], 8), np.nan, dtype=np.float64)
            visual_features[:, 0] = good_ids
            visual_features[:, 1:5] = tracked_measurements_with_map[good_mask, 1:5]
            visual_features[:, 5:8] = object_points[good_mask]
            if self.motion_only_ba_enabled:
                cam0_in_world_se3 = self._resolve_ba_correction(cam0_in_world_se3, visual_features)
            local_map.increase_health(good_ids)
            local_map.decrease_health(bad_ids)

            return (
                True,
                "Successfully estimated PnP pose",
                cam0_in_world_se3 * self.stereo_ctx.cam0_in_body_se3.inverse(),
            )
        except ValueError as e:
            return False, str(e), SE3.identity()

    def _resolve_pnp_pose(
        self,
        object_points: NDArray[np.float64],
        image_points: NDArray[np.float64],
        feat_ids: NDArray[np.int32],
    ) -> tuple[SE3, NDArray[np.int32], NDArray[np.int32]]:
        """Resolve the PnP pose."""
        distortion_coeffs = np.array([0, 0, 0, 0, 0])
        _, rvec, tvec, inliners = cv2.solvePnPRansac(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=self.stereo_ctx.stereo_k,
            distCoeffs=distortion_coeffs,
            iterationsCount=100,
            reprojectionError=3.0,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if inliners is None or inliners.sum() < self.pnp_points_threshold:
            raise ValueError("Not enough inliners for PnP")
        inliner_idx = inliners.ravel()
        mask = np.zeros(len(feat_ids), dtype=bool)
        mask[inliner_idx] = True
        good_ids = feat_ids[mask]
        bad_ids = feat_ids[~mask]
        rvec, tvec = cv2.solvePnPRefineLM(
            objectPoints=object_points[mask],
            imagePoints=image_points[mask],
            cameraMatrix=self.stereo_ctx.stereo_k,
            distCoeffs=distortion_coeffs,
            rvec=rvec,
            tvec=tvec,
        )
        rot, _ = cv2.Rodrigues(rvec)
        new_rotation = Rotation.from_matrix(rot.transpose())
        new_translation = -new_rotation.as_matrix() @ tvec.reshape(1, 3).flatten()

        return SE3(new_rotation, new_translation), good_ids, bad_ids

    def _resolve_ba_correction(self, pose_k: SE3, visual_features: NDArray[np.float64]) -> SE3:
        """Resolve the BA correction."""
        graph = gtsam.NonlinearFactorGraph()
        pose_key = X(0)
        initial_values = gtsam.Values()
        initial_values.insert(pose_key, pose_k.as_gtsam_pose())

        for visual_feature in visual_features:
            feat_id = int(visual_feature[0])
            landmark_key = L(feat_id)
            is_stereo = not np.isnan(visual_feature[3])

            initial_values.insert(
                landmark_key, gtsam.Point3(visual_feature[5], visual_feature[6], visual_feature[7])
            )
            fixed_noise = gtsam.noiseModel.Constrained.All(3)
            freeze_prior = gtsam.PriorFactorPoint3(
                landmark_key, gtsam.Point3(visual_feature[5], visual_feature[6], visual_feature[7]), fixed_noise
            )
            graph.add(freeze_prior)

            if is_stereo:
                stereo_point = gtsam.StereoPoint2(visual_feature[1], visual_feature[3], visual_feature[2])
                factor = gtsam.GenericStereoFactor3D(
                    stereo_point, self.ba_robust_noise_3, pose_key, landmark_key, self.stereo_ctx.stereo_k_gtsam
                )
                graph.add(factor)
            else:
                mono_point = gtsam.Point2(visual_feature[1], visual_feature[2])
                factor = gtsam.GenericProjectionFactorCal3_S2(
                    mono_point, self.ba_robust_noise_2, pose_key, landmark_key, self.stereo_ctx.cam0_k_gtsam
                )
                graph.add(factor)

        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, self.opt_params)
        result = optimizer.optimize()
        refined_pose = result.atPose3(pose_key)
        return SE3.from_gtsam_pose(refined_pose)
