from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import cv2
import gtsam
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from core.pose_tracker.frame_to_frame_pnp_store import PnPMapSchema
from core.transformations.special_euclidian_3_dim import SE3
from logger import spawn_logger

if TYPE_CHECKING:
    from core.camera_model.stereo_camera_ctx import StereoContext

type FeatureIds = NDArray[np.int32]
type InlierMask = NDArray[np.bool_]
type ObjectPoints = NDArray[np.float64]
type PixelPoints = NDArray[np.float64]
type VisualFeatures = NDArray[np.float64]


X = gtsam.symbol_shorthand.X
L = gtsam.symbol_shorthand.L


class PnpSolveStatus(StrEnum):
    """PnP solver status."""

    SUCCESS = "success"
    NOT_ENOUGH_POINTS = "not_enough_points"
    PNP_FAILED = "pnp_failed"
    NOT_ENOUGH_INLIERS = "not_enough_inliers"


@dataclass(frozen=True, slots=True)
class PnpSolverConfig:
    """PnP solver configuration."""

    min_points: int = 4
    ransac_iterations: int = 100
    ransac_reprojection_error_px: float = 3.0
    ransac_confidence: float = 0.999
    refine_with_lm: bool = True
    motion_only_ba_enabled: bool = True
    ba_max_iterations: int = 5
    ba_relative_error_tol: float = 1e-3
    ba_absolute_error_tol: float = 1e-3
    ba_lambda_initial: float = 1e-3
    ba_lambda_factor: float = 10.0
    ba_lambda_lower_bound: float = 1e-6
    ba_lambda_upper_bound: float = 1e3
    ba_huber_k: float = 1.345
    ba_noise_sigma_px: float = 1.0


@dataclass(frozen=True, slots=True, repr=False)
class PnpPoseResult:
    """PnP pose result."""

    status: PnpSolveStatus
    reason: str
    cam0_in_reference: SE3
    inlier_feat_ids: FeatureIds
    outlier_feat_ids: FeatureIds
    inlier_mask: InlierMask
    reprojection_errors: NDArray[np.float64]

    @property
    def ok(self) -> bool:
        """Return True when the solver produced a pose."""
        return self.status is PnpSolveStatus.SUCCESS

    @property
    def inlier_count(self) -> int:
        """Return the number of inlier correspondences."""
        return int(np.count_nonzero(self.inlier_mask))

    @property
    def inlier_ratio(self) -> float:
        """Return the inlier ratio over all input correspondences."""
        if self.inlier_mask.shape[0] == 0:
            return 0.0
        return self.inlier_count / self.inlier_mask.shape[0]

    @classmethod
    def failed(
        cls,
        status: PnpSolveStatus,
        reason: str,
        feat_ids: FeatureIds,
        inlier_mask: InlierMask | None = None,
    ) -> PnpPoseResult:
        """Create a failed PnP result."""
        normalized_feat_ids = np.asarray(feat_ids, dtype=np.int32)
        if inlier_mask is None:
            inlier_mask = np.zeros(normalized_feat_ids.shape[0], dtype=bool)
        return cls(
            status=status,
            reason=reason,
            cam0_in_reference=SE3.identity(),
            inlier_feat_ids=normalized_feat_ids[inlier_mask],
            outlier_feat_ids=normalized_feat_ids[~inlier_mask],
            inlier_mask=inlier_mask,
            reprojection_errors=np.empty((0,), dtype=np.float64),
        )

    def __repr__(self) -> str:
        """Return a compact debug representation."""
        return (
            f"PnpPoseResult(status={self.status.value}, reason={self.reason}, "
            f"inliers={self.inlier_count}/{self.inlier_mask.shape[0]}, "
            f"cam0_in_reference={self.cam0_in_reference})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class _PnPRansacResult:
    """PnP RANSAC result."""

    status: PnpSolveStatus
    reason: str | None
    pose: SE3
    inlier_mask: InlierMask
    outlier_mask: InlierMask

    @property
    def ok(self) -> bool:
        """Return True when RANSAC produced a pose."""
        return self.status is PnpSolveStatus.SUCCESS

    @classmethod
    def failed(
        cls,
        reason: str,
        count: int,
        inlier_mask: InlierMask | None = None,
        status: PnpSolveStatus = PnpSolveStatus.PNP_FAILED,
    ) -> _PnPRansacResult:
        """Create a failed PnP RANSAC result."""
        mask = np.zeros(count, dtype=bool) if inlier_mask is None else inlier_mask
        return cls(
            status=status,
            reason=reason,
            pose=SE3.identity(),
            inlier_mask=mask,
            outlier_mask=np.logical_not(mask),
        )

    @classmethod
    def not_enough_inliers(cls, inlier_mask: InlierMask) -> _PnPRansacResult:
        """Create a result with not enough inliers."""
        return cls(
            status=PnpSolveStatus.NOT_ENOUGH_INLIERS,
            reason="Not enough PnP inliers",
            pose=SE3.identity(),
            inlier_mask=inlier_mask,
            outlier_mask=np.logical_not(inlier_mask),
        )


@dataclass(frozen=True, slots=True, repr=False)
class _MotionOnlyBaResult:
    """Motion-only BA result."""

    pose: SE3
    ok: bool
    reason: str | None

    @classmethod
    def failed(cls, reason: str) -> _MotionOnlyBaResult:
        """Create a failed motion-only BA result."""
        return cls(pose=SE3.identity(), ok=False, reason=reason)


class PnpPoseSolver:
    """Pure PnP pose solver for 3D reference points and current cam0 observations."""

    def __init__(self, stereo_ctx: StereoContext, config: PnpSolverConfig | None = None) -> None:
        """Initialize the PnP pose solver."""
        self.stereo_ctx = stereo_ctx
        self.config = config or PnpSolverConfig()

        self.opt_params = gtsam.LevenbergMarquardtParams()
        self.opt_params.setMaxIterations(self.config.ba_max_iterations)
        self.opt_params.setRelativeErrorTol(self.config.ba_relative_error_tol)
        self.opt_params.setAbsoluteErrorTol(self.config.ba_absolute_error_tol)
        self.opt_params.setlambdaInitial(self.config.ba_lambda_initial)
        self.opt_params.setlambdaFactor(self.config.ba_lambda_factor)
        self.opt_params.setlambdaLowerBound(self.config.ba_lambda_lower_bound)
        self.opt_params.setlambdaUpperBound(self.config.ba_lambda_upper_bound)
        self.opt_params.setVerbosity("SILENT")

        self.ba_noise_model_3 = gtsam.noiseModel.Isotropic.Sigma(3, self.config.ba_noise_sigma_px)
        self.ba_noise_model_2 = gtsam.noiseModel.Isotropic.Sigma(2, self.config.ba_noise_sigma_px)
        self.ba_robust_noise_3 = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(self.config.ba_huber_k), self.ba_noise_model_3
        )
        self.ba_robust_noise_2 = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(self.config.ba_huber_k), self.ba_noise_model_2
        )
        self.fixed_point_noise = gtsam.noiseModel.Constrained.All(3)
        self.logger = spawn_logger("pnp_solver")

    @classmethod
    def default_factory(cls, stereo_ctx: StereoContext, config: PnpSolverConfig | None = None) -> PnpPoseSolver:
        """Create a default `PnpPoseSolver`."""
        return cls(stereo_ctx, config=config)

    def solve_visual_features(self, visual_features: VisualFeatures) -> PnpPoseResult:
        """Estimate current cam0 pose in the reference frame from internal visual features."""
        visual_features_size = visual_features.shape[0]
        feat_ids = visual_features[:, PnPMapSchema.FEAT_ID].astype(np.int32, copy=False)

        if visual_features_size < self.config.min_points:
            return PnpPoseResult.failed(
                status=PnpSolveStatus.NOT_ENOUGH_POINTS,
                reason=f"Not enough visual features: {visual_features_size} < {self.config.min_points}",
                feat_ids=feat_ids,
            )

        pnp_result = self._estimate_cam0_in_reference_pnp_ransac(visual_features)
        if not pnp_result.ok:
            return PnpPoseResult.failed(
                status=pnp_result.status,
                reason=pnp_result.reason or "PnP RANSAC failed",
                feat_ids=feat_ids,
                inlier_mask=pnp_result.inlier_mask,
            )

        pose = pnp_result.pose

        if self.config.motion_only_ba_enabled:
            mo_ba_result = self._motion_only_ba(pose, visual_features[pnp_result.inlier_mask])
            if mo_ba_result.ok:
                pose = mo_ba_result.pose
            else:
                self.logger.warning(f"Motion-only BA failed: {mo_ba_result.reason} -> using PnP RANSAC pose")

        return PnpPoseResult(
            status=PnpSolveStatus.SUCCESS,
            reason=pnp_result.reason or "Successfully estimated cam0_in_reference with PnP RANSAC",
            cam0_in_reference=pose,
            inlier_feat_ids=feat_ids[pnp_result.inlier_mask],
            outlier_feat_ids=feat_ids[pnp_result.outlier_mask],
            inlier_mask=pnp_result.inlier_mask,
            reprojection_errors=self.reprojection_errors(
                pose,
                visual_features[:, PnPMapSchema.XYZ],
                visual_features[:, PnPMapSchema.LEFT_UV],
            ),
        )

    def reprojection_errors(
        self,
        cam0_in_reference: SE3,
        object_points: ObjectPoints,
        left_uv: PixelPoints,
    ) -> NDArray[np.float64]:
        """Compute reprojection errors for reference-frame 3D points in current cam0."""
        normalized_points = np.asarray(object_points, dtype=np.float64)
        normalized_uv = np.asarray(left_uv, dtype=np.float64)
        errors = np.full((normalized_points.shape[0],), np.inf, dtype=np.float64)
        cam0_in_object = cam0_in_reference.inverse()
        points_cam0 = cam0_in_object.rotation().apply(normalized_points) + cam0_in_object.translation()
        valid_mask = (
            np.all(np.isfinite(normalized_points), axis=1)
            & np.all(np.isfinite(normalized_uv), axis=1)
            & (points_cam0[:, 2] > 0.0)
        )
        if not np.any(valid_mask):
            return errors

        k_matrix = self.stereo_ctx.stereo_k
        projected_uv = np.column_stack(
            (
                k_matrix[0, 0] * points_cam0[valid_mask, 0] / points_cam0[valid_mask, 2] + k_matrix[0, 2],
                k_matrix[1, 1] * points_cam0[valid_mask, 1] / points_cam0[valid_mask, 2] + k_matrix[1, 2],
            )
        )
        errors[valid_mask] = np.linalg.norm(projected_uv - normalized_uv[valid_mask], axis=1)
        return errors

    def _estimate_cam0_in_reference_pnp_ransac(self, visual_features: VisualFeatures) -> _PnPRansacResult:
        """Estimate current cam0 pose in the reference frame with PnP RANSAC."""
        object_points: ObjectPoints = visual_features[:, PnPMapSchema.XYZ]
        image_points: PixelPoints = visual_features[:, PnPMapSchema.LEFT_UV]
        mask = np.zeros(visual_features.shape[0], dtype=bool)
        dist_coeffs = np.zeros((5,), dtype=np.float64)
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=object_points,
                imagePoints=image_points,
                cameraMatrix=self.stereo_ctx.stereo_k,
                distCoeffs=dist_coeffs,
                iterationsCount=self.config.ransac_iterations,
                reprojectionError=self.config.ransac_reprojection_error_px,
                confidence=self.config.ransac_confidence,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error as error:
            return _PnPRansacResult.failed(str(error), visual_features.shape[0])

        if not ok or rvec is None or tvec is None or inliers is None:
            reason = (
                "cv2.solvePnPRansac not successful"
                if not ok
                else "cv2.solvePnPRansac returned None rvec, tvec, or inliers"
            )
            return _PnPRansacResult.failed(
                reason,
                visual_features.shape[0],
            )

        mask[inliers.ravel()] = True
        if int(np.count_nonzero(mask)) < self.config.min_points:
            return _PnPRansacResult.not_enough_inliers(mask)

        if self.config.refine_with_lm:
            try:
                rvec, tvec = cv2.solvePnPRefineLM(
                    objectPoints=object_points[mask],
                    imagePoints=image_points[mask],
                    cameraMatrix=self.stereo_ctx.stereo_k,
                    distCoeffs=dist_coeffs,
                    rvec=rvec,
                    tvec=tvec,
                )
            except cv2.error as error:
                return _PnPRansacResult.failed(str(error), visual_features.shape[0], inlier_mask=mask)

        tvec_is_finite = np.all(np.isfinite(tvec))

        if rvec is None or tvec is None or not tvec_is_finite:
            return _PnPRansacResult.failed(
                "cv2.solvePnPRefineLM returned None rvec or tvec or non-finite tvec",
                visual_features.shape[0],
                inlier_mask=mask,
            )

        rot, _ = cv2.Rodrigues(rvec)
        new_rotation = Rotation.from_matrix(rot.transpose())
        new_translation = -new_rotation.as_matrix() @ tvec.reshape(1, 3).flatten()
        pose = SE3(new_rotation, new_translation)
        return _PnPRansacResult(
            status=PnpSolveStatus.SUCCESS,
            reason=None,
            pose=pose,
            inlier_mask=mask,
            outlier_mask=np.logical_not(mask),
        )

    def _motion_only_ba(self, cam0_in_reference: SE3, visual_features: VisualFeatures) -> _MotionOnlyBaResult:
        """Refine cam0 pose with fixed reference landmarks."""
        graph = gtsam.NonlinearFactorGraph()
        pose_key = X(0)
        values = gtsam.Values()
        values.insert(pose_key, cam0_in_reference.as_gtsam_pose())

        for visual_feature in visual_features:
            feat_id = int(visual_feature[PnPMapSchema.FEAT_ID])
            landmark_key = L(feat_id)
            landmark = gtsam.Point3(*visual_feature[PnPMapSchema.XYZ])
            values.insert(landmark_key, landmark)
            graph.add(gtsam.PriorFactorPoint3(landmark_key, landmark, self.fixed_point_noise))
            is_stereo = not np.isnan(visual_feature[PnPMapSchema.RIGHT_U])

            if is_stereo:
                stereo_point = gtsam.StereoPoint2(
                    visual_feature[PnPMapSchema.LEFT_U],
                    visual_feature[PnPMapSchema.RIGHT_U],
                    visual_feature[PnPMapSchema.LEFT_V],
                )
                graph.add(
                    gtsam.GenericStereoFactor3D(
                        stereo_point,
                        self.ba_robust_noise_3,
                        pose_key,
                        landmark_key,
                        self.stereo_ctx.stereo_k_gtsam,
                    )
                )
                continue

            mono_point = gtsam.Point2(
                visual_feature[PnPMapSchema.LEFT_U],
                visual_feature[PnPMapSchema.LEFT_V],
            )
            graph.add(
                gtsam.GenericProjectionFactorCal3_S2(
                    mono_point,
                    self.ba_robust_noise_2,
                    pose_key,
                    landmark_key,
                    self.stereo_ctx.cam0_k_gtsam,
                )
            )

        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, self.opt_params)
        try:
            result = optimizer.optimize()
        except RuntimeError as error:
            return _MotionOnlyBaResult.failed(str(error))

        pose = SE3.from_gtsam_pose(result.atPose3(pose_key))
        return _MotionOnlyBaResult(pose=pose, ok=True, reason=None)
