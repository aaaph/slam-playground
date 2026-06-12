import cv2
import gtsam
import numpy as np
from attr import dataclass
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from core.camera_model.stereo_camera_ctx import StereoContext
from core.loop_closure.vpr_frame import VPRFrame
from core.transformations.special_euclidian_3_dim import SE3

type QueryUV = NDArray[np.float64]
type ReferenceXYZ = NDArray[np.float64]
type Matrix = NDArray[np.float64]
type InlierMask = NDArray[np.bool_]
type VisualFeatures = NDArray[np.float64]
X = gtsam.symbol_shorthand.X
L = gtsam.symbol_shorthand.L


@dataclass(frozen=True, slots=True, repr=False)
class PnpResult:
    """PnP result."""

    success: bool
    reason: str
    reference_t_query: SE3
    inliner_mask: InlierMask
    reprojection_errors: NDArray[np.float64]

    @property
    def inliners_count(self) -> int:
        """Get the number of inliners."""
        return int(np.count_nonzero(self.inliner_mask))

    @property
    def inliners_ratio(self) -> float:
        """Get the ratio of inliners."""
        if len(self.inliner_mask) == 0:
            return 0.0
        return self.inliners_count / len(self.inliner_mask)

    def __repr__(self) -> str:
        """Return a string representation of the PnP result."""
        median_error = (
            float(np.median(self.reprojection_errors[self.inliner_mask]))
            if self.success and self.reprojection_errors.shape[0] > 0
            else np.inf
        )
        inliers_str = f"{self.inliners_count}/{len(self.inliner_mask)}"
        return (
            f"PnpResult(success={self.success}, reason={self.reason}, "
            f"reference_t_query={self.reference_t_query}, inliers={inliers_str}, "
            f"median_error={median_error})"
        )

    @classmethod
    def failed(cls, reason: str, inliner_mask: InlierMask) -> "PnpResult":
        """Create a failed PnP result."""
        return cls(
            success=False,
            reason=reason,
            reference_t_query=SE3.identity(),
            inliner_mask=inliner_mask,
            reprojection_errors=np.empty((0,), dtype=np.float64),
        )


class VPRPNPEstimator:
    """VPR PnP estimator."""

    def __init__(self, stereo_ctx: StereoContext) -> None:
        """Initialize the VPR PnP estimator."""
        self.stereo_ctx = stereo_ctx
        self.min_pnp_points = 12

        self.pnp_ransac_iterations = 100
        self.pnp_reprojection_error_px = 3.0
        self.pnp_confidence = 0.999

        self.ba_noise_model_3 = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
        self.ba_noise_model_2 = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        self.ba_robust_noise_3 = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.345), self.ba_noise_model_3
        )
        self.ba_robust_noise_2 = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.345), self.ba_noise_model_2
        )
        self.fixed_point_noise = gtsam.noiseModel.Constrained.All(3)
        self.ba_params = gtsam.DoglegParams()
        self.ba_params.setDeltaInitial(1.0)

    @classmethod
    def default_factory(cls, stereo_ctx: StereoContext) -> "VPRPNPEstimator":
        """Create a default `VPRPNPEstimator`."""
        return cls(stereo_ctx)

    def estimate_query_pose(
        self, _query_frame: VPRFrame, _reference_frame: VPRFrame, matches: list[cv2.DMatch]
    ) -> PnpResult:
        """Estimate the query pose."""
        query_left_uv = _query_frame.left_uv[[match.queryIdx for match in matches]].astype(np.float64)
        query_right_uv = _query_frame.right_uv[[match.queryIdx for match in matches]].astype(np.float64)
        reference_points = _reference_frame.points_xyz[[match.trainIdx for match in matches]].astype(np.float64)
        visual_features = np.full((len(matches), 8), np.nan, dtype=np.float64)
        visual_features[:, 0] = [match.queryIdx for match in matches]
        visual_features[:, 1:5] = np.column_stack((query_left_uv, query_right_uv))
        visual_features[:, 5:8] = reference_points
        ref_t_query, inliner_mask, reason = self._estimate_reference_t_query_pnp_ransac(
            reference_points, query_left_uv
        )
        if ref_t_query is None:
            return PnpResult.failed(reason, inliner_mask)

        optimized_ref_t_query = self._motion_only_ba(ref_t_query, visual_features[inliner_mask])

        return PnpResult(
            success=True,
            reason=reason,
            reference_t_query=optimized_ref_t_query,
            inliner_mask=inliner_mask,
            reprojection_errors=self.reprojection_errors(optimized_ref_t_query, reference_points, query_left_uv),
        )

    def reprojection_errors(
        self,
        reference_t_query: SE3,
        reference_points: ReferenceXYZ,
        query_uv: QueryUV,
    ) -> NDArray[np.float64]:
        """Compute query-image reprojection errors for reference-frame 3D points."""
        query_t_reference = reference_t_query.inverse()
        reference_points_in_query = np.array(
            [query_t_reference.act_on_vector(point) for point in reference_points],
            dtype=np.float64,
        )
        errors = np.full((reference_points.shape[0],), np.inf, dtype=np.float64)
        valid_depth_mask = reference_points_in_query[:, 2] > 0.0
        if not np.any(valid_depth_mask):
            return errors

        k_matrix = self.stereo_ctx.stereo_k
        projected_uv = np.column_stack(
            [
                k_matrix[0, 0]
                * reference_points_in_query[valid_depth_mask, 0]
                / reference_points_in_query[valid_depth_mask, 2]
                + k_matrix[0, 2],
                k_matrix[1, 1]
                * reference_points_in_query[valid_depth_mask, 1]
                / reference_points_in_query[valid_depth_mask, 2]
                + k_matrix[1, 2],
            ]
        )
        errors[valid_depth_mask] = np.linalg.norm(projected_uv - query_uv[valid_depth_mask], axis=1)
        return errors

    def _estimate_reference_t_query_pnp_ransac(
        self,
        reference_points: ReferenceXYZ,
        query_uv: QueryUV,
    ) -> tuple[SE3 | None, InlierMask, str]:
        """Estimate reference_T_query with OpenCV PnP RANSAC."""
        if reference_points.shape[0] < self.min_pnp_points:
            return None, np.empty((0,), dtype=bool), "Not enough points"

        dist_coeffs = np.zeros((5,), dtype=np.float64)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=reference_points,
            imagePoints=query_uv,
            cameraMatrix=self.stereo_ctx.stereo_k,
            distCoeffs=dist_coeffs,
            iterationsCount=self.pnp_ransac_iterations,
            reprojectionError=self.pnp_reprojection_error_px,
            confidence=self.pnp_confidence,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        mask = np.zeros(reference_points.shape[0], dtype=bool)
        if not ok or rvec is None or tvec is None or inliers is None:
            return None, mask, "not ok or rvec is None or tvec is None or inliers is None"

        inliner_idx = inliers.ravel()
        mask[inliner_idx] = True
        if int(np.count_nonzero(mask)) < self.min_pnp_points:
            return None, mask, "Not enough inliers"

        rvec, tvec = cv2.solvePnPRefineLM(
            objectPoints=reference_points[mask],
            imagePoints=query_uv[mask],
            cameraMatrix=self.stereo_ctx.stereo_k,
            distCoeffs=dist_coeffs,
            rvec=rvec,
            tvec=tvec,
        )
        rot, _ = cv2.Rodrigues(rvec)
        new_rotation = Rotation.from_matrix(rot.transpose())
        new_translation = -new_rotation.as_matrix() @ tvec.reshape(1, 3).flatten()
        reference_t_query = SE3(new_rotation, new_translation)
        return reference_t_query, mask, "Successfully estimated reference_T_query with PnP RANSAC"

    def _motion_only_ba(self, reference_t_query: SE3, visual_features: VisualFeatures) -> SE3:
        """Motion-only BA."""
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        pose_key = X(0)
        values.insert(pose_key, reference_t_query.as_gtsam_pose())

        # visual features schema (feat_id, left_u, left_v, right_u, right_v, x, y, z) - shape = 8
        for visual_feature in visual_features:
            feat_id = int(visual_feature[0])
            landmark_key = L(feat_id)
            is_stereo = not np.isnan(visual_feature[3])
            landmark = gtsam.Point3(visual_feature[5], visual_feature[6], visual_feature[7])
            values.insert(landmark_key, landmark)
            graph.add(gtsam.PriorFactorPoint3(landmark_key, landmark, self.fixed_point_noise))
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

        optimizer = gtsam.DoglegOptimizer(graph, values, self.ba_params)
        result = optimizer.optimize()
        return SE3.from_gtsam_pose(result.atPose3(pose_key))
