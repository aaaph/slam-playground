import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

import gtsam
from core.feature_tracker.feature import Feature
from core.pose_tracker.feature_triangulation import FeatureTriangulation
from core.pose_tracker.local_map import LocalMap
from core.transformations.special_euclidian_3_dim import SE3
from core.types.stereo_camera_dto import StereoCameraDto

Vector3d = NDArray[np.float32]
FeatureId = int
CameraInWorld = SE3
FeatureIds = NDArray[np.int32]
NewLandmarks = dict[FeatureId, Vector3d]
X = gtsam.symbol_shorthand.X
L = gtsam.symbol_shorthand.L


class PoseTracker:
    """Pose tracker."""

    def __init__(
        self,
        initial_pose: SE3,
        stereo_camera_dto: StereoCameraDto,
        local_map: LocalMap,
        feat_triangulation: FeatureTriangulation,
    ) -> None:
        """Initialize the pose tracker."""
        self.active_pose = initial_pose

        self.local_map = local_map
        self.feat_triangulation = feat_triangulation

        self.stereo_k = stereo_camera_dto.stereo_k
        self.cam0_k = stereo_camera_dto.cam0_k
        self.stereo_k_gtsam = stereo_camera_dto.stereo_k_gtsam
        self.left_cam_k_gtsam = stereo_camera_dto.cam0_k_gtsam
        self.cam0_in_body = stereo_camera_dto.T_body_cam0
        self.body_in_cam0 = self.cam0_in_body.inverse()

    @classmethod
    def default_factory(
        cls, initial_pose: SE3, stereo_camera_dto: StereoCameraDto, map_capacity: int = 800
    ) -> "PoseTracker":
        """Create a default `PoseTracker` with a new local map and feature triangulation helper."""
        local_map = LocalMap(map_capacity)
        feat_triangulation = FeatureTriangulation.from_stereo_camera_dto(stereo_camera_dto)
        return cls(initial_pose, stereo_camera_dto, local_map, feat_triangulation)

    def estimate_first(self, ts: float, features: list[Feature]) -> tuple[CameraInWorld, NewLandmarks]:
        """Estimate the first pose. The method is used to bootstrap the pose tracker when local map is empty."""
        cam0_in_world_se3 = self.active_pose
        new_landmarks = self._landmark_triangulation(ts, cam0_in_world_se3, features)
        return cam0_in_world_se3, new_landmarks

    def estimate(self, ts: float, features: list[Feature]) -> tuple[CameraInWorld, NewLandmarks]:
        """Estimate the pose."""
        if self.local_map.empty():
            return self.estimate_first(ts, features)
        cam0_in_world_se3, good_feat_ids = self._pnp_pose_prediction(features)
        cam0_in_world_se3 = self._ba_pose_correction(cam0_in_world_se3, features, good_feat_ids)
        self.active_pose = cam0_in_world_se3

        new_landmarks = self._landmark_triangulation(ts, cam0_in_world_se3, features)

        return cam0_in_world_se3, new_landmarks

    def _pnp_pose_prediction(self, features: list[Feature]) -> tuple[CameraInWorld, FeatureIds]:
        """
        Solve the PnP problem using RANSAC and LM refinement.

        The method returns the pose estimation based on current 3D points and image points.
        """
        object_points = []
        image_points = []
        feat_ids = []

        for feature in features:
            feat_id = feature.feat_id
            _, uv_left, _ = feature.get_active_stereo_pair()
            if self.local_map.exists(feat_id):
                feat_3d = self.local_map.get_point(feat_id)
                object_points.append(feat_3d)
                image_points.append((uv_left[0], uv_left[1]))
                feat_ids.append(feat_id)
        object_points = np.array(object_points)
        image_points = np.array(image_points)
        feat_ids = np.array(feat_ids, dtype=np.int32)
        cam0_in_world_se3, good_feat_ids = PoseTracker._resolve_pnp_pose(
            object_points, image_points, feat_ids, self.stereo_k
        )
        return cam0_in_world_se3, good_feat_ids

    def _ba_pose_correction(self, pose_k: SE3, features: list[Feature], inliners: FeatureIds) -> CameraInWorld:
        """
        Perform motion-only bundle adjustment.

        The method does non-linear optimization to refine the pose estimation.
        """
        object_points = []
        image_points = []
        for feature in features:
            feat_id = feature.feat_id
            if feat_id not in inliners:
                continue
            meas = feature.get_active_measurement()
            if meas.is_stereo():
                uv_left, uv_right = meas.pair()
                image_points.append((uv_left[0], uv_left[1], uv_right[0]))
                object_points.append(self.local_map.get_point(feat_id))
            else:
                uv_left = meas.left
                image_points.append((uv_left[0], uv_left[1], None))
                object_points.append(self.local_map.get_point(feat_id))
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        return PoseTracker._resolve_ba_correction(
            pose_k,
            object_points,
            image_points,
            self.stereo_k_gtsam,
            self.left_cam_k_gtsam,
        )

    def _landmark_triangulation(self, ts: float, cam0_in_world_se3: SE3, features: list[Feature]) -> NewLandmarks:
        """
        Triangulate landmarks.

        The method triangulates new landmarks based on the current pose estimation and feature measurements.
        """
        body_in_world_se3 = cam0_in_world_se3 * self.body_in_cam0
        new_landmarks: NewLandmarks = {}

        for feature in features:
            if self.local_map.exists(feature.feat_id):
                continue
            meas = feature.get_active_measurement()
            initialized = False

            if meas.is_stereo():
                result = self.feat_triangulation.make_initial_guess_by_stereo_pair(feature)
                good_feature, initial_guess = result
                if good_feature:
                    feat_in_camera_vec = initial_guess.copy()
                    feat_in_world_vec = cam0_in_world_se3 @ feat_in_camera_vec
                    feature.state = "stable"
                    new_landmarks[feature.feat_id] = feat_in_world_vec
                    initialized = True

            if not initialized:
                delta_a, delta_b = self.feat_triangulation.compute_feature_linear_system_update(
                    feature, ts, body_in_world_se3
                )
                feature.apply_linear_system_update(delta_a, delta_b)
                good_feature, initial_guess = self.feat_triangulation.make_linear_triangulation_guess(
                    feature, cam0_in_world_se3
                )
                if good_feature:
                    new_landmarks[feature.feat_id] = initial_guess
                    feature.state = "stable"

        self.local_map.add_points(new_landmarks)
        return new_landmarks

    @staticmethod
    def _resolve_pnp_pose(
        object_points: NDArray[np.float64],
        image_points: NDArray[np.float64],
        feat_ids: NDArray[np.int32],
        k_matrix: np.ndarray,
    ) -> tuple[SE3, NDArray[np.int32]]:
        """Resolve the PnP pose."""
        distortion_coeffs = np.array([0, 0, 0, 0, 0])
        _, rvec, tvec, inliners = cv2.solvePnPRansac(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=k_matrix,
            distCoeffs=distortion_coeffs,
            iterationsCount=100,
            reprojectionError=3.0,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        rvec, tvec = cv2.solvePnPRefineLM(
            objectPoints=object_points[inliners],
            imagePoints=image_points[inliners],
            cameraMatrix=k_matrix,
            distCoeffs=distortion_coeffs,
            rvec=rvec,
            tvec=tvec,
        )
        rot, _ = cv2.Rodrigues(rvec)
        new_rotation = Rotation.from_matrix(rot.transpose())
        new_translation = -new_rotation.as_matrix() @ tvec.reshape(1, 3).flatten()

        return SE3(new_rotation, new_translation), feat_ids[inliners]

    @staticmethod
    def _resolve_ba_correction(
        pose_k_initial_guess: SE3,
        points_3d: NDArray[np.float64],
        measurements: NDArray[np.float64],
        stereo_k_matrix: gtsam.Cal3_S2Stereo,
        mono_k_matrix: gtsam.Cal3_S2,
    ) -> SE3:
        """Motion-Only Bundle Adjustment using GTSAM."""
        graph = gtsam.NonlinearFactorGraph()
        pose_key = X(0)
        measurement_noise_3d = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
        measurement_noise_2d = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        robust_noise_3d = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.345), measurement_noise_3d
        )
        robust_noise_2d = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.345), measurement_noise_2d
        )
        initial_values = gtsam.Values()
        initial_values.insert(pose_key, pose_k_initial_guess.as_gtsam_pose())
        for idx, (point_3d, measurement) in enumerate(zip(points_3d, measurements, strict=False)):
            landmark_key = L(idx)
            ul, v, ur = measurement
            if not np.isnan(ur):
                stereo_point = gtsam.StereoPoint2(ul, ur, v)
                factor = gtsam.GenericStereoFactor3D(
                    stereo_point, robust_noise_3d, pose_key, landmark_key, stereo_k_matrix
                )
                graph.add(factor)
            else:
                point_2d = gtsam.Point2(ul, v)
                factor = gtsam.GenericProjectionFactorCal3_S2(
                    point_2d, robust_noise_2d, pose_key, landmark_key, mono_k_matrix
                )
                graph.add(factor)

            initial_values.insert(landmark_key, point_3d)
            fixed_noise = gtsam.noiseModel.Constrained.All(3)
            freeze_prior = gtsam.PriorFactorPoint3(landmark_key, gtsam.Point3(*point_3d), fixed_noise)
            graph.add(freeze_prior)

        params = gtsam.DoglegParams()
        params.setDeltaInitial(1.0)
        # params.setVerbosity("ERROR")
        optimizer = gtsam.DoglegOptimizer(graph, initial_values, params)
        result = optimizer.optimize()
        refined_pose = result.atPose3(pose_key)

        return SE3.from_gtsam_pose(refined_pose)
