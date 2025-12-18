import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

import gtsam
from core.feature_tracker.feature_tracker import FeatureTracker
from core.transformations.special_euclidian_3_dim import SE3

X = gtsam.symbol_shorthand.X
L = gtsam.symbol_shorthand.L


def resolve_pnp_pose(
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


def motion_only_bundle_adjustment(
    pose_k_initial_guess: SE3,
    points_3d: np.ndarray,
    measurements: np.ndarray,
    stereo_k_matrix: gtsam.Cal3_S2Stereo,
    mono_k_matrix: gtsam.Cal3_S2,
) -> SE3:
    """Motion only bundle adjustment using GTSAM."""
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


def draw_features(concatenated: np.ndarray, ft: FeatureTracker) -> None:
    """Draw the features on the concatenated image."""
    for feat in ft.iterate_through_features(["new", "tracked", "stable"]):
        _, left_uv, right_uv = feat.get_active_stereo_pair()
        lx, ly = left_uv
        cv2.circle(concatenated, (int(lx), int(ly)), 1, feat.feature_color(), -1)
        if right_uv is not None:
            rx, ry = right_uv
            cv2.circle(concatenated, (int(rx) + ft.IMAGE_SHAPE["w"], int(ry)), 2, feat.feature_color(), -1)
    cv2.putText(
        concatenated, f"feat count: {ft.feat_count()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
    )
