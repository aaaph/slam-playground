import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

import gtsam
from core.feature_tracker.feature import Feature
from core.feature_tracker.feature_triangulation import FeatureTriangulation
from core.transformations.frame_resolver import FrameResolver
from core.transformations.special_euclidian_3_dim import SE3
from dataset.euroc import EurocDataset, GroundTruth
from gtsam import InitializePose3, Point3, Pose3, Rot3, Symbol
from logger import spawn_logger

logger = spawn_logger(app="example_gtsam_visual_slam")
X = gtsam.symbol_shorthand.X
L = gtsam.symbol_shorthand.L
measurement_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
distortion_coeffs = np.array([0, 0, 0, 0, 0])


euroc_dataset = EurocDataset.mh_01_easy()
feat_iterator = euroc_dataset.feat_db_iterate()
first_ground_truth = euroc_dataset.first_ground_truth()
# initial_estimate.print()


k_stereo = euroc_dataset.config.stereo.k_matrix_in_gtsam()
k_mono = euroc_dataset.config.cam0.k_matrix_in_gtsam()
baseline = euroc_dataset.config.stereo.baseline
feat_triang = FeatureTriangulation(euroc_dataset.config.stereo.k_rect_left, baseline)
transform_tree = euroc_dataset.config.transform_tree()
frame_resolver = FrameResolver(transform_tree)


def get_initial_camera_pose(ground_truth: GroundTruth) -> Pose3:
    """Get the initial pose from the ground truth."""
    quat = ground_truth["gt_orientation"]
    # x, y, z, w = quat
    pos = ground_truth["gt_position"]

    body_in_world_se3 = SE3.from_quat_and_translation(quat, pos)
    cam0_in_body_se3 = frame_resolver.transform_tree.nodes["cam0"].t_bs
    cam0_in_world_se3 = body_in_world_se3 * cam0_in_body_se3

    cam0_in_world_rot = cam0_in_world_se3.rotation().as_matrix()
    cam0_in_world_translation = cam0_in_world_se3.translation()

    # Create GTSAM Rot3 from matrix (more reliable than Quaternion constructor)
    rot_matrix_gtsam = Rot3(cam0_in_world_rot)

    return Pose3(
        rot_matrix_gtsam,
        Point3(cam0_in_world_translation[0], cam0_in_world_translation[1], cam0_in_world_translation[2]),
    )


def map_config_to_gtsam_k_matrix(k_matrix: np.ndarray, baseline: float) -> gtsam.Cal3_S2Stereo:
    """Map the stereo configuration to a GTSAM K matrix."""
    fx = k_matrix[0, 0]
    fy = k_matrix[1, 1]
    skew = k_matrix[0, 1]
    cx = k_matrix[0, 2]
    cy = k_matrix[1, 2]
    return gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, baseline)


def map_gtsam_pose_to_se3(pose: Pose3) -> SE3:
    """Map a GTSAM pose to a SE3 transformation."""
    rot = Rotation.from_matrix(pose.rotation().matrix())
    translation = pose.translation()
    return SE3(rot, translation)


def resolve_pnp_pose(
    object_points: NDArray[np.float64], image_points: NDArray[np.float64], k_matrix: np.ndarray
) -> Pose3:
    """Resolve the PnP pose."""
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

    return Pose3(
        gtsam.Rot3(new_rotation.as_matrix()), Point3(new_translation[0], new_translation[1], new_translation[2])
    )


init_graph = gtsam.NonlinearFactorGraph()
init_values = InitializePose3.initialize(init_graph)
prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
init_graph.add(gtsam.PriorFactorPose3(X(0), get_initial_camera_pose(first_ground_truth), prior_noise))
drone_camera_in_world_pose = get_initial_camera_pose(first_ground_truth)
init_values.insert(X(0), drone_camera_in_world_pose)
mono_base_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
stereo_base_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
loss_function = gtsam.noiseModel.mEstimator.Huber.Create(1.345)
stereo_noise = gtsam.noiseModel.Robust.Create(loss_function, stereo_base_noise)
mono_noise = gtsam.noiseModel.Robust.Create(loss_function, mono_base_noise)

isam_params = gtsam.ISAM2Params()
isam_params.setRelinearizeThreshold(0.01)
isam_params.relinearizeSkip = 1
isam = gtsam.ISAM2(isam_params)
isam.update(init_graph, init_values)
result = isam.calculateEstimate()

previous_3d_points = {}
counter = 0
for frame_id, ts, feat_in_frame in feat_iterator:
    # print(f"frame_id {frame_id} ts {ts:.0f}")
    frame_graph = gtsam.NonlinearFactorGraph()
    frame_values = gtsam.Values()

    # pose stuff
    x_state = X(frame_id)
    if frame_id > 0:
        # need to resolve PnP problem to get the initial guess
        object_points = []
        image_points = []
        for feat_id, (uv_left, _) in feat_in_frame.items():
            if feat_id in previous_3d_points:
                feat_3d = previous_3d_points[feat_id]
                object_points.append(feat_3d)
                image_points.append((uv_left[0], uv_left[1]))
        object_points = np.array(object_points)
        image_points = np.array(image_points)
        drone_camera_in_world_pose = resolve_pnp_pose(
            object_points, image_points, euroc_dataset.config.stereo.k_rect_left
        )
        logger.debug(f"frame_id {frame_id} position estimate: {drone_camera_in_world_pose.translation()}")
        frame_values.insert(x_state, drone_camera_in_world_pose)
        prev_drone_pose = result.atPose3(X(frame_id - 1))
        T_prev_curr = prev_drone_pose.between(drone_camera_in_world_pose)
        odometry_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        between_factor = gtsam.BetweenFactorPose3(X(frame_id - 1), X(frame_id), T_prev_curr, odometry_noise)
        # frame_graph.add(between_factor)
    # landmark stuff
    for feat_id, (uv_left, uv_right) in feat_in_frame.items():
        landmark = L(feat_id)
        skip_landmark = False
        # check if landmark is already in the graph
        if not result.exists(landmark):
            # need to make initial estimate for the landmark
            # there is a guarantee that landmark has stereo pair
            if uv_right is None:
                continue
            feature = Feature.spawn_from_left_and_right(feat_id, ts, uv_left, uv_right)
            good, feat_in_cam0_translation = feat_triang.make_initial_guess_by_stereo_pair(feature)
            if good:
                drone_camera_in_world_se3 = map_gtsam_pose_to_se3(drone_camera_in_world_pose)

                feat_in_world_translation = (
                    drone_camera_in_world_se3.rotation().as_matrix() @ feat_in_cam0_translation
                    + drone_camera_in_world_se3.translation()
                )
                # the intiail_guess is in camera frame, we need to convert it to world frame
                logger.debug(f"Initial guess for landmark {feat_id}: {feat_in_world_translation} in world frame")

                x, y, z = feat_in_world_translation
                frame_values.insert(landmark, gtsam.Point3(x, y, z))
            else:
                skip_landmark = True
                logger.warning(f"Feature {feat_id} is not good, skipping")

        if skip_landmark:
            continue
        ul, v = uv_left
        if uv_right is not None:
            ur, _ = uv_right
            stereo_point = gtsam.StereoPoint2(ul, ur, v)
            stereo_factor = gtsam.GenericStereoFactor3D(stereo_point, stereo_noise, x_state, landmark, k_stereo)
            frame_graph.add(stereo_factor)
        else:
            mono_point = gtsam.Point2(ul, v)
            mono_factor = gtsam.GenericProjectionFactorCal3_S2(mono_point, mono_noise, x_state, landmark, k_mono)
            # frame_graph.add(mono_factor)
        # logger.debug(f"Feature {feat_id} has left {uv_left} and right {uv_right}")

    logger.debug(f"Frame {frame_id} has {frame_graph.size()} factors")
    t1 = time.time()
    isam.update(frame_graph, frame_values)
    result = isam.calculateEstimate()
    t2 = time.time()
    logger.debug(f"[isam2 update] time taken: {t2 - t1} seconds, factors: {frame_graph.size()}")
    for key in result.keys():  # noqa: SIM118
        sym = Symbol(key)
        landmark_symbol_id = 108
        if sym.chr() == landmark_symbol_id:
            feat_id = sym.index()
            landmark_point = result.atPoint3(key)
            previous_3d_points[feat_id] = landmark_point
            if feat_id == 0:
                # print(f"Landmark 0: {landmark_point}")
                pass
    actual_state = result.atPose3(x_state)
    actual_state_se3 = map_gtsam_pose_to_se3(actual_state)
    logger.debug(f"Actual state: {actual_state_se3} at frame {frame_id}")

    counter += 1
    limit = 400
    if counter > limit:
        break

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
# convert 3d points to numpy array
for feat_3d in previous_3d_points.values():
    ax.scatter(feat_3d[0], feat_3d[1], feat_3d[2])
to_plot = np.array(list(previous_3d_points.values()))
ax.scatter(to_plot[:, 0], to_plot[:, 1], to_plot[:, 2], color="red")
ax.scatter(
    actual_state_se3.translation()[0],
    actual_state_se3.translation()[1],
    actual_state_se3.translation()[2],
    color="blue",
)
plt.show()
