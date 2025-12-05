import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from core.feature_tracker.feature import Feature
from core.feature_tracker.feature_triangulation import FeatureTriangulation
from core.transformations.frame_resolver import FrameResolver, FrameTransform
from core.transformations.special_euclidian_3_dim import SE3
from dataset.euroc import EurocDataset, GroundTruth
from logger import spawn_logger

logger = spawn_logger(app="example_pnp_solve")
euroc_dataset = EurocDataset.mh_01_easy()
feat_iterator = euroc_dataset.feat_db_iterate()
first_ground_truth = euroc_dataset.first_ground_truth()


stereo_k_matrix = euroc_dataset.config.stereo.k_rect_left
baseline = euroc_dataset.config.stereo.baseline
feat_triang = FeatureTriangulation(stereo_k_matrix, baseline)
distortion_coeffs = np.array([0, 0, 0, 0, 0])
transform_tree = euroc_dataset.config.transform_tree()
frame_resolver = FrameResolver(transform_tree)


def get_initial_pose_se3(ground_truth: GroundTruth) -> SE3:
    """Get the initial pose from the ground truth."""
    quat = ground_truth["gt_orientation"]
    pos = ground_truth["gt_position"]
    return SE3.from_quat_and_translation(quat, pos)


def resolve_pnp_pose(object_points: NDArray[np.float64], image_points: NDArray[np.float64]) -> SE3:
    """Resolve the PnP pose."""
    _, rvec, tvec, inliners = cv2.solvePnPRansac(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=stereo_k_matrix,
        distCoeffs=distortion_coeffs,
        iterationsCount=100,
        reprojectionError=3.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    rvec, tvec = cv2.solvePnPRefineLM(
        objectPoints=object_points[inliners],
        imagePoints=image_points[inliners],
        cameraMatrix=stereo_k_matrix,
        distCoeffs=distortion_coeffs,
        rvec=rvec,
        tvec=tvec,
    )
    rot, _ = cv2.Rodrigues(rvec)
    new_rotation = Rotation.from_matrix(rot.transpose())
    new_translation = -new_rotation.as_matrix() @ tvec.reshape(1, 3).flatten()
    cam0_in_world_se3 = SE3(new_rotation, new_translation)
    cam0_in_world_transform = FrameTransform(source="world", target="cam0", transform=cam0_in_world_se3)
    return (
        frame_resolver.with_dynamic(cam0_in_world_transform)
        .from_("cam0")
        .move_to("body")
        .apply_se3(cam0_in_world_se3)
    )


previous_3d_points = {}

counter = 0
for frame_id, ts, feat_in_frame in feat_iterator:
    if frame_id == 0:
        drone_pose_estimate = get_initial_pose_se3(first_ground_truth)
        logger.debug(f"Initial pose: {drone_pose_estimate}")
    else:
        # map uv with 3d points
        object_points = []
        image_points = []
        for feat_id, (uv_left, _) in feat_in_frame.items():
            if feat_id in previous_3d_points:
                feat_3d = previous_3d_points[feat_id]
                object_points.append(feat_3d)
                image_points.append((uv_left[0], uv_left[1]))
        object_points = np.array(object_points)
        image_points = np.array(image_points)
        drone_pose_estimate = resolve_pnp_pose(object_points, image_points)
        logger.debug(f"Pose estimate: {drone_pose_estimate}")
    # landmark stuff
    for feat_id, (uv_left, uv_right) in feat_in_frame.items():
        if frame_id == 0:
            feature = Feature.spawn_from_left_and_right(feat_id, ts, uv_left, uv_right)
            initial_guess = feat_triang.make_initial_guess(feature)
            drone_in_world = FrameTransform(source="world", target="body", transform=drone_pose_estimate)
            feat_in_world_translation = (
                frame_resolver.with_dynamic(drone_in_world)
                .from_("cam0")
                .move_to("world")
                .apply_vector(initial_guess)
            )
            # print(f"Initial guess for landmark {feat_id}: {feat_in_world_translation} in world frame")
            previous_3d_points[feat_id] = feat_in_world_translation
        else:
            pass

    counter += 1  # noqa: SIM113
    limit = 1
    if counter > limit:
        break
