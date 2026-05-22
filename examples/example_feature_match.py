import argparse

import cv2
import numpy as np

from core.camera_model.stereo_camera_model import StereoCameraModel
from core.loop_closure.vpr_detector import VPRDetector
from core.loop_closure.vpr_frame import VPRFrame
from core.loop_closure.vpr_matcher import VPRMatcher
from dataset.euroc import EurocDataset
from logger import spawn_logger

MIN_ESSENTIAL_POINTS = 5
MIN_RIGID3D_POINTS = 3
ESSENTIAL_RANSAC_PROB = 0.999
RIGID3D_RANSAC_THRESHOLD_M = 0.15
RIGID3D_RANSAC_ITERATIONS = 100
RIGID3D_RANSAC_SEED = 42
logger = spawn_logger(app="example_feature_match")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        msg = f"expected a positive integer, got {value}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match VPR features and run essential + rigid 3D geometric verification."
    )
    parser.add_argument(
        "--n-features",
        type=_positive_int,
        default=1000,
        help="Total ORB feature budget before grid splitting.",
    )
    parser.add_argument(
        "--grid",
        type=_positive_int,
        nargs=2,
        default=(8, 8),
        metavar=("ROWS", "COLS"),
        help="Detector grid as rows and columns.",
    )
    parser.add_argument(
        "--show-opencv",
        action="store_true",
        help="Show OpenCV windows with descriptor matches and rigid 3D inlier matches.",
    )
    return parser.parse_args()


def _essential_inlier_mask(
    query_vpr_frame: VPRFrame,
    reference_vpr_frame: VPRFrame,
    matches: list[cv2.DMatch],
    normalized_threshold: float,
) -> np.ndarray:
    if len(matches) < MIN_ESSENTIAL_POINTS:
        return np.zeros((len(matches),), dtype=bool)

    query_bearings = query_vpr_frame.bearings[[match.queryIdx for match in matches]]
    reference_bearings = reference_vpr_frame.bearings[[match.trainIdx for match in matches]]
    query_norm = query_bearings[:, :2] / query_bearings[:, 2:3]
    reference_norm = reference_bearings[:, :2] / reference_bearings[:, 2:3]

    _essential_matrix, inliers = cv2.findEssentialMat(
        reference_norm,
        query_norm,
        np.eye(3),
        method=cv2.RANSAC,
        prob=ESSENTIAL_RANSAC_PROB,
        threshold=normalized_threshold,
    )
    if inliers is None:
        return np.zeros((len(matches),), dtype=bool)
    return inliers.ravel().astype(bool)


def _valid_3d_points(points: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(points), axis=1) & (points[:, 2] > 0.0)


def _estimate_rigid_transform(query_points: np.ndarray, reference_points: np.ndarray) -> np.ndarray:
    query_centroid = np.mean(query_points, axis=0)
    reference_centroid = np.mean(reference_points, axis=0)
    query_centered = query_points - query_centroid
    reference_centered = reference_points - reference_centroid

    covariance = query_centered.T @ reference_centered
    u_matrix, _singular_values, vt_matrix = np.linalg.svd(covariance)
    rotation = vt_matrix.T @ u_matrix.T
    if np.linalg.det(rotation) < 0:
        vt_matrix[-1, :] *= -1
        rotation = vt_matrix.T @ u_matrix.T
    translation = reference_centroid - rotation @ query_centroid

    reference_t_query = np.eye(4, dtype=np.float64)
    reference_t_query[:3, :3] = rotation
    reference_t_query[:3, 3] = translation
    return reference_t_query


def _rigid_transform_residuals(
    query_points: np.ndarray,
    reference_points: np.ndarray,
    reference_t_query: np.ndarray,
) -> np.ndarray:
    transformed_query = query_points @ reference_t_query[:3, :3].T + reference_t_query[:3, 3]
    return np.linalg.norm(reference_points - transformed_query, axis=1)


def _estimate_rigid_transform_ransac(query_points: np.ndarray, reference_points: np.ndarray) -> np.ndarray | None:
    if query_points.shape[0] < MIN_RIGID3D_POINTS:
        return None

    rng = np.random.default_rng(RIGID3D_RANSAC_SEED)
    best_inliers = np.empty((0,), dtype=bool)
    best_median_residual = np.inf
    best_transform = None

    for _ in range(RIGID3D_RANSAC_ITERATIONS):
        sample_indices = rng.choice(query_points.shape[0], size=MIN_RIGID3D_POINTS, replace=False)
        reference_t_query = _estimate_rigid_transform(
            query_points[sample_indices],
            reference_points[sample_indices],
        )
        residuals = _rigid_transform_residuals(query_points, reference_points, reference_t_query)
        inliers = residuals < RIGID3D_RANSAC_THRESHOLD_M
        inliers_count = int(np.count_nonzero(inliers))
        median_residual = np.inf if inliers_count == 0 else float(np.median(residuals[inliers]))

        if inliers_count > int(np.count_nonzero(best_inliers)) or (
            inliers_count == int(np.count_nonzero(best_inliers)) and median_residual < best_median_residual
        ):
            best_inliers = inliers
            best_median_residual = median_residual
            best_transform = reference_t_query

    if best_transform is None:
        return None
    if int(np.count_nonzero(best_inliers)) >= MIN_RIGID3D_POINTS:
        return _estimate_rigid_transform(query_points[best_inliers], reference_points[best_inliers])
    return best_transform


def _rigid3d_verification(
    query_vpr_frame: VPRFrame,
    reference_vpr_frame: VPRFrame,
    essential_matches: list[cv2.DMatch],
) -> tuple[np.ndarray | None, list[cv2.DMatch], np.ndarray, np.ndarray]:
    if len(essential_matches) < MIN_RIGID3D_POINTS:
        return None, [], np.empty((0,), dtype=bool), np.empty((0,), dtype=np.float64)

    query_points = query_vpr_frame.points_xyz[[match.queryIdx for match in essential_matches]].astype(np.float64)
    reference_points = reference_vpr_frame.points_xyz[[match.trainIdx for match in essential_matches]].astype(
        np.float64
    )
    valid_mask = _valid_3d_points(query_points) & _valid_3d_points(reference_points)
    valid_matches = [match for match, is_valid in zip(essential_matches, valid_mask, strict=False) if is_valid]
    query_points = query_points[valid_mask]
    reference_points = reference_points[valid_mask]

    if query_points.shape[0] < MIN_RIGID3D_POINTS:
        return None, valid_matches, np.zeros((len(valid_matches),), dtype=bool), np.empty((0,), dtype=np.float64)

    reference_t_query = _estimate_rigid_transform_ransac(query_points, reference_points)
    if reference_t_query is None:
        return (
            None,
            valid_matches,
            np.zeros((len(valid_matches),), dtype=bool),
            np.full((len(valid_matches),), np.inf, dtype=np.float64),
        )

    residuals = _rigid_transform_residuals(query_points, reference_points, reference_t_query)
    inlier_mask = residuals < RIGID3D_RANSAC_THRESHOLD_M
    if int(np.count_nonzero(inlier_mask)) >= MIN_RIGID3D_POINTS:
        reference_t_query = _estimate_rigid_transform(query_points[inlier_mask], reference_points[inlier_mask])
        residuals = _rigid_transform_residuals(query_points, reference_points, reference_t_query)
        inlier_mask = residuals < RIGID3D_RANSAC_THRESHOLD_M

    return reference_t_query, valid_matches, inlier_mask, residuals


def main() -> None:
    """Run feature matching and optional OpenCV visualization."""
    args = _parse_args()

    euroc_dataset = EurocDataset.mh_01_easy()
    stereo_dataset = euroc_dataset.stereo().with_format("numpy")
    stereo_camera = StereoCameraModel.from_cameras_config(euroc_dataset.config.cam0, euroc_dataset.config.cam1)
    vpr_detector = VPRDetector.from_stereo_ctx(
        stereo_camera.as_stereo_ctx(),
        n_features=args.n_features,
        grid=tuple(args.grid),
    )
    vpr_matcher = VPRMatcher.default_factory()

    query_frame = stereo_dataset[593]
    query_left, query_right = query_frame["stereo"][0], query_frame["stereo"][1]
    query_left, query_right = stereo_camera.process_stereo(query_left, query_right)
    query_detection = vpr_detector.detect_stereo(query_left, query_right)
    query_vpr_frame = VPRFrame.from_detection(
        frame_id=0,
        kf_id=0,
        timestamp=0.0,
        detection=query_detection,
    )

    reference_frame = stereo_dataset[162]
    reference_left, reference_right = reference_frame["stereo"][0], reference_frame["stereo"][1]
    reference_left, reference_right = stereo_camera.process_stereo(reference_left, reference_right)
    reference_detection = vpr_detector.detect_stereo(reference_left, reference_right)
    reference_vpr_frame = VPRFrame.from_detection(
        frame_id=1,
        kf_id=1,
        timestamp=0.0,
        detection=reference_detection,
    )

    matches = vpr_matcher.match(query_vpr_frame.descriptors, reference_vpr_frame.descriptors)
    normalized_threshold = 1.0 / stereo_camera.stereo_k[0, 0]
    essential_inlier_mask = _essential_inlier_mask(
        query_vpr_frame,
        reference_vpr_frame,
        matches,
        normalized_threshold,
    )
    essential_matches = [
        match for match, is_inlier in zip(matches, essential_inlier_mask, strict=False) if is_inlier
    ]

    reference_t_query, rigid_matches, rigid_inlier_mask, rigid_residuals = _rigid3d_verification(
        query_vpr_frame,
        reference_vpr_frame,
        essential_matches,
    )
    rigid_inlier_matches = [
        match for match, is_inlier in zip(rigid_matches, rigid_inlier_mask, strict=False) if is_inlier
    ]

    logger.info(f"descriptor matches: {len(matches)}")
    logger.info(f"essential inliers: {int(np.count_nonzero(essential_inlier_mask))}/{len(matches)}")
    logger.info(f"valid 3D matches: {len(rigid_matches)}")
    logger.info(f"rigid 3D inliers: {int(np.count_nonzero(rigid_inlier_mask))}/{len(rigid_matches)}")
    if np.any(rigid_inlier_mask):
        logger.info(f"rigid 3D median residual, m: {float(np.median(rigid_residuals[rigid_inlier_mask])):.4f}")
    if reference_t_query is not None:
        logger.info(f"reference_T_query:\n{np.array2string(reference_t_query, precision=4, suppress_small=True)}")

    if args.show_opencv:
        query_left_out = cv2.cvtColor(query_left, cv2.COLOR_GRAY2BGR)
        query_left_out = cv2.drawKeypoints(
            query_left_out,
            query_detection.keypoints,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        reference_left_out = cv2.cvtColor(reference_left, cv2.COLOR_GRAY2BGR)
        reference_left_out = cv2.drawKeypoints(
            reference_left_out,
            reference_detection.keypoints,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        matched_vis = cv2.drawMatches(
            query_left_out,
            query_detection.keypoints,
            reference_left_out,
            reference_detection.keypoints,
            matches,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        rigid_inlier_vis = cv2.drawMatches(
            query_left,
            query_detection.keypoints,
            reference_left,
            reference_detection.keypoints,
            rigid_inlier_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        cv2.imshow("matched_vis", matched_vis)
        cv2.imshow("rigid_inlier_vis", rigid_inlier_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
