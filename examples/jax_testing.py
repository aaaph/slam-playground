import gtsam
from dataset.euroc import EurocDataset, GroundTruth
from gtsam import Point3, Pose3, Rot3
from gtsam.gtsam import InitializePose3

X = gtsam.symbol_shorthand.X
L = gtsam.symbol_shorthand.L


def create_init_prior_factor(ground_truth: GroundTruth) -> gtsam.NonlinearEqualityPose3:
    """Create an initial prior factor for the ground truth."""
    quat = ground_truth["gt_orientation"]
    pos = ground_truth["gt_position"]
    rot_matrix = Rot3.Quaternion(quat[3], quat[0], quat[1], quat[2])
    first_pose = Pose3(rot_matrix, Point3(pos[0], pos[1], pos[2]))
    # prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]))

    return gtsam.NonlinearEqualityPose3(X(0), first_pose)


euroc_dataset = EurocDataset.mh_01_easy()
feat_iterator = euroc_dataset.feat_db_iterate()
first_ground_truth = euroc_dataset.first_ground_truth()
graph = gtsam.NonlinearFactorGraph()
graph.add(create_init_prior_factor(first_ground_truth))
initial_estimate = InitializePose3.initialize(graph)
# initial_estimate.print()

stereo_k_matrix = euroc_dataset.config.stereo.k_rect_left
fx = stereo_k_matrix[0, 0]
fy = stereo_k_matrix[1, 1]
skew = stereo_k_matrix[0, 1]
cx = stereo_k_matrix[0, 2]
cy = stereo_k_matrix[1, 2]
baseline = euroc_dataset.config.stereo.baseline
k = gtsam.Cal3_S2Stereo(
    fx,
    fy,
    skew,
    cx,
    cy,
    baseline,
)
measurement_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
# print(f"stereo k matrix: {k}")

counter = 0
for frame_id, _, feat_in_frame in feat_iterator:
    x_state = X(frame_id + 1)
    initial_estimate.insert(x_state, gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, 0)))
    for feat_id, (uv_left, uv_right) in feat_in_frame.items():
        landmark = L(feat_id)
        ul, v = uv_left
        if uv_right is not None:
            ur, _ = uv_right
            stereo_point = gtsam.StereoPoint2(ul, ur, v)
            stereo_factor = gtsam.GenericStereoFactor3D(stereo_point, measurement_noise, x_state, landmark, k)
            graph.add(stereo_factor)
        # print(f"Feature {feat_id} has left {uv_left} and right {uv_right}")

    counter += 1  # noqa: SIM113
    limit = 2
    if counter > limit:
        break
# initial_estimate.print()
# graph.print()
