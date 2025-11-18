from typing import Literal

import cv2
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.spatial.transform import Rotation

from core.feature_tracker.feature import Feature
from core.feature_tracker.feature_tracker import FeatureTracker
from core.filter.augmentator import Augmentator
from core.filter.initializer import Initializer
from core.filter.propagator import Propagator
from core.filter.state import CameraClone, State
from core.filter.updater import Updater
from core.transformations.helpers import skew
from dataset.euroc import EurocDataset


def measurement_jacobian(  # noqa: PLR0913
    uv: np.ndarray,
    cam_id: Literal[0, 1],
    p_f_in_w: np.ndarray,
    clone: CameraClone,
    k_matrices: tuple[np.ndarray, np.ndarray],
    camera_body_sensor_transforms: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Measurement Jacobian for the Multi-State Constraint Kalman Filter."""
    p_i_in_w = clone.p
    q_w_to_i = clone.q
    t_cam_imu = camera_body_sensor_transforms[0] if cam_id == 0 else camera_body_sensor_transforms[1]
    k_matrix = k_matrices[0] if cam_id == 0 else k_matrices[1]
    fx = k_matrix[0, 0]
    fy = k_matrix[1, 1]
    cx = k_matrix[0, 2]
    cy = k_matrix[1, 2]

    t_w_to_i = np.eye(4)
    t_w_to_i[:3, :3] = Rotation.from_quat(q_w_to_i).as_matrix()
    t_w_to_i[:3, 3] = p_i_in_w
    t_w_to_cam = t_w_to_i @ t_cam_imu
    p_cam_in_w = t_w_to_cam[:3, 3]
    rot_w_to_cam = t_w_to_cam[:3, :3]
    rot_cam_to_w = rot_w_to_cam.T

    p_f_in_cam = rot_w_to_cam.T @ (p_f_in_w - p_cam_in_w)
    x, y, z = p_f_in_cam
    dzn_dp = np.eye(2, 3, dtype=np.float32)
    dzn_dp[0, 0] = 1 / z
    dzn_dp[1, 1] = 1 / z
    dzn_dp[0, 2] = -x / (z * z)
    dzn_dp[1, 2] = -y / (z * z)

    dz_dzn = np.eye(2, 2, dtype=np.float32)
    # 123dz_dzn[0, 0] = fx
    # 123dz_dzn[1, 1] = fy
    dz_dp = dz_dzn @ dzn_dp  # 2x3

    dp_f_in_c_dxc = np.eye(3, 6, dtype=np.float32)
    dp_f_in_c_dxc[:, :3] = -skew(p_f_in_cam)
    dp_f_in_c_dxc[:, 3:] = -rot_cam_to_w

    h_f = dz_dp @ rot_cam_to_w  # (2,3)
    h_x = dz_dp @ dp_f_in_c_dxc  # (2,6)

    a = h_x.copy()  # 2x6
    u = np.zeros(6, dtype=np.float32)  # 6x1
    g = np.array([0, 0, -9.81], dtype=np.float32)
    u[:3] = Rotation.from_quat(q_w_to_i).as_matrix() @ g
    u[3:] = skew(p_f_in_w - p_i_in_w) @ g

    h_x = a - (a @ u)[:, None] * u / (u @ u)

    # Convert uv from pixel coordinates to normalized coordinates
    uv_normalized = np.array([(uv[0] - cx) / fx, (uv[1] - cy) / fy])

    # Project feature point to normalized coordinates
    uv_projected = np.array([(x / z), (y / z)])

    # Calculate residual in normalized coordinates
    resudial = uv_normalized - uv_projected
    return resudial, uv_projected, h_x, h_f, uv_normalized


def feature_jacobian(
    state: State,
    feat: Feature,
    k_matrices: tuple[np.ndarray, np.ndarray],
    t_bs_matrices: tuple[np.ndarray, np.ndarray],
) -> [np.ndarray, np.ndarray]:
    """Feature Jacobian for the Multi-State Constraint Kalman Filter."""
    p_f_in_w = feat.p_fw
    uv_real = []
    uv_projected = []
    jacobian_row_size = feat.size * 2
    r_j = np.zeros(jacobian_row_size, dtype=np.float32)
    h_fj = np.zeros((jacobian_row_size, 3), dtype=np.float32)
    h_xj = np.zeros((jacobian_row_size, 15 + (30) * 6), dtype=np.float32)
    for idx, (ts, cam_id, u, v) in enumerate(feat.iterate()):
        clone = state.sliding_window.get_by_timestamp(ts)
        if clone is None:
            msg = f"Clone not found for timestamp: {ts}"
            raise ValueError(msg)
        uv = np.array([u, v])
        ri, uvp, h_xi, h_fi, uvn = measurement_jacobian(uv, cam_id, p_f_in_w, clone, k_matrices, t_bs_matrices)
        r_j[idx * 2 : (idx + 1) * 2] = ri
        h_fj[idx * 2 : (idx + 1) * 2, :3] = h_fi
        col = idx // 2
        h_xj[idx * 2 : (idx + 1) * 2, 15 + 6 * col : 15 + 6 * (col + 1)] = h_xi
        uv_real.append(uvn)
        uv_projected.append(uvp)
    u, _, _ = np.linalg.svd(h_fj)
    a = u[:, 3:]
    h_x0 = a.T @ h_xj
    r0 = a.T @ r_j

    uv_real = np.array(uv_real)
    uv_projected = np.array(uv_projected)
    return uv_real, uv_projected, r0, h_x0


chi_squared_test_dict = {}
for i in range(1, 1000):
    chi_squared_test_dict[i] = scipy.stats.chi2.ppf(0.05, i)


def gating_test(p: np.ndarray, r: np.ndarray, h: np.ndarray) -> bool:
    """Chi2 test for the Multi-State Constraint Kalman Filter."""
    """  p1 = h @ p @ h.T
    p2 = 0.035**2 * np.identity(len(h))
    gamma = r @ np.linalg.solve(p1 + p2, r)
    return gamma < chi_squared_test_dict[dof] """
    r = r.reshape(-1, 1)
    m = h.shape[0]
    s = h @ p @ h.T + (0.5**2) * np.identity(m)
    chol = np.linalg.cholesky(s)
    y = np.linalg.solve(chol, r)
    gamma = float(y.T @ y)
    dof = m
    return gamma < chi_squared_test_dict[dof]


def draw_features(concatenated: np.ndarray, ft: FeatureTracker) -> None:
    """Draw the features on the concatenated image."""
    my_feat_id = 50
    for feat in ft.iterate_through_features(["new", "tracked", "stable"]):
        _, left_uv, right_uv = feat.get_active_stereo_pair()
        lx, ly = left_uv
        cv2.circle(concatenated, (int(lx), int(ly)), 1, feat.feature_color(), -1)
        if right_uv is not None:
            rx, ry = right_uv
            cv2.circle(concatenated, (int(rx) + ft.IMAGE_SHAPE["w"], int(ry)), 1, feat.feature_color(), -1)
        if feat.feat_id == my_feat_id:
            cv2.circle(concatenated, (int(lx), int(ly)), 5, (200, 150, 255), -1)
    cv2.putText(
        concatenated, f"feat count: {ft.feat_count()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
    )


euroc_dataset = EurocDataset.mh_01_easy()
ds = euroc_dataset.all()

k_matrix_left = euroc_dataset.config.stereo.k_rect_left
k_matrix_right = euroc_dataset.config.stereo.k_rect_right
k_matrix_left_inv = np.linalg.inv(k_matrix_left)
k_matrix_right_inv = np.linalg.inv(k_matrix_right)

t_bs_left = euroc_dataset.config.cam0.body_sensor_transform
t_bs_right = euroc_dataset.config.cam1.body_sensor_transform

first_ground_truth = euroc_dataset.first_ground_truth()

ft = FeatureTracker(euroc_dataset.config.stereo)
state = Initializer().initialize_from_dict(
    State(),
    first_ground_truth["timestamp"],
    dictionary={
        "position": first_ground_truth["gt_position"],
        "orientation": first_ground_truth["gt_orientation"],
        "velocity": first_ground_truth["gt_velocity"],
        "acc_bias": first_ground_truth["gt_acc_bias"],
        "gyro_bias": first_ground_truth["gt_gyro_bias"],
    },
)
updater = Updater()
propagator = Propagator.from_imu_config(euroc_dataset.config.imu0)
augmentator = Augmentator()
my_feat_observations: list[tuple[np.ndarray, np.ndarray, np.ndarray, Literal[0, 1]]] = []
last_p_fw: np.ndarray | None = None
uv_real = []
uv_projected = []
update_p_i_in_w = None
sliding_window_poses = []
for item in ds.to_iterable_dataset():
    timestamp = item["timestamp"]
    has_stereo = item["stereo"][0] is not None
    has_imu = item["has_imu"]
    has_ground_truth = item["has_ground_truth"]
    gyro = item["gyro"]
    acc = item["acc"]
    if has_imu:
        result, state = propagator.state_propagation(state, (timestamp, gyro, acc))
        if not result:
            continue

    if has_stereo and has_imu:
        stereo = item["stereo"]
        clone, state = augmentator.augment_clone(state)
        left, right = ft.feed(timestamp, (stereo[0], stereo[1]))
        left_out, right_out = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR), cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
        concatenated = np.concatenate([left_out, right_out], axis=1)

        oldest_ts = ft.get_oldest_timestamp()
        oldest_candidates = state.sliding_window.get_oldest_than(oldest_ts)

        my_feat_id = 50
        for feat in ft.iterate_through_features(["new", "tracked", "stable"]):
            result = feat.update_linear_system(
                clone, (k_matrix_left_inv, k_matrix_right_inv), (t_bs_left, t_bs_right)
            )
            if feat.feat_id == my_feat_id:
                my_feat_observations.extend(result)
                if feat.p_fw is not None:
                    last_p_fw = feat.p_fw
        max_sliding_window_size = 30
        if state.sliding_window.size() >= max_sliding_window_size:
            h_x_list = []
            r_list = []
            my_feat = ft.get_feature_by_id(my_feat_id)
            uvr, uvp, rj, h_xj = feature_jacobian(
                state, my_feat, (k_matrix_left, k_matrix_right), (t_bs_left, t_bs_right)
            )
            uv_real.extend(uvr)
            uv_projected.extend(uvp)
            h_x_list.append(h_xj)
            r_list.append(rj)
            success = gating_test(state.covariance.sigma, rj, h_xj)

            h_x = np.vstack(h_x_list)  # we have only 1 feature
            r = np.concatenate(r_list)  # we have only 1 feature

            # UPDATE FLOW START:
            # 1. Gating test
            """ print(f"h_x.shape: {h_x.shape[0], h_x.shape[1]}")
            if h_x.shape[0] > h_x.shape[1]:
                q, R = np.linalg.qr(h_x)
                h_thin = R
                r_thin = q.T @ r
            else:
                h_thin = h_x
                r_thin = r
            """
            h_thin = h_x
            r_thin = r
            p = state.covariance.sigma
            s = h_thin @ p @ h_thin.T + (0.1**2) * np.identity(h_thin.shape[0])
            k_transpose = np.linalg.solve(s, h_thin @ p)
            k = k_transpose.T
            delta_x = k @ r_thin

            delta_x_imu = delta_x[:15]
            delta_p = delta_x_imu[:3]
            delta_v = delta_x_imu[6:9]
            delta_rot = delta_x_imu[3:6]
            delta_b_a = delta_x_imu[9:12]
            delta_b_g = delta_x_imu[12:15]

            def apply_new_rotation(rotation_error: jax.Array, rotation_pred: Rotation) -> jax.Array:
                """Apply the new rotation to the rotation."""
                delta_rotation = Rotation.from_rotvec(rotation_error)
                new_q = (delta_rotation * rotation_pred).as_quat()
                return new_q / jnp.linalg.norm(new_q)

            def map_clone_pose(clone: CameraClone, dx: np.ndarray = delta_x) -> tuple[np.ndarray, np.ndarray]:
                """Map the pose of the clone."""
                clone_id = clone.clone_id
                delta_x_clone = dx[15 + 6 * clone_id : 15 + 6 * (clone_id + 1)]
                delta_p_clone = delta_x_clone[:3]
                delta_q_clone = delta_x_clone[3:7]
                new_p = clone.p + delta_p_clone
                new_q = apply_new_rotation(delta_q_clone, Rotation.from_quat(clone.q))
                before_p = clone.p
                sliding_window_poses.append((before_p, new_p))
                return new_p, new_q

            state = state.map_inertial_state(
                lambda x, dp=delta_p, dv=delta_v, dba=delta_b_a, dbg=delta_b_g, drot=delta_rot: (
                    x.map_position(lambda pos, d=dp: jnp.add(pos, d))
                    .map_velocity(lambda v, d=dv: jnp.add(v, d))
                    .map_acc_bias(lambda b_a, d=dba: jnp.add(b_a, d))
                    .map_gyro_bias(lambda b_g, d=dbg: jnp.add(b_g, d))
                    .map_orientation(lambda quat, d=drot: apply_new_rotation(d, Rotation.from_quat(quat)))
                )
            ).map_poses_in_sliding_window(map_clone_pose)

            # UPDATE FLOW END
            update_p_i_in_w = state.inertial_state.get_pose().at[0:3].get()

            I_KH = np.eye(len(k)) - k @ h_thin
            sigma_new = I_KH @ p
            sigma_new = (sigma_new + sigma_new.T) / 2
            state.apply_covariance(sigma_new)
            break

        draw_features(concatenated, ft)
        cv2.imshow("concatenated", concatenated)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break
        else:
            continue

fig = plt.figure()
ax = fig.add_subplot()

feat_data = []
for _timestamp, p_iw, _q_wi, _b_i, _cam_id, _u, _v in my_feat_observations:
    ax.plot(p_iw[0], p_iw[1], "bo")

if update_p_i_in_w is not None:
    update_p_i_in_w = np.array(update_p_i_in_w)
    ax.plot(update_p_i_in_w[0], update_p_i_in_w[1], "ro")

for before_p, new_p in sliding_window_poses:
    ax.plot(before_p[0], before_p[1], "go")
    ax.plot(new_p[0], new_p[1], "ro")
    ax.plot([before_p[0], new_p[0]], [before_p[1], new_p[1]], "g-", linewidth=2, alpha=0.7)
fig2 = plt.figure()
ax2 = fig2.add_subplot()
for i, uv in enumerate(uv_real):
    ax2.plot(uv[0], uv[1], "bo")
    ax2.plot(uv_projected[i][0], uv_projected[i][1], "ro")
    ax2.plot([uv[0], uv_projected[i][0]], [uv[1], uv_projected[i][1]], "g-", linewidth=2, alpha=0.7)


plt.show()
