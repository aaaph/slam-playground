import jax.numpy as jnp

from core.filter.initializer import Initializer
from core.filter.propagator import Propagator
from core.filter.state import State
from core.filter.updater import Updater
from dataset.euroc import EurocDataset

euroc_dataset = EurocDataset.mh_01_easy()

ground_truth = euroc_dataset.ground_truth()
imu = euroc_dataset.imu()
imu_gt_dataset = euroc_dataset.imu_and_ground_truth()
first_ground_truth = ground_truth[0]
first_imu = imu[0]


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
propagator = Propagator(
    ng=1.69e-4,
    na=2.0e-2,
    nba=3.0e-2,
    nbg=1.9393e-08,
)
state = updater.state_update(
    state,
    (first_ground_truth["gt_position"], first_ground_truth["gt_orientation"]),
)
imu_gt_iterator = imu_gt_dataset.to_iterable_dataset()
input()
each_x_imu_items = 20
i = 0
predicted_positions = []
updated_positions = []
for imu_data in imu_gt_iterator:
    timestamp = imu_data["timestamp"]
    gyro = jnp.array(imu_data["gyro"])
    acc = jnp.array(imu_data["acc"])
    gt_position = imu_data["gt_position"]
    gt_orientation = imu_data["gt_orientation"]

    has_ground_truth = imu_data["has_ground_truth"]
    has_imu = imu_data["has_imu"]

    if has_ground_truth and i > each_x_imu_items:
        result, state = propagator.state_propagation(state, (timestamp, gyro, acc))
        predicted_positions.append(state.inertial_state.p)
        state = updater.state_update(state, (gt_position, gt_orientation))
        position = state.inertial_state.p
        orientation = state.inertial_state.q
        position_uncertainty = state.covariance.sigma[0:3, 0:3]
        i = 0
        updated_positions.append(position)
    else:
        result, state = propagator.state_propagation(state, (timestamp, gyro, acc))
        position = state.inertial_state.p
        orientation = state.inertial_state.q
        position_uncertainty = state.covariance.sigma[0:3, 0:3]
        predicted_positions.append(position)
    i += 1
    input()
