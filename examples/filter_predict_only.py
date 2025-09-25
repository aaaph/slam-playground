from core.filter.initializer import Initializer
from core.filter.propagator import Propagator
from core.filter.state import State
from dataset.euroc import EurocDataset

euroc_dataset = EurocDataset.mh_01_easy()

ground_truth = euroc_dataset.ground_truth()
imu = euroc_dataset.imu()
first_ground_truth = ground_truth[0]
first_imu = imu[0]

initializer = Initializer()

state = initializer.initialize_from_row(
    State(),
    (
        first_imu["timestamp"],
        first_ground_truth["gt_position"],
        first_ground_truth["gt_orientation"],
        first_ground_truth["gt_velocity"],
        first_ground_truth["gt_gyro_bias"],
        first_ground_truth["gt_acc_bias"],
    ),
)

propagator = Propagator(ng=0.0, na=0.0, nba=0.0, nbg=0.0)
