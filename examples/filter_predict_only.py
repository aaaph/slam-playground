import jax.numpy as jnp

from core.filter.initializer import Initializer
from core.filter.propagator import Propagator
from core.filter.state import State
from dataset.euroc import EurocDataset

euroc_dataset = EurocDataset.mh_01_easy()
ground_truth = euroc_dataset.ground_truth()
first_ground_truth = ground_truth[0]

initializer = Initializer()
row = jnp.array(
    [
        first_ground_truth["gt_position"][0],
        first_ground_truth["gt_position"][1],
        first_ground_truth["gt_position"][2],
        first_ground_truth["gt_orientation"][0],
        first_ground_truth["gt_orientation"][1],
        first_ground_truth["gt_orientation"][2],
        first_ground_truth["gt_orientation"][3],
        first_ground_truth["gt_velocity"][0],
        first_ground_truth["gt_velocity"][1],
        first_ground_truth["gt_velocity"][2],
        first_ground_truth["gt_gyro_bias"][0],
        first_ground_truth["gt_gyro_bias"][1],
        first_ground_truth["gt_gyro_bias"][2],
        first_ground_truth["gt_acc_bias"][0],
        first_ground_truth["gt_acc_bias"][1],
        first_ground_truth["gt_acc_bias"][2],
    ]
)
state = initializer.initialize_from_row(State(), timestamp=first_ground_truth["timestamp"], array=row)

propagator = Propagator(ng=0.0, na=0.0, nba=0.0, nbg=0.0)
