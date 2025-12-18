import pytest
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

traj_est = PoseTrajectory3D()
traj_ref = file_interface.read_euroc_csv_trajectory("path_to_euroc_gt_trajectory.csv")

traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, 0.01)
pose_relation = metrics.PoseRelation.translation_part
ape_metric = metrics.APE(pose_relation)
ape_metric.process_data((traj_ref, traj_est))

rmse = ape_metric.get_statistic(metrics.StatisticsType.rmse)
mean = ape_metric.get_statistic(metrics.StatisticsType.mean)
min_error = ape_metric.get_statistic(metrics.StatisticsType.min)
max_error = ape_metric.get_statistic(metrics.StatisticsType.max)


class TestEurocMH01Easy01:
    """Regression test for the Euroc MH_01_easy dataset scenario."""


@pytest.mark.regression
def test_euroc_mh_easy_01():
    """Test the Euroc MH_01_easy dataset scenario."""
