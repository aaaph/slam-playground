import numpy as np

import gtsam
from logger import spawn_logger

logger = spawn_logger(app="example_fixed_lag_smoother")


def batch_fixed_lag_smoother_example() -> None:
    """Batch fixed lag smoother example."""
    # Define a batch fixed lag smoother, which uses
    # Levenberg-Marquardt to perform the nonlinear optimization
    lag = 2.0
    smoother_batch = gtsam.BatchFixedLagSmoother(lag)

    # Create containers to store the factors and linearization points
    # that will be sent to the smoothers
    new_factors = gtsam.NonlinearFactorGraph()
    new_values = gtsam.Values()
    new_timestamps = {}

    # Create  a prior on the first pose, placing it at the origin
    prior_mean = gtsam.Pose2(0, 0, 0)
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))
    x1 = 0
    new_factors.push_back(gtsam.PriorFactorPose2(x1, prior_mean, prior_noise))
    new_values.insert(x1, prior_mean)
    new_timestamps[x1] = 0.0

    delta_time = 0.25
    time = 0.25

    while time <= 3.0:  # noqa: PLR2004
        previous_key = int(1000 * (time - delta_time))
        current_key = int(1000 * time)

        # assign current key to the current timestamp
        new_timestamps[current_key] = time

        # Add a guess for this pose to the new values
        # Assume that the robot moves at 2 m/s. Position is time[s] * 2[m/s]
        current_pose = gtsam.Pose2(time * 2, 0, 0)
        new_values.insert(current_key, current_pose)

        # Add odometry factors from two different sources with different error
        # stats
        odometry_measurement_1 = gtsam.Pose2(0.61, -0.08, 0.02)
        odometry_noise_1 = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
        new_factors.push_back(
            gtsam.BetweenFactorPose2(previous_key, current_key, odometry_measurement_1, odometry_noise_1)
        )

        odometry_measurement_2 = gtsam.Pose2(0.47, 0.03, 0.01)
        odometry_noise_2 = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.05]))
        new_factors.push_back(
            gtsam.BetweenFactorPose2(previous_key, current_key, odometry_measurement_2, odometry_noise_2)
        )

        # Update the smoothers with the new factors. In this case,
        # one iteration must pass for Levenberg-Marquardt to accurately
        # estimate
        if time >= 0.50:  # noqa: PLR2004
            smoother_batch.update(new_factors, new_values, new_timestamps)
            logger.info("Timestamp = " + str(time) + ", Key = " + str(current_key))  # noqa: G003
            logger.info(smoother_batch.calculateEstimatePose2(current_key))

            new_timestamps.clear()
            new_values.clear()
            new_factors.resize(0)

        time += delta_time


if __name__ == "__main__":
    batch_fixed_lag_smoother_example()
    logger.info("Example complete")
