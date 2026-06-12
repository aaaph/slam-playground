import gtsam
import numpy as np

X = gtsam.symbol_shorthand.X
V = gtsam.symbol_shorthand.V
B = gtsam.symbol_shorthand.B

dt = 0.1
integration_time = 1.0
gravity = np.array([0.0, 0.0, -9.81])

measured_acc = -gravity
measured_gyro = np.array([0.0, 0.0, 0.0])

accel_noise_sigma = 1e-4
gyro_noise_sigma = 1e-4
integration_noise_sigma = 1e-5
bias_acc_walk_sigma = 1e-4
bias_gyro_walk_sigma = 1e-4

params_imu = gtsam.PreintegrationParams(gravity)
params_imu.setAccelerometerCovariance(np.eye(3) * accel_noise_sigma**2)
params_imu.setGyroscopeCovariance(np.eye(3) * gyro_noise_sigma**2)
params_imu.setIntegrationCovariance(np.eye(3) * integration_noise_sigma**2)

bias_0 = gtsam.imuBias.ConstantBias()

pim_imu = gtsam.PreintegratedImuMeasurements(params_imu, bias_0)

for _ in range(int(integration_time / dt)):
    pim_imu.integrateMeasurement(measured_acc, measured_gyro, dt)


pose_0 = gtsam.Pose3()
vel_0 = np.zeros(3)
nav_state_0 = gtsam.NavState(pose_0, vel_0)

# how to predict
predicted_nav_state = pim_imu.predict(nav_state_0, bias_0)

pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01] * 3 + [0.01] * 3))
vel_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01] * 3))
bias_prior_sigma = 0.01
bias_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([bias_prior_sigma] * 6))
bias_walk_noise_model = gtsam.noiseModel.Diagonal.Sigmas(
    np.concatenate([np.full(3, bias_acc_walk_sigma), np.full(3, bias_gyro_walk_sigma)]) * np.sqrt(integration_time)
)

graph_imu = gtsam.NonlinearFactorGraph()
graph_imu.add(gtsam.PriorFactorPose3(X(0), pose_0, pose_noise))
graph_imu.add(gtsam.PriorFactorVector(V(0), vel_0, vel_noise))
graph_imu.add(gtsam.PriorFactorConstantBias(B(0), bias_0, bias_noise))

for k in range(2):
    graph_imu.add(gtsam.ImuFactor(X(k), V(k), X(k + 1), V(k + 1), B(k), pim_imu))
    graph_imu.add(
        gtsam.BetweenFactorConstantBias(B(k), B(k + 1), gtsam.imuBias.ConstantBias(), bias_walk_noise_model)
    )

initial_estimate = gtsam.Values()
for k in range(3):
    initial_estimate.insert(X(k), pose_0)
    initial_estimate.insert(V(k), vel_0)
    initial_estimate.insert(B(k), bias_0)


optimizer_imu = gtsam.LevenbergMarquardtOptimizer(graph_imu, initial_estimate)
result_imu = optimizer_imu.optimize()
result_imu.print("")
