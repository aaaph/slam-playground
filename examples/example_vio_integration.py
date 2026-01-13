import numpy as np

from dataset.euroc import EurocDataset
from logger import spawn_logger
from visualizer.foxglove.factories.foxglove_imu_stereo_factory import FoxgloveImuStereoFactory, ImuAndImageContext

euroc = EurocDataset.mh_01_easy()
euroc_iterator = euroc.imu_and_stereo().to_iterable_dataset()
logger = spawn_logger(app="example_vio_integration")

viz = FoxgloveImuStereoFactory().create_imu_stereo_viz(wait_for_client=True, viz_type="websocket")
for sample in euroc_iterator:
    ts = float(sample["timestamp"])
    gyro = np.array(sample["gyro"])
    acc = np.array(sample["acc"])
    stereo = np.array(sample["stereo"])

    has_imu = gyro is not None and acc is not None
    has_stereo = stereo[0] is not None
    logger.info(f"Timestamp: {ts:.0f}, Has IMU: {has_imu}, Has Stereo: {has_stereo}")
    ground_truth = euroc.find_nearest_ground_truth_by_timestamp_se3(ts)
    viz_context = ImuAndImageContext(
        timestamp=ts,
        gyro=gyro,
        acc=acc,
        frame=stereo[0],
        ground_truth_se3=ground_truth,
    )
    viz.send(viz_context)
