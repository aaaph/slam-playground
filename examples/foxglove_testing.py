import foxglove
from foxglove.channels import SceneUpdateChannel
from foxglove.schemas import (
    Color,
    FrameTransform,
    Pose,
    Quaternion,
    SceneEntity,
    SceneUpdate,
    SpherePrimitive,
    Vector3,
)
from foxglove.websocket import Capability

from core.filter.initializer import Initializer
from core.filter.propagator import Propagator
from core.filter.state import State
from dataset.euroc import EurocDataset
from visualizer.foxglove_listener import FoxgloveServerListener

# Our example logs data on a couple of different topics, so we'll create a
# channel for each. We can use a channel like SceneUpdateChannel to log
# Foxglove schemas, or a generic Channel to log custom data.
scene_channel = SceneUpdateChannel("/scene")

server = foxglove.start_server(
    server_listener=FoxgloveServerListener(),
    capabilities=[Capability.ClientPublish],
    supported_encodings=["json"],
)


euroc_dataset = EurocDataset.mh_01_easy()

ground_truth = euroc_dataset.ground_truth()
imu = euroc_dataset.imu()
first_ground_truth = ground_truth[0]
first_imu = imu[0]

initializer = Initializer()

state = initializer.initialize_from_row(
    State(),
    (
        (first_ground_truth["timestamp"]),
        first_ground_truth["gt_position"],
        first_ground_truth["gt_orientation"],
        first_ground_truth["gt_velocity"],
        first_ground_truth["gt_gyro_bias"],
        first_ground_truth["gt_acc_bias"],
    ),
)

propagator = Propagator(
    ng=euroc_dataset.config.imu0.payload["gyroscope_noise_density"],
    na=euroc_dataset.config.imu0.payload["accelerometer_noise_density"],
    nba=euroc_dataset.config.imu0.payload["accelerometer_random_walk"],
    nbg=euroc_dataset.config.imu0.payload["gyroscope_random_walk"],
)

imu_iterator = imu.to_iterable_dataset()

for imu_data in imu_iterator:
    timestamp = imu_data["timestamp"]
    gyro = imu_data["gyro"]
    acc = imu_data["acc"]

    result, state = propagator.state_propagation(state, (timestamp, gyro, acc))

    position = state.inertial_state.p
    orientation = state.inertial_state.q
    position_uncertainty = state.covariance.sigma[0:3, 0:3]

    foxglove.log(
        "/tf",
        FrameTransform(
            parent_frame_id="world",
            child_frame_id="imu",
            translation=Vector3(x=position[0], y=position[1], z=position[2]),
            rotation=Quaternion(x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3]),
        ),
    )
    scene_channel.log(
        SceneUpdate(
            entities=[
                SceneEntity(
                    frame_id="imu",
                    spheres=[
                        SpherePrimitive(
                            pose=Pose(
                                position=Vector3(x=0, y=0, z=0),
                                orientation=Quaternion(x=0, y=0, z=0, w=1),
                            ),
                            size=Vector3(
                                x=position_uncertainty[0, 0],
                                y=position_uncertainty[1, 1],
                                z=position_uncertainty[2, 2],
                            ),
                            color=Color(r=0, g=1.0, b=0, a=0.2),
                        )
                    ],
                ),
            ]
        )
    )
    input()
