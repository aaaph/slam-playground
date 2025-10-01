import foxglove
from foxglove.channels import SceneUpdateChannel
from foxglove.schemas import (
    Color,
    FrameTransform,
    FrameTransforms,
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
from core.filter.updater import Updater
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
imu_gt_dataset = euroc_dataset.imu_and_ground_truth()
first_ground_truth = ground_truth[0]
first_imu = imu[0]


state = Initializer().initialize_from_dict(
    State(),
    timestamp=first_ground_truth["timestamp"],
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
    (first_ground_truth["gt_position"], first_ground_truth["gt_orientation"], first_ground_truth["gt_velocity"]),
)
imu_gt_iterator = imu_gt_dataset.to_iterable_dataset()
input()
make_update_each_x_imu_items = 20
i = 0
for imu_data in imu_gt_iterator:
    i += 1
    timestamp = imu_data["timestamp"]
    gyro = imu_data["gyro"]
    acc = imu_data["acc"]
    gt_position = imu_data["gt_position"]
    gt_orientation = imu_data["gt_orientation"]
    gt_velocity = imu_data["gt_velocity"]
    has_ground_truth = imu_data["has_ground_truth"]
    has_imu = imu_data["has_imu"]

    if has_ground_truth and i > make_update_each_x_imu_items:
        result, state = propagator.state_propagation(state, (timestamp, gyro, acc))
        state = updater.state_update(state, (gt_position, gt_orientation, gt_velocity))
        position = state.inertial_state.p
        orientation = state.inertial_state.q
        position_uncertainty = state.covariance.sigma[0:3, 0:3]
        i = 0
    else:
        result, state = propagator.state_propagation(state, (timestamp, gyro, acc))
        position = state.inertial_state.p
        orientation = state.inertial_state.q
        position_uncertainty = state.covariance.sigma[0:3, 0:3]

    if has_ground_truth:
        foxglove.log(
            "/tf",
            FrameTransforms(
                transforms=[
                    FrameTransform(
                        parent_frame_id="world",
                        child_frame_id="imu",
                        translation=Vector3(x=position[0], y=position[1], z=position[2]),
                        rotation=Quaternion(
                            x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3]
                        ),
                    ),
                    FrameTransform(
                        parent_frame_id="world",
                        child_frame_id="ground_truth",
                        translation=Vector3(x=gt_position[0], y=gt_position[1], z=gt_position[2]),
                        rotation=Quaternion(
                            x=gt_orientation[0],
                            y=gt_orientation[1],
                            z=gt_orientation[2],
                            w=gt_orientation[3],
                        ),
                    ),
                ]
            ),
        )
    else:
        foxglove.log(
            "/tf",
            FrameTransforms(
                transforms=[
                    FrameTransform(
                        parent_frame_id="world",
                        child_frame_id="imu",
                        translation=Vector3(x=position[0], y=position[1], z=position[2]),
                        rotation=Quaternion(
                            x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3]
                        ),
                    )
                ]
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
