import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from core.graph_optimizer.smart_vio_optimizer import SmartVIOOptimizer
from rerun.blueprint.archetypes import ForceLink, ForceManyBody

from core.camera_model.stereo_camera_model import StereoCameraModel
from core.front_end.keyframe_selector import SelectReason
from core.graph_optimizer.optimizer_types import OptKeyframe
from core.transformations.special_euclidian_3_dim import SE3
from dataset.sensor_config import CameraSensor

blueprint = rrb.Blueprint(
    rrb.GraphView(
        origin="test_node_graph",
        name="test_node_graph",
        force_link=ForceLink(distance=60),
        force_many_body=ForceManyBody(strength=-60),
        defaults=[rr.GraphNodes.from_fields(show_labels=False)],
    )
)

cam0_conf = CameraSensor(
    {
        "resolution": (752, 480),
        "camera_model": "pinhole",
        "intrinsics": (458.654, 457.296, 367.215, 248.375),
        "distortion_model": "radial-tangential",
        "distortion_coefficients": (-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05),
        "T_BS": {
            "cols": 4,
            "rows": 4,
            "data": [
                0.0148655429818,
                -0.999880929698,
                0.00414029679422,
                -0.0216401454975,
                0.999557249008,
                0.0149672133247,
                0.025715529948,
                -0.064676986768,
                -0.0257744366974,
                0.00375618835797,
                0.999660727178,
                0.00981073058949,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        },
    }
)
cam1_conf = CameraSensor(
    {
        "resolution": (752, 480),
        "camera_model": "pinhole",
        "intrinsics": (457.587, 456.134, 379.999, 255.238),
        "distortion_model": "radial-tangential",
        "distortion_coefficients": (-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05),
        "T_BS": {
            "cols": 4,
            "rows": 4,
            "data": [
                0.0125552670891,
                -0.999755099723,
                0.0182237714554,
                -0.0198435579556,
                0.999598781151,
                0.0130119051815,
                0.0251588363115,
                0.0453689425024,
                -0.0253898008918,
                0.0179005838253,
                0.999517347078,
                0.00786212447038,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        },
    }
)
active_track = np.array(
    [
        [0.0, 104.97448, 151.0232, 88.630646, 151.0983],
        [1.0, 108.95517, 150.03362, 92.62471, 150.12679],
        [2.0, 106.99274, 155.06313, 90.69218, 155.12927],
        [3.0, 76.072105, 85.05008, 58.585823, 84.86435],
        [4.0, 58.96277, 32.956642, 40.99414, 32.90497],
        [5.0, 74.996635, 81.97859, 57.4856, 81.8415],
        [6.0, 206.00143, 80.99979, 187.718, 80.94506],
        [7.0, 260.9987, 80.01949, 243.74312, 80.19046],
        [8.0, 211.05966, 82.981834, 192.7672, 82.93396],
        [9.0, 262.75363, 85.003654, 245.38556, 85.08636],
        [10.0, 198.01851, 151.00595, 179.75465, 150.96199],
        [11.0, 263.03024, 78.00598, 245.79309, 78.10736],
        [12.0, 547.98016, 91.98258, 525.2076, 91.83333],
        [13.0, 552.9773, 98.97931, 530.1903, 98.85397],
        [14.0, 552.0212, 87.002235, 529.2662, 86.8863],
        [15.0, 557.93616, 91.9521, 535.2311, 91.8494],
        [16.0, 554.9467, 96.9978, 532.13385, 96.83855],
        [17.0, 551.9825, 89.990814, 529.21747, 89.84863],
        [18.0, 578.0895, 120.02344, 556.60626, 119.95298],
        [19.0, 566.97546, 60.939857, 544.1087, 60.06797],
        [20.0, 568.01044, 50.056393, 544.7217, 49.503544],
        [21.0, 732.03326, 128.02762, 709.3501, 128.22522],
        [22.0, 571.9164, 65.98471, 549.2303, 65.62563],
        [23.0, 570.9903, 37.074802, 547.4337, 36.515755],
        [25.0, 169.04674, 315.99527, 141.58984, 315.76703],
        [26.0, 162.06482, 316.98483, 134.78879, 316.84723],
        [27.0, 167.0183, 316.00803, 139.84515, 315.84354],
        [28.0, 186.05045, 226.0, 167.84789, 226.07596],
        [29.0, 151.9927, 314.9606, 124.34268, 314.83374],
        [30.0, 372.02866, 286.99374, 348.68384, 286.90817],
        [31.0, 362.9897, 277.0003, 340.27542, 276.8389],
        [32.0, 375.01273, 285.99753, 351.62418, 285.93628],
        [33.0, 364.99768, 280.9951, 342.03903, 280.85208],
        [34.0, 372.99448, 299.99683, 347.6476, 299.77325],
        [35.0, 367.0014, 280.99216, 344.0317, 280.84158],
        [36.0, 394.99347, 306.99866, 369.29507, 306.80267],
        [37.0, 390.96948, 309.01166, 365.12866, 308.85425],
        [38.0, 394.99933, 317.00464, 368.52274, 317.0149],
        [39.0, 385.98615, 316.99915, 359.45078, 316.99332],
        [40.0, 384.0066, 308.99805, 357.94482, 308.92392],
        [41.0, 376.0163, 306.99573, 350.161, 306.8967],
        [42.0, 613.9922, 300.99628, 589.5587, 301.02405],
        [43.0, 643.00793, 309.9992, 617.494, 309.82443],
        [44.0, 635.0067, 312.004, 609.6115, 311.92606],
        [45.0, 656.9806, 314.99188, 630.729, 314.80225],
        [46.0, 645.01495, 316.00336, 619.26373, 315.92822],
        [47.0, 626.99426, 302.0085, 602.39264, 301.96332],
        [48.0, 153.99464, 338.00928, 124.325, 337.938],
        [50.0, 127.99922, 340.99683, 97.67162, 340.9349],
        [51.0, 144.01027, 343.99893, 114.17843, 343.78046],
        [52.0, 116.99682, 357.99405, 85.338936, 357.86063],
        [54.0, 188.97502, 324.00272, 161.23804, 323.77026],
        [55.0, 195.0404, 334.02246, 167.11212, 333.85315],
        [56.0, 187.98941, 328.00012, 160.0982, 327.81964],
        [57.0, 249.99197, 348.04147, 219.2879, 347.88068],
        [58.0, 251.01447, 354.0571, 220.19463, 353.95938],
        [59.0, 254.98909, 350.98593, 224.34807, 350.86612],
        [60.0, 422.99344, 341.9983, 393.79425, 341.9052],
        [61.0, 403.99854, 336.0054, 375.21002, 335.96042],
        [66.0, 701.00055, 334.01236, 672.9242, 333.88083],
        [68.0, 708.033, 343.0154, 679.20166, 342.93768],
        [69.0, 678.98865, 329.9936, 651.6497, 329.89932],
        [71.0, 724.9924, 343.9903, 695.6107, 343.74402],
        [72.0, 459.0, 384.0, np.nan, np.nan],
        [73.0, 451.0, 379.0, np.nan, np.nan],
        [74.0, 470.0, 403.0, np.nan, np.nan],
        [75.0, 512.0, 438.0, np.nan, np.nan],
    ]
)

camera_model = StereoCameraModel.from_cameras_config(cam0_conf, cam1_conf)
camera_ctx = camera_model.as_stereo_ctx()

optimizer = SmartVIOOptimizer.from_stereo_ctx(camera_ctx)

keyframe_one = OptKeyframe(
    keyframe_id=0,
    select_reason=SelectReason.TIME_ELAPSED,
    active_track=active_track,
    timestamp=10.0,
    pose=SE3.identity(),
)

keyframe_two = OptKeyframe(
    keyframe_id=1,
    select_reason=SelectReason.TIME_ELAPSED,
    active_track=active_track,
    timestamp=14.0,
    pose=SE3.identity(),
)
keyframe_three = OptKeyframe(
    keyframe_id=2,
    select_reason=SelectReason.TIME_ELAPSED,
    active_track=active_track,
    timestamp=18.0,
    pose=SE3.identity(),
)
optimizer.add_new_keyframe(keyframe_one)
graph_arrow = optimizer.get_graph_arrow()

rr.init("test_get_graph_arrow", default_blueprint=blueprint, spawn=True)

nodes = graph_arrow["nodes"]["ids"].to_pylist()
labels = graph_arrow["nodes"]["labes"].to_pylist()

colors = []
for node_type in graph_arrow["nodes"]["types"].to_pylist():
    if node_type == "pose":
        colors.append((0, 255, 255))  # purple
    elif node_type == "landmark":
        colors.append((125, 125, 255))  # light blue
    elif node_type == "factor":
        colors.append((255, 255, 255))  # white
    else:
        colors.append((125, 125, 125))  # gray
radii = []
for node_type in graph_arrow["nodes"]["types"].to_pylist():
    if node_type == "pose":
        radii.append(50)
    elif node_type == "landmark":
        radii.append(150)
    elif node_type == "factor":
        radii.append(10)
    else:
        radii.append(10)

rr.log(
    "test_node_graph",
    rr.GraphNodes(node_ids=nodes, labels=labels, radii=np.array(radii), colors=colors),
    rr.GraphEdges(edges=graph_arrow["edges"]["tuples"].to_pylist()),
)

optimizer.add_new_keyframe(keyframe_two)
graph_arrow = optimizer.get_graph_arrow()
nodes = graph_arrow["nodes"]["ids"].to_pylist()
labels = graph_arrow["nodes"]["labes"].to_pylist()

colors = []
for node_type in graph_arrow["nodes"]["types"].to_pylist():
    if node_type == "pose":
        colors.append((0, 255, 255))  # purple
    elif node_type == "landmark":
        colors.append((125, 125, 255))  # light blue
    elif node_type == "factor":
        colors.append((255, 255, 255))  # white
    else:
        colors.append((125, 125, 125))  # gray
radii = []
for node_type in graph_arrow["nodes"]["types"].to_pylist():
    if node_type == "pose":
        radii.append(50)
    elif node_type == "landmark":
        radii.append(150)
    elif node_type == "factor":
        radii.append(10)
    else:
        radii.append(10)

rr.log(
    "test_node_graph",
    rr.GraphNodes(node_ids=nodes, labels=labels, radii=np.array(radii), colors=colors),
    rr.GraphEdges(edges=graph_arrow["edges"]["tuples"].to_pylist()),
)

optimizer.add_new_keyframe(keyframe_three)
graph_arrow = optimizer.get_graph_arrow()
nodes = graph_arrow["nodes"]["ids"].to_pylist()
labels = graph_arrow["nodes"]["labes"].to_pylist()

colors = []
for node_type in graph_arrow["nodes"]["types"].to_pylist():
    if node_type == "pose":
        colors.append((0, 255, 255))  # purple
    elif node_type == "landmark":
        colors.append((125, 125, 255))  # light blue
    elif node_type == "factor":
        colors.append((255, 255, 255))  # white
    else:
        colors.append((125, 125, 125))  # gray
radii = []
for node_type in graph_arrow["nodes"]["types"].to_pylist():
    if node_type == "pose":
        radii.append(50)
    elif node_type == "landmark":
        radii.append(150)
    elif node_type == "factor":
        radii.append(10)
    else:
        radii.append(10)

rr.log(
    "test_node_graph",
    rr.GraphNodes(node_ids=nodes, labels=labels, radii=np.array(radii), colors=colors),
    rr.GraphEdges(edges=graph_arrow["edges"]["tuples"].to_pylist()),
)
