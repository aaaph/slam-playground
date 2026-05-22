from dataclasses import dataclass
from enum import IntEnum, auto

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

import gtsam
from core.transformations.special_euclidian_3_dim import SE3
from logger import spawn_logger

X = gtsam.symbol_shorthand.X


trajectory_arrow_schema = pa.schema(
    [
        pa.field("iteration", pa.int64()),
        pa.field("poses", pa.list_(pa.list_(pa.float32(), list_size=8))),
        pa.field("edges", pa.list_(pa.list_(pa.int32(), list_size=3))),
    ]
)


class EdgeType(IntEnum):
    """Edge type."""

    ODOMETRY = auto()
    LOOP_CLOSURE = auto()


@dataclass(frozen=True, slots=True)
class Edge:
    """Essential graph edge DTO."""

    from_key: int
    to_key: int
    type: EdgeType


@dataclass(frozen=True, slots=True)
class PoseGraphSnapshot:
    """Pose graph snapshot."""

    iteration: int
    poses: NDArray[np.float32]
    edges: NDArray[np.int32]

    def to_arrow(self) -> pa.RecordBatch:
        """Convert the snapshot to an arrow."""
        return pa.RecordBatch.from_pydict(
            {
                "iteration": [self.iteration],
                "poses": [self.poses.astype(np.float32, copy=False).tolist()],
                "edges": [self.edges.astype(np.int32, copy=False).tolist()],
            },
            schema=trajectory_arrow_schema,
        )

    @classmethod
    def from_arrow(cls, arrow: pa.RecordBatch) -> "PoseGraphSnapshot":
        """Convert an arrow to a snapshot."""
        poses = np.asarray(arrow.column("poses")[0].as_py(), dtype=np.float32).reshape(-1, 8)
        edges = np.asarray(arrow.column("edges")[0].as_py(), dtype=np.int32).reshape(-1, 3)
        return cls(
            iteration=int(arrow.column("iteration")[0].as_py()),
            poses=poses,
            edges=edges,
        )


@dataclass(frozen=True, slots=True)
class LoopClosure:
    """
    Loop closure edge.

    The transform must follow GTSAM BetweenFactorPose3 convention:
    transform ~= X(from_key).between(X(to_key)).
    """

    from_key: int
    to_key: int
    transform: SE3
    cam0_in_body: SE3

    @property
    def body_reference_t_body_query(self) -> SE3:
        """Get the body reference transform in body query frame."""
        return self.cam0_in_body * self.transform * self.cam0_in_body.inverse()


class PoseGraphOptimizator:
    """Pose graph optimizator(PGO)."""

    def __init__(self) -> None:
        """Initialize the pose graph optimizator."""
        self.logger = spawn_logger(app="PGO")
        self.factors = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.params = gtsam.LevenbergMarquardtParams()
        self.params.setlambdaInitial(0.0)
        self.params.setlambdaLowerBound(0.0)
        self.params.setlambdaUpperBound(0.0)
        self.last_kf_id = -1
        self.pose_dict: dict[int, gtsam.Pose3] = {}
        self.raw_pose_dict: dict[int, gtsam.Pose3] = {}
        self.edges: list[Edge] = []
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4])
        self.between_noise = gtsam.noiseModel.Diagonal.Sigmas([0.05, 0.05, 0.05, 0.15, 0.15, 0.15])
        loop_base_noies = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1, 0.3, 0.3, 0.3])
        self.loop_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.345), loop_base_noies
        )
        self.iteration = 0
        self.diff = SE3.identity()

    def optimize(self) -> gtsam.Values:
        """Optimize the graph."""
        self.logger.info(f"Optimizing graph, f:{self.factors.size()}, v:{self.values.size()}")

        optimizer = gtsam.LevenbergMarquardtOptimizer(self.factors, self.values, self.params)
        self.values = optimizer.optimize()
        self.pose_dict[self.last_kf_id] = self.values.atPose3(X(self.last_kf_id))
        last_opt_pose = SE3.from_gtsam_pose(self.pose_dict[self.last_kf_id])
        last_raw_pose = SE3.from_gtsam_pose(self.raw_pose_dict[self.last_kf_id])
        self.diff = last_opt_pose * last_raw_pose.inverse()
        self.iteration += 1
        return self.values

    def update_by_pose(self, kf_id: int, se3: SE3) -> None:
        """Update the graph with a pose."""
        raw_se3 = se3
        corrected_se3 = self.diff * se3

        gtsam_raw_pose = raw_se3.as_gtsam_pose()
        gtsam_corrected_pose = corrected_se3.as_gtsam_pose()

        self.values.insert(X(kf_id), gtsam_corrected_pose)
        self.pose_dict[kf_id] = gtsam_corrected_pose
        self.raw_pose_dict[kf_id] = gtsam_raw_pose

        if self.last_kf_id == -1:
            factor = gtsam.PriorFactorPose3(X(kf_id), gtsam_corrected_pose, self.prior_noise)
            self.factors.add(factor)
        else:
            from_id = X(self.last_kf_id)
            to_id = X(kf_id)
            between = self.raw_pose_dict[self.last_kf_id].between(self.raw_pose_dict[kf_id])
            factor = gtsam.BetweenFactorPose3(from_id, to_id, between, self.between_noise)
            self.factors.add(factor)
            self.edges.append(Edge(self.last_kf_id, kf_id, EdgeType.ODOMETRY))
        self.last_kf_id = kf_id

    def update_by_loop_closure(self, loop_closure: LoopClosure) -> None:
        """Update the graph with a loop closure."""
        from_key = loop_closure.from_key
        to_key = loop_closure.to_key
        relative_pose = loop_closure.body_reference_t_body_query.as_gtsam_pose()
        factor = gtsam.BetweenFactorPose3(X(from_key), X(to_key), relative_pose, self.loop_noise)
        self.factors.add(factor)
        self.edges.append(Edge(from_key, to_key, EdgeType.LOOP_CLOSURE))

    def poses_ndarray(self) -> NDArray[np.float32]:  # shape: (n, 8) - [kf_id, quat_xyzw, vec]
        """Get the poses as a numpy array. schema: [kf_id, qx, qy, qz, qw, px, py, pz]."""
        poses_array = np.zeros((self.values.size(), 8), dtype=np.float32)
        for i, kf_id in enumerate(self.pose_dict):
            pose = self.values.atPose3(X(kf_id))
            quat = Rotation.from_matrix(pose.rotation().matrix()).as_quat()
            vec = pose.translation()
            poses_array[i, :] = np.concatenate([[kf_id], quat, vec])
        return poses_array

    def edges_ndarray(self) -> NDArray[np.int32]:  # shape: (n, 3) - [kf_id_from, kf_id_to, edge_type]
        """Get the edges as a numpy array. schema: [kf_id_from, kf_id_to]."""
        edges_array = np.zeros((len(self.edges), 3), dtype=np.int32)
        for i, edge in enumerate(self.edges):
            edges_array[i, 0] = edge.from_key
            edges_array[i, 1] = edge.to_key
            edges_array[i, 2] = edge.type.value
        return edges_array

    def to_trajectory(self) -> PoseGraphSnapshot:
        """Convert the graph to a trajectory."""
        return PoseGraphSnapshot(
            iteration=self.iteration,
            poses=self.poses_ndarray(),
            edges=self.edges_ndarray(),
        )
