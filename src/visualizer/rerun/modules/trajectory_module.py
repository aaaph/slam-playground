from typing import Any

import numpy as np
import rerun as rr
from pydantic import BaseModel

from core.graph_optimizer.pose_graph_optimizator import EdgeType, PoseGraphSnapshot, trajectory_arrow_schema
from pipeline.annotations import Ctx
from visualizer.rerun.modules.abc_module import IVizModule


class TrajectoryModuleOptions(BaseModel):
    """Trajectory module options."""

    throw_on_nothing: bool = False


class TrajectoryModule(IVizModule):
    """Trajectory module."""

    def __init__(self, property_name: str, entity_path: str, raw_options: dict[str, Any]) -> None:
        """Initialize the trajectory module."""
        self.options = TrajectoryModuleOptions(**raw_options)
        self.property_name = property_name
        self.entity_path = entity_path
        self.throw_on_nothing = self.options.throw_on_nothing

    def setup(self) -> None:
        """Set up the trajectory module."""

    def process(self, context: Ctx) -> None:
        """Process the trajectory data."""
        exists = context.exists(self.property_name)
        if not exists and self.throw_on_nothing:
            msg = f"Features data not found in context: {self.property_name}"
            raise KeyError(msg)
        if not exists and not self.throw_on_nothing:
            return
        timestamp = context.get_scalar("timestamp", float)
        record_batch = context.get_record_batch(self.property_name, schema=trajectory_arrow_schema)
        trajectory = PoseGraphSnapshot.from_arrow(record_batch)

        positions = trajectory.poses[:, 5:8].astype(np.float32)

        kf_ids = trajectory.poses[:, 0].astype(np.int32)
        loop_closures = trajectory.edges[trajectory.edges[:, 2] == EdgeType.LOOP_CLOSURE.value].astype(
            np.int32, copy=False
        )
        if loop_closures.shape[0] > 0:
            id_to_pose = {int(row[0]): row[5:8].astype(np.float32) for row in trajectory.poses}
            loop_strips = []
            for from_id, to_id, _ in loop_closures:
                from_pose = id_to_pose.get(int(from_id))
                to_pose = id_to_pose.get(int(to_id))
                if from_pose is None or to_pose is None:
                    continue
                loop_strips.append(np.stack([from_pose, to_pose], axis=0))

            if loop_strips:
                loop_closures_stack = np.stack(loop_strips, axis=0)
                loop_closures_3d = rr.LineStrips3D(loop_closures_stack, colors=[255, 80, 80], radii=0.0075)
                loop_closures_entity = f"{self.entity_path}/loop_closures"
                rr.log(loop_closures_entity, loop_closures_3d)

        rr.set_time("frame_time", timestamp=timestamp / 1e9)
        line_strips_3d = rr.LineStrips3D(positions, colors=[128, 128, 128], radii=0.0075)
        line_strips_entity = f"{self.entity_path}/line"
        rr.log(line_strips_entity, line_strips_3d)
        points_entity = f"{self.entity_path}/keyframes"
        points_3d = rr.Points3D(
            positions,
            colors=[128, 128, 128],
            radii=0.03,
            labels=[str(kf_id) for kf_id in kf_ids],
            show_labels=False,
        )
        rr.log(points_entity, points_3d)

    def __repr__(self) -> str:
        """Return the string representation of the trajectory module."""
        return f"TrajectoryModule(property_name={self.property_name}, entity_path={self.entity_path})"
