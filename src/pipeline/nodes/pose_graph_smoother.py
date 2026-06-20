import time
from dataclasses import dataclass

from dora import Node

from core.front_end.keyframe import KF, keyframe_schema
from core.graph_optimizer.pose_graph_optimizator import LoopClosure, PoseGraphOptimizator
from core.transformations.special_euclidian_3_dim import SE3
from logger import spawn_logger
from pipeline.annotations import Ctx, Metadata
from pipeline.context import PipelineContext
from pipeline.decorators import handle, reactive, send_pipeline_context_output
from pipeline.nodes.base import PipelineNode


@dataclass(frozen=True, slots=True)
class PendingLoop:
    """Pending loop."""

    loop_closure: LoopClosure
    created_at_ns: int
    reference_timestamp: float | int


@reactive
class PoseGraphSmoother(PipelineNode):
    """Pose graph smoother."""

    def __init__(self) -> None:
        """Initialize the pose graph smoother."""
        self.node = Node()
        self.logger = spawn_logger(app="pose_graph_smoother")
        self.pgo = PoseGraphOptimizator()
        self.loop_wait_timeout_ns = 1_000_000_000  # 1 second
        self.pending_loops: dict[tuple[int, int], PendingLoop] = {}

    @handle("fixedlag_frame", "visualization")
    def handle_fixedlag_frame(self, ctx: Ctx) -> PipelineContext:
        """Handle the keyframe event."""
        kf = KF.list_from_arrow(ctx.get_record_batch("keyframes", keyframe_schema))[0]
        kf_id = kf.keyframe_id
        self.logger.debug(f"[PGO]: updating pose for kf_id: {kf_id}")
        se3 = SE3.from_matrix(ctx.get_ndarray("optimized_pose", (4, 4)))
        self.pgo.update_by_pose(kf_id, se3)
        return self._visualization_ctx(ctx.get_scalar("timestamp"))

    @handle("detected_loop", "visualization")
    def handle_detected_loop(self, ctx: Ctx) -> PipelineContext:
        """Handle the detected loop event."""
        now_ns = time.time_ns()
        timestamp = ctx.get_scalar("timestamp")

        from_key = int(ctx.get_scalar("from_key"))
        to_key = int(ctx.get_scalar("to_key"))
        transform = SE3.from_matrix(ctx.get_ndarray("transform", (4, 4)))
        cam0_in_body = SE3.from_matrix(ctx.get_ndarray("cam0_in_body", (4, 4)))
        detected_loop = LoopClosure(from_key, to_key, transform, cam0_in_body)

        pending_loop = PendingLoop(detected_loop, now_ns, timestamp)
        self.pending_loops[(from_key, to_key)] = pending_loop
        self.logger.debug(f"[PGO]: pending loop: {pending_loop}")
        return self._visualization_ctx(timestamp)

    @handle("reconcile_tick", "visualization")
    def reconcile(self, metadata: Metadata) -> PipelineContext | None:
        """Reconcile the pending loops."""
        timestamp = 0
        ready_loops = {}
        for loop_key, pending_loop in list(self.pending_loops.items()):
            loop = pending_loop.loop_closure
            from_exists = self.pgo.has_pose(loop.from_key)
            to_exists = self.pgo.has_pose(loop.to_key)
            if not from_exists or not to_exists:
                self.logger.trace(
                    f"[PGO]: loop {loop_key} not ready, from_exists: {from_exists}, to_exists: {to_exists}"
                )
                continue
            ready_loops[loop_key] = pending_loop

        if not ready_loops:
            return None

        loop_count = len(ready_loops)
        self.logger.debug(f"[PGO]: reconciling {loop_count} loops")

        for loop_key, pending_loop in ready_loops.items():
            timestamp = max(timestamp, pending_loop.reference_timestamp)
            self.pgo.update_by_loop_closure(pending_loop.loop_closure)
            self.pending_loops.pop(loop_key)

        self.pgo.optimize()
        self.logger.debug(f"[PGO]: reconciled {loop_count} loops, pgo_T_odom: {self.pgo.diff}")

        new_ctx = PipelineContext.from_timestamp(timestamp).set_ndarray("pgo_T_odom", self.pgo.diff.as_matrix())
        send_pipeline_context_output(self.node, "pgo_frame", new_ctx, metadata)
        return new_ctx

    def _visualization_ctx(self, timestamp: float) -> PipelineContext:
        """Create a PGO visualization context."""
        pgo_ctx = PipelineContext.from_timestamp(timestamp)
        pgo_ctx.set_record_batch("pgo_graph", self.pgo.to_trajectory().to_arrow())
        return pgo_ctx


if __name__ == "__main__":
    node = PoseGraphSmoother()
    node.run()
