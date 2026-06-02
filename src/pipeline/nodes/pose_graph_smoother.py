import time
from dataclasses import dataclass
from typing import Any

from dora import Node

from core.front_end.keyframe import KF, keyframe_schema
from core.graph_optimizer.pose_graph_optimizator import LoopClosure, PoseGraphOptimizator
from core.transformations.special_euclidian_3_dim import SE3
from logger import spawn_logger
from pipeline.annotations import Ctx, Metadata
from pipeline.context import PipelineContext
from pipeline.decorators import on_input, reactive, send_pipeline_context_output


@dataclass(frozen=True, slots=True)
class PendingLoop:
    """Pending loop."""

    loop_closure: LoopClosure
    created_at_ns: int
    reference_timestamp: float | int


@reactive
class PoseGraphSmoother:
    """Pose graph smoother."""

    def __init__(self) -> None:
        """Initialize the pose graph smoother."""
        self.node = Node()
        self.logger = spawn_logger(app="pose_graph_smoother")
        self.pgo = PoseGraphOptimizator()
        self.loop_wait_timeout_ns = 1_000_000_000  # 1 second
        self.pending_loops: dict[tuple[int, int], Any] = {}

    def run(self) -> None: ...  # noqa: D102

    @on_input("fixedlag_frame")
    def handle_fixedlag_frame(self, ctx: Ctx, metadata: Metadata) -> None:
        """Handle the keyframe event."""
        kf = KF.list_from_arrow(ctx.get_record_batch("keyframes", keyframe_schema))[0]
        kf_id = kf.keyframe_id
        self.logger.debug(f"[PGO]: updating pose for kf_id: {kf_id}")
        se3 = SE3.from_matrix(ctx.get_ndarray("optimized_pose", (4, 4)))
        self.pgo.update_by_pose(kf_id, se3)
        pgo_ctx = PipelineContext.from_timestamp(ctx.get_scalar("timestamp"))
        pgo_ctx.set_record_batch("pgo_graph", self.pgo.to_trajectory().to_arrow())
        send_pipeline_context_output(self.node, "visualization", pgo_ctx, metadata)

    @on_input("detected_loop")
    def handle_detected_loop(self, ctx: Ctx) -> None:
        """Handle the detected loop event."""
        now_ns = time.time_ns()
        timestamp = ctx.get_scalar("timestamp")

        from_key = ctx.get_scalar("from_key")
        to_key = ctx.get_scalar("to_key")
        transform = SE3.from_matrix(ctx.get_ndarray("transform", (4, 4)))
        cam0_in_body = SE3.from_matrix(ctx.get_ndarray("cam0_in_body", (4, 4)))
        detected_loop = LoopClosure(int(from_key), int(to_key), transform, cam0_in_body)

        pending_loop = PendingLoop(detected_loop, now_ns, timestamp)
        self.pending_loops[(from_key, to_key)] = pending_loop
        self.logger.debug(f"[PGO]: pending loop: {pending_loop}")

    @on_input(input_id="reconcile_tick")
    def reconcile(self, metadata: Metadata) -> None:
        """Reconcile the pending loops."""
        if not self.pending_loops:
            return
        loop_count = len(self.pending_loops)
        self.logger.debug(f"[PGO]: reconciling {loop_count} loops")
        timestamp = 0
        for pending_loop in self.pending_loops.values():
            timestamp = max(timestamp, pending_loop.reference_timestamp)
            self.pgo.update_by_loop_closure(pending_loop.loop_closure)
        self.pgo.optimize()
        self.pending_loops.clear()
        self.logger.debug(f"[PGO]: reconciled {loop_count} loops")
        pgo_ctx = PipelineContext.from_timestamp(timestamp)
        pgo_ctx.set_record_batch("pgo_graph", self.pgo.to_trajectory().to_arrow())
        send_pipeline_context_output(self.node, "visualization", pgo_ctx, metadata)


if __name__ == "__main__":
    node = PoseGraphSmoother()
    node.run()
