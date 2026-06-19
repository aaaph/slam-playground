from pathlib import Path

from dora import Node

from core.transformations.special_euclidian_3_dim import SE3
from dataset.factory import DatasetFactory
from dataset.ground_truth import GROUND_TRUTH_INDEX_CACHE_FILENAME, GroundTruthIndex
from logger import spawn_logger
from pipeline.annotations import Ctx
from pipeline.context import PipelineContext
from pipeline.decorators import on_input, reactive, to_output
from pipeline.nodes.base import PipelineNode
from pipeline.runtime_config import DatasetNodeConfig


@reactive
class TrajectoryEvaluator(PipelineNode):
    """Trajectory evaluator."""

    def __init__(self, ground_truth_index: GroundTruthIndex) -> None:
        """Initialize the trajectory evaluator."""
        self.ground_truth_index = ground_truth_index
        self.node = Node()
        self.logger = spawn_logger(app="trajectory_evaluator")
        self.init = False
        self.offset = SE3.identity()

    @on_input("frontend_frame")
    @to_output("visualization")
    def handle_frontend_frame(self, ctx: Ctx) -> PipelineContext | None:
        """Handle the frontend frame."""
        timestamp = ctx.get_scalar("timestamp")
        front_end_mode = int(ctx.get_scalar("front_end_mode"))
        pose_estimate = ctx.get_ndarray("pose_estimate", (4, 4))
        pose_estimate_se3 = SE3.from_matrix(pose_estimate)
        if front_end_mode > 1 and not self.init:
            self.offset = pose_estimate_se3 * self.ground_truth_index.nearest(timestamp).se3().inverse()
            self.logger.info(f"Front-End Initialization Done: Alignment offset: {self.offset}")
            self.init = True
        if not self.init:
            self.logger.trace("Frontend still not initialized")
            return None

        ground_truth_aligned_se3 = self.offset * self.ground_truth_index.nearest(timestamp).se3()
        self.logger.info(f"Ground truth aligned SE3: {ground_truth_aligned_se3}")
        new_ctx = PipelineContext.from_timestamp(timestamp)
        new_ctx.set_ndarray("ground_truth_aligned_se3", ground_truth_aligned_se3.as_matrix())
        return new_ctx


def resolve_ground_truth_index_cache_path(runtime_config: DatasetNodeConfig, factory: DatasetFactory) -> Path:
    """Resolve the ground truth index cache path for the active dataset."""
    repo_root = (runtime_config.repo_root or Path.cwd()).resolve()
    cache_root = runtime_config.dataset_cache_path
    if cache_root is None:
        resolved = factory.registry.resolve(runtime_config.dataset_name)
        cache_root = resolved.dataset.cache or resolved.dataset.root / "cache"
    if not cache_root.is_absolute():
        cache_root = repo_root / cache_root
    return cache_root / GROUND_TRUTH_INDEX_CACHE_FILENAME


if __name__ == "__main__":
    runtime_config = TrajectoryEvaluator.runtime_config_as(DatasetNodeConfig)
    factory = DatasetFactory(repo_root=runtime_config.repo_root)
    cache_path = resolve_ground_truth_index_cache_path(runtime_config, factory)
    logger = spawn_logger(app="trajectory_evaluator")
    if cache_path.exists():
        logger.info(f"Loading ground truth index cache: {cache_path}")
    else:
        logger.info(f"Ground truth index cache not found; building cache: {cache_path}")
    ground_truth_index = GroundTruthIndex.load_or_build(
        cache_path,
        lambda: factory.load_ground_truth_dataset(runtime_config.dataset_name).ground_truth(),
    )
    trajectory_evaluator = TrajectoryEvaluator(ground_truth_index)
    trajectory_evaluator.run()
