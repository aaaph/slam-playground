from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
from dora import Node

from core.transformations.helpers import calculate_ape
from core.transformations.special_euclidian_3_dim import SE3
from dataset.factory import DatasetFactory
from dataset.ground_truth import GROUND_TRUTH_INDEX_CACHE_FILENAME, GroundTruthIndex
from logger import spawn_logger
from pipeline.annotations import Ctx
from pipeline.context import PipelineContext
from pipeline.decorators import on_input, reactive, to_output
from pipeline.nodes.base import PipelineNode
from pipeline.runtime_config import DatasetNodeConfig

trajectory_metrics_schema = pa.schema(
    [
        pa.field("ape_translation_m", pa.float64()),
        pa.field("ape_rotation_deg", pa.float64()),
        pa.field("rmse_translation_m", pa.float64()),
        pa.field("rmse_rotation_deg", pa.float64()),
        pa.field("gt_timestamp_diff_ms", pa.float64()),
        pa.field("gt_timestamp_abs_diff_ms", pa.float64()),
        pa.field("samples_count", pa.int64()),
    ]
)


@dataclass(frozen=True)
class TrajectoryMetrics:
    """Trajectory evaluation metrics for one frame."""

    ape_translation_m: float
    ape_rotation_deg: float
    rmse_translation_m: float
    rmse_rotation_deg: float
    gt_timestamp_diff_ms: float
    gt_timestamp_abs_diff_ms: float
    samples_count: int

    def as_arrow(self) -> pa.RecordBatch:
        """Convert trajectory metrics to an Arrow record batch."""
        return pa.RecordBatch.from_pydict(
            {
                "ape_translation_m": [self.ape_translation_m],
                "ape_rotation_deg": [self.ape_rotation_deg],
                "rmse_translation_m": [self.rmse_translation_m],
                "rmse_rotation_deg": [self.rmse_rotation_deg],
                "gt_timestamp_diff_ms": [self.gt_timestamp_diff_ms],
                "gt_timestamp_abs_diff_ms": [self.gt_timestamp_abs_diff_ms],
                "samples_count": [self.samples_count],
            },
            schema=trajectory_metrics_schema,
        )


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
        self.ape_translation_sq_sum = 0.0
        self.ape_rotation_sq_sum = 0.0
        self.ape_samples_count = 0

    @on_input("slam_frame")
    @to_output("visualization")
    def handle_slam_frame(self, ctx: Ctx) -> PipelineContext | None:
        """Handle the slam frame."""
        timestamp = ctx.get_scalar("timestamp")
        timestamp_ns = int(timestamp)
        init_mode = int(ctx.get_scalar("init_mode"))
        pose_estimate = ctx.get_ndarray("slam_pose", (4, 4))
        pose_estimate_se3 = SE3.from_matrix(pose_estimate)
        if init_mode > 1 and not self.init:
            self.offset = pose_estimate_se3 * self.ground_truth_index.nearest(timestamp_ns).se3().inverse()
            self.logger.debug(f"Front-End Initialization Done: Alignment offset: {self.offset}")
            self.init = True
        if not self.init:
            self.logger.debug("Frontend still not initialized")
            return None

        ground_truth = self.ground_truth_index.nearest(timestamp_ns)
        ground_truth_timestamp_ns = ground_truth.timestamp_ns
        timestamp_difference_ns = timestamp_ns - ground_truth_timestamp_ns
        self.logger.trace(f"Timestamp difference: {timestamp_difference_ns} ns")
        ground_truth_aligned_se3 = self.offset * ground_truth.se3()
        self.logger.trace(f"Ground truth aligned SE3: {ground_truth_aligned_se3}")

        ape_translation_m, ape_rotation_deg = calculate_ape(pose_estimate_se3, ground_truth_aligned_se3)
        metrics = self.update_metrics(
            ape_translation_m=ape_translation_m,
            ape_rotation_deg=ape_rotation_deg,
            timestamp_difference_ns=timestamp_difference_ns,
        )

        self.logger.trace(
            f"APE translation: {metrics.ape_translation_m} m, APE rotation: {metrics.ape_rotation_deg} deg, "
            f"RMSE translation: {metrics.rmse_translation_m} m, RMSE rotation: {metrics.rmse_rotation_deg} deg"
        )
        new_ctx = PipelineContext.from_timestamp(timestamp)
        (
            new_ctx.set_ndarray("ground_truth_aligned_se3", ground_truth_aligned_se3.as_matrix()).set_record_batch(
                "trajectory_metrics", metrics.as_arrow()
            )
        )
        return new_ctx

    def update_metrics(
        self,
        *,
        ape_translation_m: float,
        ape_rotation_deg: float,
        timestamp_difference_ns: int,
    ) -> TrajectoryMetrics:
        """Update running trajectory metrics and return the current metric snapshot."""
        self.ape_translation_sq_sum += ape_translation_m**2
        self.ape_rotation_sq_sum += ape_rotation_deg**2
        self.ape_samples_count += 1
        rmse_translation_m = (self.ape_translation_sq_sum / self.ape_samples_count) ** 0.5
        rmse_rotation_deg = (self.ape_rotation_sq_sum / self.ape_samples_count) ** 0.5
        gt_timestamp_diff_ms = timestamp_difference_ns / 1e6
        return TrajectoryMetrics(
            ape_translation_m=ape_translation_m,
            ape_rotation_deg=ape_rotation_deg,
            rmse_translation_m=rmse_translation_m,
            rmse_rotation_deg=rmse_rotation_deg,
            gt_timestamp_diff_ms=gt_timestamp_diff_ms,
            gt_timestamp_abs_diff_ms=abs(gt_timestamp_diff_ms),
            samples_count=self.ape_samples_count,
        )


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
