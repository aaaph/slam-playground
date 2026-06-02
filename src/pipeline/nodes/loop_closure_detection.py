import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np
from dora import Node

import pydbow3  # ty: ignore[unresolved-import]
from core.front_end.keyframe import KF, keyframe_schema
from core.graph_optimizer.pose_graph_optimizator import LoopClosure
from core.loop_closure.vpr_detector import VPRDetector
from core.loop_closure.vpr_frame import VPRFrame
from core.loop_closure.vpr_place_index import VPRPlaceIndex, VPRPlaceIndexConfig
from core.loop_closure.vpr_verifier import VPRFrameVerifier, VPRFrameVerifierConfig
from core.transformations.special_euclidian_3_dim import SE3
from dataset.euroc import EurocDataset
from logger import spawn_logger
from pipeline.annotations import Ctx, Metadata
from pipeline.context import PipelineContext
from pipeline.decorators import handle, on_input, reactive, send_pipeline_context_output
from visualizer.opencv.loop_closure import LoopClosureOpenCVVisualizer, LoopClosureVisualizationConfig


@dataclass(frozen=True, slots=True)
class LCDConfig:
    """Place recognition configuration."""

    vocabulary_path: Path
    max_k: int = field(default_factory=lambda: int(os.getenv("VPR_MAX_DB_RESULTS", "50")))
    recent_db_window: int = field(default_factory=lambda: int(os.getenv("VPR_RECENT_DB_WINDOW", "15")))
    alpha: float = field(default_factory=lambda: float(os.getenv("VPR_ALPHA", "0.1")))
    min_nss_factor: float = field(default_factory=lambda: float(os.getenv("VPR_MIN_NSS_FACTOR", "0.005")))
    island_db_gap: int = field(default_factory=lambda: int(os.getenv("VPR_ISLAND_DB_GAP", "2")))
    min_island_size: int = field(default_factory=lambda: int(os.getenv("VPR_MIN_ISLAND_SIZE", "3")))
    max_island_span: int = field(default_factory=lambda: int(os.getenv("VPR_MAX_ISLAND_SPAN", "30")))
    min_island_density: float = field(default_factory=lambda: float(os.getenv("VPR_MIN_ISLAND_DENSITY", "0.6")))
    min_island_weighted_score: float = field(
        default_factory=lambda: float(os.getenv("VPR_MIN_ISLAND_WEIGHTED_SCORE", "0.025"))
    )
    temporal_db_tolerance: int = field(default_factory=lambda: int(os.getenv("VPR_TEMPORAL_DB_TOLERANCE", "8")))
    temporal_min_overlap_ratio: float = field(
        default_factory=lambda: float(os.getenv("VPR_TEMPORAL_MIN_OVERLAP_RATIO", "0.3"))
    )
    deque_size: int = field(default_factory=lambda: int(os.getenv("VPR_DEQUE_SIZE", "5")))
    min_votes: int = field(default_factory=lambda: int(os.getenv("VPR_MIN_VOTES", "3")))
    min_essential_matches: int = field(default_factory=lambda: int(os.getenv("VPR_MIN_ESSENTIAL_MATCHES", "12")))
    min_essential_inliers: int = field(default_factory=lambda: int(os.getenv("VPR_MIN_ESSENTIAL_INLIERS", "8")))
    min_essential_matches_ratio: float = field(
        default_factory=lambda: float(os.getenv("VPR_MIN_ESSENTIAL_MATCHES_RATIO", "0.5"))
    )
    min_rigid3d_inliers: int = field(default_factory=lambda: int(os.getenv("VPR_MIN_RIGID3D_INLIERS", "12")))
    min_rigid3d_inliers_ratio: float = field(
        default_factory=lambda: float(os.getenv("VPR_MIN_RIGID3D_INLIERS_RATIO", "0.7"))
    )
    max_rigid3d_median_residual_m: float = field(
        default_factory=lambda: float(os.getenv("VPR_MAX_RIGID3D_MEDIAN_RESIDUAL_M", "0.05"))
    )
    orb_features: int = field(default_factory=lambda: int(os.getenv("VPR_ORB_FEATURES", "500")))
    orb_grid_rows: int = field(default_factory=lambda: int(os.getenv("VPR_ORB_GRID_ROWS", "1")))
    orb_grid_cols: int = field(default_factory=lambda: int(os.getenv("VPR_ORB_GRID_COLS", "1")))
    visualization: LoopClosureVisualizationConfig = field(default_factory=LoopClosureVisualizationConfig)

    def to_place_index_config(self) -> VPRPlaceIndexConfig:
        """Create a VPRPlaceIndexConfig from the PlaceRecognitionConfig."""
        return VPRPlaceIndexConfig(
            max_k=self.max_k,
            recent_db_window=self.recent_db_window,
            alpha=self.alpha,
            min_nss_factor=self.min_nss_factor,
            island_db_gap=self.island_db_gap,
            min_island_size=self.min_island_size,
            max_island_span=self.max_island_span,
            min_island_density=self.min_island_density,
        )

    def to_verifier_config(self) -> VPRFrameVerifierConfig:
        """Create a VPRFrameVerifierConfig from the PlaceRecognitionConfig."""
        return VPRFrameVerifierConfig(
            history_size=self.deque_size,
            min_island_weighted_score=self.min_island_weighted_score,
            min_votes=self.min_votes,
            temporal_db_tolerance=self.temporal_db_tolerance,
            temporal_min_overlap_ratio=self.temporal_min_overlap_ratio,
            min_essential_matches=self.min_essential_matches,
            min_essential_inliers=self.min_essential_inliers,
            min_essential_matches_ratio=self.min_essential_matches_ratio,
            min_rigid3d_inliers=self.min_rigid3d_inliers,
            min_rigid3d_inliers_ratio=self.min_rigid3d_inliers_ratio,
            max_rigid3d_median_residual_m=self.max_rigid3d_median_residual_m,
        )

    @classmethod
    def from_env_path(cls, env_variable: str = "VOCABULARY_PATH") -> Self:
        """Create a PlaceRecognitionConfig from a YAML file."""
        vocabulary_path = os.getenv(env_variable, "../vocabulary/ORBvoc.dbow3")
        return cls(vocabulary_path=Path(vocabulary_path))

    def __repr__(self) -> str:
        """Return a string representation of the PlaceRecognitionConfig."""
        return (
            f"PlaceRecognitionConfig(vocabulary_path={self.vocabulary_path}, "
            f"max_k={self.max_k}, recent_db_window={self.recent_db_window}, alpha={self.alpha}, "
            f"min_nss_factor={self.min_nss_factor}, island_db_gap={self.island_db_gap}, "
            f"min_island_size={self.min_island_size}, max_island_span={self.max_island_span}, "
            f"min_island_density={self.min_island_density}, "
            f"min_island_weighted_score={self.min_island_weighted_score}, "
            f"temporal_db_tolerance={self.temporal_db_tolerance}, "
            f"temporal_min_overlap_ratio={self.temporal_min_overlap_ratio}, "
            f"deque_size={self.deque_size}, min_votes={self.min_votes}, "
            f"min_essential_matches={self.min_essential_matches}, "
            f"min_essential_inliers={self.min_essential_inliers}, "
            f"min_essential_matches_ratio={self.min_essential_matches_ratio}, "
            f"min_rigid3d_inliers={self.min_rigid3d_inliers}, "
            f"min_rigid3d_inliers_ratio={self.min_rigid3d_inliers_ratio}, "
            f"max_rigid3d_median_residual_m={self.max_rigid3d_median_residual_m}, "
            f"orb_features={self.orb_features}, "
            f"orb_grid_rows={self.orb_grid_rows}, orb_grid_cols={self.orb_grid_cols}, "
            f"visualization={self.visualization})"
        )


@reactive
class LoopClosureDetectionNode:
    """Loop closure detection node."""

    def __init__(self, config: LCDConfig) -> None:
        """Initialize the loop closure detection node."""
        self.node = Node()
        self.logger = spawn_logger(app="lcd")

        self.euroc_dataset = EurocDataset.mh_01_easy()

        self.config = config
        self.vocabulary_path = self.config.vocabulary_path
        self.vocabulary = pydbow3.Vocabulary()
        self.vocabulary.load(str(self.vocabulary_path))
        self.logger.info(f"[VPR]: vocabulary loaded: {self.vocabulary_path}")

        self.vio_ctx = EurocDataset.mh_01_easy().config.as_vio_ctx()

        self.logger.info(self.config)
        self.vpr_detector = VPRDetector.from_stereo_ctx(
            self.vio_ctx.stereo,
            n_features=self.config.orb_features,
            grid=(self.config.orb_grid_rows, self.config.orb_grid_cols),
        )
        self.place_index = VPRPlaceIndex.from_vocabulary(
            self.vocabulary,
            self.config.to_place_index_config(),
        )
        self.vpr_frame_verifier = VPRFrameVerifier.default_factory(
            self.vio_ctx.stereo,
            self.config.to_verifier_config(),
        )
        self.image_cache = dict[int, np.ndarray]()
        self.loop_visualizer = LoopClosureOpenCVVisualizer(self.config.visualization)

    def run(self) -> None: ...  # noqa: D102

    @handle("keyframes", "visualization")
    def handle_keyframes(self, ctx: Ctx, metadata: Metadata) -> Ctx:
        """Handle the fixedlag frame event."""
        kf = KF.list_from_arrow(ctx.get_record_batch("keyframes", keyframe_schema))[0]
        image_shape = (ctx.get_scalar("height"), ctx.get_scalar("width"))
        left_image = ctx.get_image("left_rect", image_shape)
        right_image = ctx.get_image("right_rect", image_shape)

        next_id = self.place_index.next_frame_id()
        detection = self.vpr_detector.detect_stereo(left_image, right_image)
        query_frame = VPRFrame.from_detection(next_id, kf.keyframe_id, kf.timestamp, detection)
        self.logger.debug(f"[VPR]: query frame: {query_frame.kf_id}")
        accepted = False
        retrieval_ok, place, reference_frame = self.place_index.find_loop_candidate(query_frame)
        if retrieval_ok:
            self.logger.info(f"[VPR]: query frame {query_frame}")
            self.logger.info(f"[VPR]: reference frame: {reference_frame}")
            self.logger.info(f"[VPR]: reference place: {place}")
            verify_result = self.vpr_frame_verifier.verify(query_frame, place, reference_frame)
            self.logger.info(f"[VPR]: verify result: {verify_result}")
            accepted = verify_result.accepted

        if accepted:
            _detected_loop = LoopClosure(
                from_key=reference_frame.kf_id,
                to_key=query_frame.kf_id,
                transform=verify_result.se3,
                cam0_in_body=self.vio_ctx.stereo.cam0_in_body_se3,
            )
            detected_loop_ctx = PipelineContext.from_timestamp(ctx.get_scalar("timestamp"))
            detected_loop_ctx.set_scalar("from_key", reference_frame.kf_id)
            detected_loop_ctx.set_scalar("to_key", query_frame.kf_id)
            detected_loop_ctx.set_ndarray("transform", verify_result.se3.as_matrix())
            detected_loop_ctx.set_ndarray("cam0_in_body", self.vio_ctx.stereo.cam0_in_body_se3.as_matrix())
            send_pipeline_context_output(self.node, "detected_loop", detected_loop_ctx, metadata)
            reference_image = self.image_cache[reference_frame.frame_id]
            loop_image = self.loop_visualizer.draw_loop_image(
                left_image,
                reference_image,
                query_frame,
                reference_frame,
                verify_result,
            )
            (
                ctx.set_image("loop_image", loop_image)
                .set_scalar("loop_image_width", loop_image.shape[1])
                .set_scalar("loop_image_height", loop_image.shape[0])
                .set_ndarray("vpr_reference_frame", SE3.identity().as_matrix())
                .set_ndarray("vpr_query_frame", verify_result.se3.as_matrix())
                .set_ndarray("vpr_reference_points", reference_frame.pointcloud)
                .set_ndarray("vpr_query_points", query_frame.pointcloud)
                .set_scalar("vpr_reference_points_size", reference_frame.pointcloud_size)
                .set_scalar("vpr_query_points_size", query_frame.pointcloud_size)
            )

        self.place_index.add_frame(query_frame)
        self.image_cache[query_frame.frame_id] = left_image.copy()

        return ctx.set_record_batch("vpr_features", query_frame.active_feat_tensor.as_arrow()).set_image(
            "vpr_left_image", left_image
        )

    @on_input("helthcheck")
    def handle_helthcheck(self) -> None:
        """Handle the helthcheck event."""
        self.logger.trace("Still alive")


if __name__ == "__main__":
    node = LoopClosureDetectionNode(LCDConfig.from_env_path(env_variable="VOCABULARY_PATH"))
    node.run()
