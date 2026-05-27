import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

import pydbow3  # ty: ignore[unresolved-import]
from core.front_end.keyframe import KF, keyframe_schema
from core.graph_optimizer.pose_graph_optimizator import LoopClosure, PoseGraphOptimizator
from core.loop_closure.vpr_detector import VPRDetector
from core.loop_closure.vpr_frame import VPRFrame
from core.loop_closure.vpr_place_index import VPRPlaceIndex, VPRPlaceIndexConfig
from core.loop_closure.vpr_verifier import VerifyResult, VPRFrameVerifier, VPRFrameVerifierConfig
from core.transformations.special_euclidian_3_dim import SE3
from dataset.euroc import EurocDataset
from logger import spawn_logger
from pipeline.annotations import Ctx
from pipeline.decorators import on_input, on_stop, reactive, to_output


@dataclass(frozen=True, slots=True)
class VPRVisualizationConfig:
    """VPR visualization and payload formatting configuration."""

    grayscale_image_ndim: int = 2
    color_image_ndim: int = 3
    rgb_channels: int = 3
    point_xyz_dim: int = 3
    pointcloud_columns: int = 5
    match_point_radius: int = 3
    match_line_thickness: int = 1
    match_inlier_color: tuple[int, int, int] = (0, 255, 0)
    match_text_color: tuple[int, int, int] = (255, 0, 0)
    text_font: int = cv2.FONT_HERSHEY_SIMPLEX
    text_origin_x: int = 10
    title_origin_y: int = 30
    stats_origin_y: int = 60
    title_font_scale: float = 1.0
    stats_font_scale: float = 0.75
    text_thickness: int = 2
    text_line_gap_px: int = 8


@dataclass(frozen=True, slots=True)
class PlaceRecognitionConfig:
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
    visualization: VPRVisualizationConfig = field(default_factory=VPRVisualizationConfig)

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
    def from_env_path(cls, env_variable: str = "VOCABULARY_PATH") -> "PlaceRecognitionConfig":
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
class PlaceRecognitionNode:
    """Place recognition node. VPR - Visual Place Recognition."""

    def run(self) -> None: ...  # noqa: D102

    def __init__(self, config: PlaceRecognitionConfig) -> None:
        """Initialize the place recognition node."""
        self.logger = spawn_logger(app="place_recognition_node")

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
        self.pgo = PoseGraphOptimizator()

    @on_input("ctx")
    @to_output("ctx")
    def handle_ctx(self, ctx: Ctx) -> Ctx:
        """
        Handle the keyframe event.

        This function is called when a new keyframe is detected.
        It performs VPR detection, caching, and database insertion.
        The resulting VPR frame is then added to the feature tensor and returned.
        New KF
            -> build_current_vpr_frame
            -> query_most_similar_place
            -> verify_place
            -> make_loop_proposal
            -> insert_current_vpr_frame
        """
        if not ctx.exists("keyframes"):
            return ctx

        kf = KF.list_from_arrow(ctx.get_record_batch("keyframes", keyframe_schema))[0]
        image_shape = (ctx.get_scalar("height"), ctx.get_scalar("width"))
        left_image = ctx.get_image("left_rect", image_shape)
        right_image = ctx.get_image("right_rect", image_shape)

        next_id = self.place_index.next_frame_id()
        detection = self.vpr_detector.detect_stereo(left_image, right_image)
        query_frame = VPRFrame.from_detection(next_id, kf.keyframe_id, kf.timestamp, detection)
        self.pgo.update_by_pose(query_frame.kf_id, SE3.from_matrix(ctx.get_ndarray("optimized_pose", (4, 4))))

        retrieval_ok, place, reference_frame = self.place_index.find_loop_candidate(query_frame)
        if retrieval_ok:
            self.logger.info(f"[VPR]: query frame {query_frame}")
            self.logger.info(f"[VPR]: reference frame: {reference_frame}")
            self.logger.info(f"[VPR]: reference place: {place}")
            verify_result = self.vpr_frame_verifier.verify(query_frame, place, reference_frame)
            self.logger.info(f"[VPR]: verify result: {verify_result}")
            if verify_result.accepted:
                self.pgo.update_by_loop_closure(
                    LoopClosure(
                        reference_frame.kf_id,
                        query_frame.kf_id,
                        verify_result.se3,
                        self.vio_ctx.stereo.cam0_in_body_se3,
                    )
                )
                self.pgo.optimize()

                reference_image = self.image_cache[reference_frame.frame_id]
                loop_image = self.draw_loop_image(
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

        return (
            ctx.set_record_batch("vpr_features", query_frame.active_feat_tensor.as_arrow())
            .set_image("vpr_left_image", left_image)
            .set_record_batch("pgo_graph", self.pgo.to_trajectory().to_arrow())
            .reassemble()
        )

    def draw_loop_image(
        self,
        query_image: NDArray[np.uint8],
        reference_image: NDArray[np.uint8],
        query_frame: VPRFrame,
        reference_frame: VPRFrame,
        verify_result: VerifyResult,
    ) -> NDArray[np.uint8]:
        """Draw a side-by-side loop closure image with verified feature correspondences."""
        visualization_config = self.config.visualization
        query_image = self._to_rgb_image(query_image)
        reference_image = self._to_rgb_image(reference_image)
        loop_image = np.ascontiguousarray(np.concatenate([query_image, reference_image], axis=1), dtype=np.uint8)
        loop_image = self._draw_verified_matches(
            loop_image,
            query_frame,
            reference_frame,
            verify_result,
            reference_x_offset=query_image.shape[1],
        )
        query_text_bottom = self._draw_text_block(
            loop_image,
            [
                f"Query Keyframe: {query_frame.kf_id}",
                self._format_ransac_diff(verify_result),
            ],
            (visualization_config.text_origin_x, visualization_config.title_origin_y),
            query_image.shape[1] - 2 * visualization_config.text_origin_x,
            visualization_config.title_font_scale,
        )
        ref_text_bottom = self._draw_text_block(
            loop_image,
            [f"Ref Keyframe: {reference_frame.kf_id}"],
            (query_image.shape[1] + visualization_config.text_origin_x, visualization_config.title_origin_y),
            reference_image.shape[1] - 2 * visualization_config.text_origin_x,
            visualization_config.title_font_scale,
        )
        self._draw_text_block(
            loop_image,
            [
                (
                    f"essential: {verify_result.essential_inliers_ratio_str} "
                    f"({verify_result.essential_inliers_ratio:.2f}), "
                    f"geometric: {verify_result.geometric_inliers_ratio_str} "
                    f"({verify_result.geometric_inliers_ratio:.2f})"
                ),
            ],
            (
                visualization_config.text_origin_x,
                max(visualization_config.stats_origin_y, query_text_bottom, ref_text_bottom),
            ),
            loop_image.shape[1] - 2 * visualization_config.text_origin_x,
            visualization_config.stats_font_scale,
        )
        return loop_image

    def _to_rgb_image(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Convert a grayscale or BGR/RGB uint8 image to contiguous RGB."""
        visualization_config = self.config.visualization
        image = np.ascontiguousarray(image, dtype=np.uint8)
        if image.ndim == visualization_config.grayscale_image_ndim:
            return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), dtype=np.uint8)
        if (
            image.ndim == visualization_config.color_image_ndim
            and image.shape[2] == visualization_config.rgb_channels
        ):
            return np.ascontiguousarray(image, dtype=np.uint8)
        msg = f"Unsupported image shape for RGB visualization: {image.shape}"
        raise ValueError(msg)

    def _draw_verified_matches(
        self,
        image: NDArray[np.uint8],
        query_frame: VPRFrame,
        reference_frame: VPRFrame,
        verify_result: VerifyResult,
        *,
        reference_x_offset: int,
    ) -> NDArray[np.uint8]:
        """Draw rigid 3D inlier correspondences between query and reference images."""
        visualization_config = self.config.visualization
        query_uv = query_frame.left_uv
        reference_uv = reference_frame.left_uv
        # inlier_mask = verify_result.geometric.inlier_mask.astype(bool, copy=False)

        for match, is_inlier in zip(verify_result.matches, verify_result.inlier_mask, strict=False):
            if not is_inlier:
                continue
            if match.queryIdx >= query_uv.shape[0] or match.trainIdx >= reference_uv.shape[0]:
                continue

            query_point = PlaceRecognitionNode._uv_to_point(query_uv[match.queryIdx])
            reference_point = PlaceRecognitionNode._uv_to_point(reference_uv[match.trainIdx])
            reference_point = (reference_point[0] + reference_x_offset, reference_point[1])

            cv2.circle(
                image,
                query_point,
                visualization_config.match_point_radius,
                visualization_config.match_inlier_color,
                cv2.FILLED,
            )
            cv2.circle(
                image,
                reference_point,
                visualization_config.match_point_radius,
                visualization_config.match_inlier_color,
                cv2.FILLED,
            )
            cv2.line(
                image,
                query_point,
                reference_point,
                visualization_config.match_inlier_color,
                visualization_config.match_line_thickness,
            )

        return image

    @staticmethod
    def _format_ransac_diff(verify_result: VerifyResult) -> str:
        """Format the estimated RANSAC relative pose as compact overlay text."""
        ransac_diff_se3 = verify_result.se3.copy()
        translation = ransac_diff_se3.translation()
        quat = ransac_diff_se3.rotation().as_quat()
        if quat[3] < 0:
            quat = -quat
        return (
            "RANSAC ref_T_query: "
            f"t=({translation[0]:+.2f}, {translation[1]:+.2f}, {translation[2]:+.2f})m, "
            f"q=({quat[0]:+.3f}, {quat[1]:+.3f}, {quat[2]:+.3f}, {quat[3]:+.3f})"
        )

    def _draw_text_block(
        self,
        image: NDArray[np.uint8],
        lines: list[str],
        origin: tuple[int, int],
        max_width_px: int,
        font_scale: float,
    ) -> int:
        """Draw wrapped text lines and return the next safe baseline y-coordinate."""
        visualization_config = self.config.visualization
        x, y = origin
        for line in lines:
            for wrapped_line in self._wrap_text(line, max_width_px, font_scale):
                (_width, height), baseline = cv2.getTextSize(
                    wrapped_line,
                    visualization_config.text_font,
                    font_scale,
                    visualization_config.text_thickness,
                )
                cv2.putText(
                    image,
                    wrapped_line,
                    (x, y),
                    visualization_config.text_font,
                    font_scale,
                    visualization_config.match_text_color,
                    visualization_config.text_thickness,
                )
                y += height + baseline + visualization_config.text_line_gap_px
        return y

    def _wrap_text(self, text: str, max_width_px: int, font_scale: float) -> list[str]:
        """Wrap text to fit into an OpenCV overlay width."""
        if max_width_px <= 0:
            return [text]

        wrapped_lines: list[str] = []
        current_line = ""
        for word in text.split():
            candidate = f"{current_line} {word}" if current_line else word
            if self._text_width(candidate, font_scale) <= max_width_px:
                current_line = candidate
                continue

            if current_line:
                wrapped_lines.append(current_line)
            if self._text_width(word, font_scale) <= max_width_px:
                current_line = word
                continue

            split_word = self._split_word_by_width(word, max_width_px, font_scale)
            wrapped_lines.extend(split_word[:-1])
            current_line = split_word[-1]

        if current_line:
            wrapped_lines.append(current_line)
        return wrapped_lines or [text]

    def _split_word_by_width(self, word: str, max_width_px: int, font_scale: float) -> list[str]:
        """Split a long token that cannot be wrapped at spaces."""
        chunks: list[str] = []
        current_chunk = ""
        for char in word:
            candidate = f"{current_chunk}{char}"
            if not current_chunk or self._text_width(candidate, font_scale) <= max_width_px:
                current_chunk = candidate
                continue

            chunks.append(current_chunk)
            current_chunk = char

        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def _text_width(self, text: str, font_scale: float) -> int:
        """Measure rendered OpenCV text width in pixels."""
        visualization_config = self.config.visualization
        (width, _height), _baseline = cv2.getTextSize(
            text,
            visualization_config.text_font,
            font_scale,
            visualization_config.text_thickness,
        )
        return width

    @staticmethod
    def _uv_to_point(uv: np.ndarray) -> tuple[int, int]:
        """Convert UV coordinates to an OpenCV integer point."""
        u, v = np.rint(uv).astype(int)
        return int(u), int(v)

    @on_input("helthcheck")
    def handle_helthcheck(self) -> None:
        """Handle the helthcheck event."""
        self.logger.trace("Still alive")

    @on_stop
    def handle_shutdown(self) -> None:
        """Handle the shutdown event."""
        self.logger.info("Place recognition node stopping...")
        self.logger.info("Place recognition node stopped")


if __name__ == "__main__":
    PlaceRecognitionNode(PlaceRecognitionConfig.from_env_path("VOCABULARY_PATH")).run()
