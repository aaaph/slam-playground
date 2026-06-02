from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from core.loop_closure.vpr_frame import VPRFrame
from core.loop_closure.vpr_verifier import VerifyResult


@dataclass(frozen=True, slots=True)
class LoopClosureVisualizationConfig:
    """Loop-closure visualization and payload formatting configuration."""

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


class LoopClosureOpenCVVisualizer:
    """OpenCV renderer for loop-closure debug overlays."""

    def __init__(self, config: LoopClosureVisualizationConfig | None = None) -> None:
        """Initialize the loop-closure visualizer."""
        self.config = config or LoopClosureVisualizationConfig()

    def draw_loop_image(
        self,
        query_image: NDArray[np.uint8],
        reference_image: NDArray[np.uint8],
        query_frame: VPRFrame,
        reference_frame: VPRFrame,
        verify_result: VerifyResult,
    ) -> NDArray[np.uint8]:
        """Draw a side-by-side loop-closure image with verified feature correspondences."""
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
            (self.config.text_origin_x, self.config.title_origin_y),
            query_image.shape[1] - 2 * self.config.text_origin_x,
            self.config.title_font_scale,
        )
        ref_text_bottom = self._draw_text_block(
            loop_image,
            [f"Ref Keyframe: {reference_frame.kf_id}"],
            (query_image.shape[1] + self.config.text_origin_x, self.config.title_origin_y),
            reference_image.shape[1] - 2 * self.config.text_origin_x,
            self.config.title_font_scale,
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
                self.config.text_origin_x,
                max(self.config.stats_origin_y, query_text_bottom, ref_text_bottom),
            ),
            loop_image.shape[1] - 2 * self.config.text_origin_x,
            self.config.stats_font_scale,
        )
        return loop_image

    def _to_rgb_image(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Convert a grayscale or RGB uint8 image to contiguous RGB."""
        image = np.ascontiguousarray(image, dtype=np.uint8)
        if image.ndim == self.config.grayscale_image_ndim:
            return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), dtype=np.uint8)
        if image.ndim == self.config.color_image_ndim and image.shape[2] == self.config.rgb_channels:
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
        query_uv = query_frame.left_uv
        reference_uv = reference_frame.left_uv

        for match, is_inlier in zip(verify_result.matches, verify_result.inlier_mask, strict=False):
            if not is_inlier:
                continue
            if match.queryIdx >= query_uv.shape[0] or match.trainIdx >= reference_uv.shape[0]:
                continue

            query_point = self._uv_to_point(query_uv[match.queryIdx])
            reference_point = self._uv_to_point(reference_uv[match.trainIdx])
            reference_point = (reference_point[0] + reference_x_offset, reference_point[1])

            cv2.circle(
                image,
                query_point,
                self.config.match_point_radius,
                self.config.match_inlier_color,
                cv2.FILLED,
            )
            cv2.circle(
                image,
                reference_point,
                self.config.match_point_radius,
                self.config.match_inlier_color,
                cv2.FILLED,
            )
            cv2.line(
                image,
                query_point,
                reference_point,
                self.config.match_inlier_color,
                self.config.match_line_thickness,
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
        x, y = origin
        for line in lines:
            for wrapped_line in self._wrap_text(line, max_width_px, font_scale):
                (_width, height), baseline = cv2.getTextSize(
                    wrapped_line,
                    self.config.text_font,
                    font_scale,
                    self.config.text_thickness,
                )
                cv2.putText(
                    image,
                    wrapped_line,
                    (x, y),
                    self.config.text_font,
                    font_scale,
                    self.config.match_text_color,
                    self.config.text_thickness,
                )
                y += height + baseline + self.config.text_line_gap_px
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
        (width, _height), _baseline = cv2.getTextSize(
            text,
            self.config.text_font,
            font_scale,
            self.config.text_thickness,
        )
        return width

    @staticmethod
    def _uv_to_point(uv: np.ndarray) -> tuple[int, int]:
        """Convert UV coordinates to an OpenCV integer point."""
        u, v = np.rint(uv).astype(int)
        return int(u), int(v)
