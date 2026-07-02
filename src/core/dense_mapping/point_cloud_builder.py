import numpy as np
from numpy.typing import NDArray

from core.dense_mapping.depth_filter import FilteredDepth
from core.transformations.special_euclidian_3_dim import SE3


class PointCloudBuilder:
    """Point cloud builder."""

    def __init__(self, k_matrix: np.ndarray, sample_stride: int) -> None:
        """Initialize the point cloud builder."""
        self.k_matrix = k_matrix
        self.sample_stride = sample_stride
        self.GRAYSCALE_IMAGE_NDIM = 2

    @classmethod
    def default_factory(cls, k_matrix: np.ndarray, sample_stride: int = 4) -> "PointCloudBuilder":
        """Create a default point cloud builder."""
        return cls(k_matrix, sample_stride=sample_stride)

    def _sample_depth_mask(self, valid_mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Sample the depth mask at a given stride."""
        stride = max(1, self.sample_stride)
        sampled_mask = np.zeros_like(valid_mask, dtype=bool)
        sampled_mask[::stride, ::stride] = valid_mask[::stride, ::stride]
        return sampled_mask

    def _project_depth_to_points(
        self, cam0_in_odom: SE3, depth: NDArray[np.float32], valid_mask: NDArray[np.bool_]
    ) -> NDArray[np.float32]:
        """Back-project selected depth pixels and transform endpoints into odom frame."""
        rows, cols = np.nonzero(valid_mask)
        if rows.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        z = depth[rows, cols].astype(np.float64, copy=False)
        x = (cols.astype(np.float64) - self.k_matrix[0, 2]) * z / self.k_matrix[0, 0]
        y = (rows.astype(np.float64) - self.k_matrix[1, 2]) * z / self.k_matrix[1, 1]
        points_cam0 = np.column_stack((x, y, z))

        rotation = cam0_in_odom.rotation().as_matrix()
        translation = cam0_in_odom.translation()
        points_odom = points_cam0 @ rotation.T + translation
        return points_odom.astype(np.float32, copy=False)

    def _pixel_colors_rgb(
        self,
        image: NDArray[np.uint8],
        valid_mask: NDArray[np.bool_],
    ) -> NDArray[np.float32]:
        """Return RGB colors for sampled pixels, promoting grayscale to RGB."""
        rows, cols = np.nonzero(valid_mask)
        if rows.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        if image.ndim == self.GRAYSCALE_IMAGE_NDIM:
            gray = image[rows, cols].astype(np.float32, copy=False)
            return np.repeat(gray[:, None], 3, axis=1)

        samples = image[rows, cols].astype(np.float32, copy=False)
        if samples.ndim == 1:
            return np.repeat(samples[:, None], 3, axis=1)
        if samples.shape[1] == 1:
            return np.repeat(samples, 3, axis=1)
        return samples[:, :3]

    def build_from_depth(self, cam0_in_odom: SE3, depth_estimate: FilteredDepth) -> NDArray[np.float32]:
        """Build the point cloud from the depth estimate."""
        depth_map = depth_estimate.depth_m
        valid_mask = self._sample_depth_mask(depth_estimate.valid_mask)
        xyz_odom = self._project_depth_to_points(cam0_in_odom, depth_map, valid_mask)
        colors_rgb = self._pixel_colors_rgb(depth_estimate.left_image, valid_mask)
        return np.column_stack((xyz_odom, colors_rgb))
