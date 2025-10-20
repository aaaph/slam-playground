from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class Observation:
    """Represents an observation of a feature - uv pixels in the image."""

    feat_id: int
    timestamp: float
    cam_id: int
    keypoint: tuple[float, float]

    @staticmethod
    def from_uv(feat_id: int, timestamp: float, cam_id: int, uv: tuple[float, float]) -> "Observation":
        """Create an observation from uv pixels and descriptor."""
        u, v = uv
        return Observation(feat_id, timestamp, cam_id, (u, v))

    def uv_jax(self) -> jax.Array:
        """Get the uv pixels of the observation."""
        return jnp.array([self.keypoint[0], self.keypoint[1]])

    def uv_tuple(self) -> tuple[float, float]:
        """Get the uv pixels of the observation as a tuple."""
        return self.keypoint

    def __str__(self) -> str:
        """Return the string representation of the observation."""
        x, y = self.keypoint
        return f"Obs(timestamp={self.timestamp:.0f}, cam_id={self.cam_id}, keypoint={x}, {y})"

    def __repr__(self) -> str:
        """Return the representation of the observation."""
        return self.__str__()
