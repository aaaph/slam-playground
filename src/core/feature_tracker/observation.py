from dataclasses import dataclass

import jax


@dataclass
class Observation:
    """Represents an observation of a feature - uv pixels in the image."""

    timestamp: float
    cam_id: int
    uv: jax.Array
