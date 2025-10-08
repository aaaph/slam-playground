import jax
import jax.numpy as jnp


class FeatureTrackerRegion:
    """Region of the feature tracker."""

    def __init__(self, region_id: tuple[int, int], mask: jax.Array) -> None:
        """Initialize the region."""
        self.region_id = region_id
        self.mask = mask

        coords = jnp.where(mask == 1)
        if len(coords[0]) == 0:
            raise ValueError("Region has no pixels")

        y_coords, x_coords = coords
        left_pixel = jnp.min(x_coords).item()
        right_pixel = jnp.max(x_coords).item()
        top_pixel = jnp.min(y_coords).item()
        bottom_pixel = jnp.max(y_coords).item()

        self.region_box = (left_pixel, top_pixel, right_pixel, bottom_pixel)

    def __repr__(self) -> str:
        """Return the representation of the region."""
        return f"FeatureTrackerRegion(region_id={self.region_id}, mask={self.mask})"

    @property
    def box(self) -> tuple[int, int, int, int]:
        """Get the box of the region."""
        return self.region_box

    def region_id_to_string(self) -> str:
        """Return the region id as a string."""
        return f"[{self.region_id[0]},{self.region_id[1]}]"
