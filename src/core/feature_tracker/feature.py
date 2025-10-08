import jax.numpy as jnp

from core.feature_tracker.observation import Observation


class Feature:
    """Represents a tracked feature with associated points and linear system matrices."""

    def __init__(self, feat_id: int) -> None:
        """Initialize a feature with the given ID."""
        self.feat_id = feat_id
        self.points: list[Observation] = []
        self.A = jnp.zeros((3, 3))
        self.B = jnp.zeros(3)

        self.p_F_in_G = None
        self.valid = False

    @staticmethod
    def from_observations(feat_id: int, observations: list[Observation]) -> "Feature":
        """Create a feature from a list of observations."""
        feature = Feature(feat_id)

        for observation in observations:
            feature.append(observation)

        return feature

    def append(self, observation: Observation) -> None:
        """Append an observation to the feature."""
        self.points.append(observation)

    def obs_count(self) -> int:
        """Get the number of observations for the feature."""
        return len(self.points)
