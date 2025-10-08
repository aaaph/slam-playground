import jax.numpy as jnp

from core.feature_tracker.feature import Feature
from core.feature_tracker.observation import Observation


class TestUnitFeature:
    """Unit test for feature."""

    def test_should_be_possible_to_create(self):
        """Test that the feature can be created."""
        feature = Feature(1)
        assert feature is not None

    def test_should_have_append_method(self):
        """Test that the feature has an append method."""
        feature = Feature(1)
        assert hasattr(feature, "append")

    def test_should_have_from_observations_method(self):
        """Test that the feature has a from_observations method."""
        feature = Feature.from_observations(1, [Observation(1, 1, jnp.array([0, 0]))])
        assert feature is not None

    def test_should_have_obs_count_method(self):
        """Test that the feature has an obs_count method."""
        feature = Feature.from_observations(
            1, [Observation(1, 1, jnp.array([0, 0])), Observation(1, 1, jnp.array([0, 0]))]
        )
        assert feature is not None
        assert feature.obs_count() == 2
