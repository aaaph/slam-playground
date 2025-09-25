from collections.abc import Iterator

import jax.numpy as jnp

from core.filter.initializer import Initializer
from core.filter.state import State
from dataset.euroc import EurocDatasetSample


class TestFilter:
    """Test filter."""

    def test_initialize_filter_and_predict(self):
        """Test that the filter can be initialized and predicted."""
        initializer = Initializer()

        state = initializer.initialize_from_row(
            State(),
            (
                1.0,
                jnp.array([0, 0, 0]),
                jnp.array([1, 0, 0, 0]),
                jnp.array([0, 0, 0]),
                jnp.array([0, 0, 0]),
                jnp.array([0, 0, 0]),
                jnp.array([0, 0, 0]),
            ),
        )

        assert isinstance(state, State)

    def test_initialize_filter_from_generator(self):
        """Test that the filter can be initialized from a generator."""
        initializer = Initializer()

        def generator() -> Iterator[EurocDatasetSample]:
            yield EurocDatasetSample(
                timestamp=jnp.array(1.0),
                stereo=(jnp.array([0, 0, 0]), jnp.array([0, 0, 0])),
                gyro=(jnp.array([0, 0, 0]), jnp.array([0, 0, 0]), jnp.array([0, 0, 0])),
                acc=(jnp.array([0, 0, 0]), jnp.array([0, 0, 0]), jnp.array([0, 0, 0])),
                gt_position=(jnp.array(0), jnp.array(0), jnp.array(0)),
                gt_orientation=(
                    jnp.array(1),
                    jnp.array(0),
                    jnp.array(0),
                    jnp.array(0),
                ),
                gt_velocity=(jnp.array(0), jnp.array(0), jnp.array(0)),
                gt_gyro_bias=(jnp.array(0), jnp.array(0), jnp.array(0)),
                gt_acc_bias=(jnp.array(0), jnp.array(0), jnp.array(0)),
            )

        item = next(generator())
        row = (
            item["timestamp"],
            item["gt_position"],
            item["gt_orientation"],
            item["gt_velocity"],
            item["gt_gyro_bias"],
            item["gt_acc_bias"],
            jnp.array([0, 0, 0]),
        )
        state = initializer.initialize_from_row(State(), row)

        assert state is not None
        assert state.inertial_state is not None
        assert jnp.allclose(state.inertial_state.q, jnp.array([1, 0, 0, 0]))
        assert state.covariance is not None
        assert jnp.allclose(state.covariance.cov, jnp.eye(18))
        assert state.ts is not None
        assert state.ts > 0
        assert state.ts == item["timestamp"]
