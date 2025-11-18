import jax.numpy as jnp
import numpy as np

from core.filter.augmentator import Augmentator
from core.filter.state import CameraClone, InertialState, State


class TestUnitInertialState:
    """Unit test for inertial state."""

    def test_should_be_possible_to_create(self):
        """Test that the inertial state can be created."""
        position = jnp.array([0, 0, 0])
        orientation = jnp.array([1, 0, 0, 0])
        velocity = jnp.array([0, 0, 0])
        acc_bias = jnp.array([0, 0, 0])
        gyro_bias = jnp.array([0, 0, 0])

        inertial_state = InertialState(
            p=position,
            q=orientation,
            v=velocity,
            b_a=acc_bias,
            b_g=gyro_bias,
        )
        assert inertial_state is not None

    def should_have_map_method(self):
        """Test that the inertial state has a map method."""
        inertial_state = InertialState(
            p=jnp.array([0, 0, 0]),
            q=jnp.array([1, 0, 0, 0]),
            v=jnp.array([0, 0, 0]),
            b_a=jnp.array([0, 0, 0]),
            b_g=jnp.array([0, 0, 0]),
        )
        assert hasattr(inertial_state, "map")
        inertial_state = inertial_state.map(
            lambda x: (x[0] + 1.0, jnp.array([x[1][0] + 2.0, 0, 0]), x[2], x[3], x[4], x[5], x[6])
        )
        assert jnp.allclose(inertial_state.p, jnp.array([2.0, 0, 0]))
        assert jnp.allclose(inertial_state.q, jnp.array([1, 0, 0, 0]))
        assert jnp.allclose(inertial_state.v, jnp.array([0, 0, 0]))
        assert jnp.allclose(inertial_state.b_a, jnp.array([0, 0, 0]))
        assert jnp.allclose(inertial_state.b_g, jnp.array([0, 0, 0]))

    def test_should_have_map_position_method(self):
        """Test that the inertial state has a map position method."""
        inertial_state = InertialState(
            p=jnp.array([15.0, 10.0, 0.5]),
            q=jnp.array([1, 0, 0, 0]),
            v=jnp.array([0, 0, 0]),
            b_a=jnp.array([0, 0, 0]),
            b_g=jnp.array([0, 0, 0]),
        )
        assert hasattr(inertial_state, "map_position")
        inertial_state = inertial_state.map_position(lambda x: x + jnp.array([2.0, 1, 0]))
        assert jnp.allclose(inertial_state.p, jnp.array([17.0, 11, 0.5]))

    def test_should_not_have_apply_timestamp_method(self):
        """Test that the inertial state has a apply timestamp method."""
        inertial_state = InertialState(
            p=jnp.array([15.0, 10.0, 0.5]),
            q=jnp.array([1, 0, 0, 0]),
            v=jnp.array([0, 0, 0]),
            b_a=jnp.array([0, 0, 0]),
            b_g=jnp.array([0, 0, 0]),
        )
        assert not hasattr(inertial_state, "apply_timestamp")
        inertial_state = inertial_state.map_position(lambda x: x + jnp.array([2.0, 1, 0]))
        assert jnp.allclose(inertial_state.p, jnp.array([17.0, 11, 0.5]))

    def test_inertial_state(self):
        """Test that the inertial state has a apply timestamp method."""
        inertial_state = InertialState(
            p=jnp.array([15.0, 10.0, 0.5]),
            q=jnp.array([1, 0, 0, 0]),
            v=jnp.array([0, 0, 0]),
            b_a=jnp.array([0, 0, 0]),
            b_g=jnp.array([0, 0, 0]),
        )
        assert inertial_state is not None

        pose = inertial_state.get_pose()
        assert pose.shape == (7,)
        assert jnp.allclose(pose[0:3], jnp.array([15.0, 10.0, 0.5]))
        assert jnp.allclose(pose[3:7], jnp.array([1, 0, 0, 0]))


class TestUnitState:
    """Unit test for state."""

    def test_should_be_possible_to_create(self):
        """Test that the state can be created."""
        state = State()
        assert state is not None

    def test_should_have_inertial_state_field(self):
        """Test that the state has an inertial state field."""
        state = State()
        assert hasattr(state, "inertial_state")

    def test_should_have_initialize_inertial_state_method(self):
        """Test that the state has an initialize inertial state method."""
        state = State()
        assert hasattr(state, "initialize_inertial_state")

    def test_should_initialize_inertial_state(self):
        """Test that the state can initialize the inertial state."""
        state = State()
        state.apply_timestamp(140.0).initialize_inertial_state(
            p=jnp.array([0, 0, 0]),
            q=jnp.array([1, 0, 0, 0]),
            v=jnp.array([0, 0, 0]),
            b_a=jnp.array([0, 0, 0]),
            b_g=jnp.array([0, 0, 0]),
        )
        assert state.inertial_state is not None
        assert state.ts == 140

    def test_should_have_covariance_field(self):
        """Test that the state has a covariance field."""
        state = State()
        assert hasattr(state, "covariance")

    def test_should_have_initialize_covariance_method(self):
        """Test that the state has an initialize covariance method."""
        state = State()
        assert hasattr(state, "initialize_covariance")
        state = state.initialize_covariance()
        assert state.covariance is not None

    def test_should_have_map_inertial_state_method(self):
        """Test that the state has a map inertial state method."""
        state = State()
        state.initialize_inertial_state(
            p=jnp.array([15.0, 10.0, 0.5]),
            q=jnp.array([1, 0, 0, 0]),
            v=jnp.array([0, 0, 0]),
            b_a=jnp.array([0, 0, 0]),
            b_g=jnp.array([0, 0, 0]),
        )
        assert hasattr(state, "map_inertial_state")
        state = state.map_inertial_state(
            lambda x: x.map_position(lambda x: x + jnp.array([1.0, 1, 0]))
            .map_position(lambda x: x + jnp.array([1.0, 0, 0]))
            .map_position(lambda x: x + jnp.array([0, 0, 0]))
        )
        assert jnp.allclose(state.inertial_state.p, jnp.array([17.0, 11, 0.5]))

    def test_should_have_sliding_window(self):
        """Test that the state has a sliding window. State should be able to manipulate the sliding window."""
        state = State()
        state.initialize_inertial_state(
            p=jnp.array([0, 0, 0]),
            q=jnp.array([1, 0, 0, 0]),
            v=jnp.array([0, 0, 0]),
            b_a=jnp.array([0, 0, 0]),
            b_g=jnp.array([0, 0, 0]),
        )
        assert state.sliding_window is not None
        assert hasattr(state.sliding_window, "get_oldest_than")
        assert callable(state.sliding_window.get_oldest_than)
        state.sliding_window.add(1.0, np.array([1.0, 0, 0, 1, 0, 0, 0]))
        state.sliding_window.add(2.0, np.array([2.0, 0, 0, 1, 0, 0, 0]))
        state.sliding_window.add(3.0, np.array([3.0, 0, 0, 1, 0, 0, 0]))
        oldest_than = state.sliding_window.get_oldest_than(2.0)
        assert len(oldest_than) == 1
        assert oldest_than[0].timestamp == 1.0

    def test_sliding_window_should_be_able_to_map_poses_of_clones(self):
        """Test that the sliding window can be mapped to map the poses of the clones."""
        state = State()
        state.sliding_window.add(1.0, np.array([1.0, 0, 0, 0, 0, 0, 1]))
        state.sliding_window.add(2.0, np.array([2.0, 0, 0, 0, 0, 0, 1]))
        state.sliding_window.add(3.0, np.array([3.0, 0, 0, 0, 0, 0, 1]))
        assert hasattr(state, "map_poses_in_sliding_window")
        assert callable(state.map_poses_in_sliding_window)

        def map_clone_pose(clone: CameraClone) -> tuple[np.ndarray, np.ndarray]:
            """Map the pose of the clone."""
            return clone.p + np.array([1.0, 0, 0]), clone.q

        state = state.map_poses_in_sliding_window(map_clone_pose)

        assert state.sliding_window.size() == 3
        first_clone = state.sliding_window.get_by_id(0)
        assert np.allclose(first_clone.p, np.array([2.0, 0, 0]))
        assert np.allclose(first_clone.q, np.array([0, 0, 0, 1]))
        second_clone = state.sliding_window.get_by_id(1)
        assert np.allclose(second_clone.p, np.array([3.0, 0, 0]))
        assert np.allclose(second_clone.q, np.array([0, 0, 0, 1]))
        third_clone = state.sliding_window.get_by_id(2)
        assert np.allclose(third_clone.p, np.array([4.0, 0, 0]))
        assert np.allclose(third_clone.q, np.array([0, 0, 0, 1]))

    def test_state_augmentation(self):
        """Test that the state can be augmented."""
        state = State()
        state.initialize_inertial_state(
            p=jnp.array([0, 0, 0]),
            q=jnp.array([1, 0, 0, 0]),
            v=jnp.array([0, 0, 0]),
            b_a=jnp.array([0, 0, 0]),
            b_g=jnp.array([0, 0, 0]),
        ).apply_timestamp(1.0)
        assert state is not None

        augmentator = Augmentator()
        _, state = augmentator.augment_clone(state)
        assert state is not None
        assert state.sliding_window is not None
        assert state.sliding_window.size() == 1

        for i in range(30):
            state = state.apply_timestamp(state.ts + i + 1.0)
            _, state = augmentator.augment_clone(state)

        assert state.sliding_window.size() > state.sliding_window.max_size
        newest_clone = state.sliding_window.get_by_id(state.sliding_window.next_id - 1)
        assert newest_clone is not None
        assert newest_clone.timestamp == state.ts
        assert newest_clone.p.shape == (3,)
        assert newest_clone.q.shape == (4,)
        assert state.sliding_window.ts_to_id[state.ts] == state.sliding_window.next_id - 1

        some_timestamp = state.ts - state.sliding_window.max_size + 1000

        state = state.apply_timestamp(some_timestamp).map_inertial_state(
            lambda x: x.map_position(lambda _: jnp.array([15.0, 10.0, 0.5]))
        )
        _, state = augmentator.augment_clone(state)
        clone = state.sliding_window.get_by_timestamp(some_timestamp)
        assert clone is not None
        assert clone.p.shape == (3,)
        assert clone.q.shape == (4,)
        assert jnp.allclose(clone.p, jnp.array([15.0, 10.0, 0.5]))
        assert jnp.allclose(clone.q, jnp.array([1, 0, 0, 0]))

        oldest_clone = state.sliding_window.get_oldest()
        assert oldest_clone is not None
        oldest_camera_id, should_be_oldest_clone = state.sliding_window.window.popitem(last=False)
        state.sliding_window.ts_to_id.pop(should_be_oldest_clone.timestamp, None)
        assert oldest_clone.clone_id == oldest_camera_id

        candidates = state.sliding_window.get_candidate_for_removal()
        assert candidates is not None

    def test_covariance_augmentation(self):
        """Test that the covariance can be augmented."""
        state = State()
        state.initialize_inertial_state(
            p=jnp.array([0, 0, 0]),
            q=jnp.array([1, 0, 0, 0]),
            v=jnp.array([0, 0, 0]),
            b_a=jnp.array([0, 0, 0]),
            b_g=jnp.array([0, 0, 0]),
        ).apply_timestamp(1.0)
        assert state is not None
        assert state.covariance is not None
        augmentator = Augmentator()
        _, state = augmentator.augment_clone(state)
        assert state is not None
        assert state.covariance is not None
        assert state.covariance.sigma.shape == (21, 21)

        # test max covarianve size, if window size is 30, the max is 15 + (6 * 30) = 195

        for i in range(40):
            state = state.apply_timestamp(state.ts + i + 1.0)
            _, state = augmentator.augment_clone(state)
