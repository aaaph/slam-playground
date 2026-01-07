import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from core.transformations.helpers import are, ate, omega, rre, rte, skew
from core.transformations.special_euclidian_3_dim import SE3


def test_skew_should_return_skew_symmetric_matrix():
    """Test that the skew function returns a skew-symmetric matrix."""
    vector = np.array([1, 2, 3])
    expected = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
    assert np.allclose(skew(vector), expected)
    assert skew(vector).shape == (3, 3)


def test_omega_should_return_omega_matrix():
    """Test that the omega function returns a 4x4 skew-symmetric matrix."""
    vector = np.array([1, 2, 3])
    skew_matrix = skew(vector)
    # assert that omega contains the skew matrix
    assert np.allclose(omega(vector)[:3, :3], -skew_matrix)
    # assert that omega contains the vector
    assert np.allclose(omega(vector)[3, :3], -vector)
    # assert that omega contains the negative of the vector
    assert np.allclose(omega(vector)[:3, 3], vector)
    assert omega(vector).shape == (4, 4)


class TestErrorsHelper:
    """Unit test for errors helper."""

    def test_ate_zero_translation(self):
        """Test that the ate function returns the correct value."""
        estimated_pose = SE3.identity()
        ground_truth_pose = SE3.identity()
        assert ate(estimated_pose, ground_truth_pose) == 0.0

    def test_ate_non_zero_translation(self):
        """Test that the ate function returns the correct value."""
        estimated_pose = SE3(t=np.array([1.0, 0.0, 0.0]))
        ground_truth_pose = SE3(t=np.array([-1.0, 0.0, 0.0]))
        assert ate(estimated_pose, ground_truth_pose) == 2.0

    def test_are_zero_rotation(self):
        """Test that the are function returns the correct value."""
        estimated_pose = SE3.identity()
        ground_truth_pose = SE3.identity()
        assert are(estimated_pose, ground_truth_pose) == 0.0

    def test_are_non_zero_rotation(self):
        """Test that the are function returns the correct value."""
        estimated_pose = SE3.identity()
        ground_truth_pose = SE3(r=Rotation.from_euler("z", 90, degrees=True))
        error = are(estimated_pose, ground_truth_pose)
        assert pytest.approx(error) == 90.0

    def test_rte_zero_translation(self):
        """Test that the rte function returns the correct value."""
        estimated_prev = SE3.identity()
        estimated_next = SE3.identity()
        ground_truth_prev = SE3.identity()
        ground_truth_next = SE3.identity()
        assert (
            rte(estimated_prev.inverse() * estimated_next, ground_truth_prev.inverse() * ground_truth_next) == 0.0
        )

    def test_rte_non_zero_translation(self):
        """Test that the rte function returns the correct value."""
        estimated_prev = SE3.identity()
        estimated_next = SE3(t=np.array([1.0, 0.0, 0.0]))
        ground_truth_prev = SE3.identity()
        ground_truth_next = SE3(t=np.array([-1.0, 0.0, 0.0]))
        assert (
            rte(estimated_prev.inverse() * estimated_next, ground_truth_prev.inverse() * ground_truth_next) == 2.0
        )

    def test_rre_zero_rotation(self):
        """Test that the rre function returns the correct value."""
        estimated_prev = SE3.identity()
        estimated_next = SE3.identity()
        ground_truth_prev = SE3.identity()
        ground_truth_next = SE3.identity()
        assert (
            rre(estimated_prev.inverse() * estimated_next, ground_truth_prev.inverse() * ground_truth_next) == 0.0
        )

    def test_rre_non_zero_rotation(self):
        """Test that the rre function returns the correct value."""
        estimated_prev = SE3.identity()
        estimated_next = SE3(r=Rotation.from_euler("z", 90, degrees=True))
        ground_truth_prev = SE3.identity()
        ground_truth_next = SE3.identity()
        error = rre(estimated_prev.inverse() * estimated_next, ground_truth_prev.inverse() * ground_truth_next)
        assert pytest.approx(error) == 90.0
