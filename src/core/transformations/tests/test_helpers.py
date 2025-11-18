import numpy as np

from core.transformations.helpers import omega, skew


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
