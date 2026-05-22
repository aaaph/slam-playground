import numpy as np
from scipy.spatial.transform import Rotation

from core.transformations.special_euclidian_3_dim import SE3


class TestUnitSE3:
    """Unit test for SE3 class."""

    def test_should_be_possible_to_create_with_default_values(self):
        """Test that the SE3 can be created."""
        se3 = SE3()
        assert se3 is not None
        assert hasattr(se3, "as_matrix")
        assert callable(se3.as_matrix)
        assert se3.as_matrix().shape == (4, 4)
        rot = se3.rotation()
        assert np.allclose(rot.as_quat(), np.array([0, 0, 0, 1]))
        assert np.array_equal(se3.translation(), np.zeros(3, dtype=np.float64))
        assert np.allclose(se3.as_matrix(), np.eye(4, dtype=np.float64))
        # test from_matrix
        se3_from_matrix = SE3.from_matrix(np.eye(4, dtype=np.float64))
        assert np.allclose(se3_from_matrix.as_matrix(), np.eye(4, dtype=np.float64))
        # should be possible to stringify
        assert str(se3) == "SE3(quat_xyzw=[0. 0. 0. 1.], vec=[0. 0. 0.])"

    def test_se3_mul_should_return_correct_result(self):
        """Test that the SE3 mul method returns the correct result."""
        parent = SE3()  # in world frame, default values
        # child 1 in parent
        child1 = SE3(t=np.array([2, 0, 0]), r=Rotation.from_quat(np.array([0, 0, 0, 1])))
        # child 2 in child 1
        child2_quat = np.array([0.828459, 0.058956, 0.553641, -0.060514])
        child2_translation = np.array([1.2500000000000002, 2.1650635094610964, 0])
        child2 = SE3(t=child2_translation, r=Rotation.from_quat(child2_quat))

        child_1_in_parent = parent * child1
        child_2_in_parent = parent * child1 * child2  # Compose parent->child1->child2

        child_1_in_parent_quat = child_1_in_parent.rotation().as_quat()
        child_1_in_parent_translation = child_1_in_parent.translation()
        assert np.allclose(np.round(child_1_in_parent_quat, 2), np.array([0, 0, 0, 1]))
        assert np.allclose(np.round(child_1_in_parent_translation, 2), np.array([2, 0, 0]))

        child_2_in_parent_quat = child_2_in_parent.rotation().as_quat()
        child_2_in_parent_translation = child_2_in_parent.translation()
        assert np.allclose(np.round(child_2_in_parent_quat, 2), np.array([0.83, 0.06, 0.55, -0.06]))
        assert np.allclose(np.round(child_2_in_parent_translation, 2), np.array([3.25, 2.17, 0]))

    def test_from_quat_and_translation(self):
        """Test that the SE3 can be created from a quaternion and a translation."""
        quat = np.array([0.828459, 0.058956, 0.553641, -0.060514])
        translation = np.array([1.2500000000000002, 2.1650635094610964, 0])
        se3 = SE3.from_quat_and_translation(quat, translation)
        assert se3 is not None
        assert np.allclose(se3.rotation().as_quat(), quat)
        assert np.allclose(se3.translation(), translation)

    def test_from_matrix_should_copy_readonly_input_views(self):
        """SE3 should not retain readonly matrix views from Arrow-backed arrays."""
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = np.array([1.0, 2.0, 3.0])
        matrix.setflags(write=False)

        se3 = SE3.from_matrix(matrix)
        composed = SE3.identity() * se3

        assert composed.translation().flags.writeable
        np.testing.assert_allclose(composed.translation(), np.array([1.0, 2.0, 3.0]))

    def test_act_on_vector(self):
        """Test that the SE3 can act on a vector."""
        se3 = SE3()
        vector = np.array([1, 2, 3])

        result = se3 @ vector
        assert np.allclose(result, vector)

    def test_from_rpy_xyz(self):
        """Test that the SE3 can be created from a roll, pitch, yaw and a translation."""
        rpy = np.array([0.1, 0.2, 0.3])
        translation = np.array([1.0, 2.0, 3.0])
        se3 = SE3.from_rpy_xyz(rpy, translation)
        assert se3 is not None
        assert np.allclose(se3.rotation().as_euler("xyz"), rpy)
        assert np.allclose(se3.translation(), translation)
        # validated with https://dugas.ch/transform_viewer/multi.html
        assert np.allclose(
            se3.rotation().as_quat(),
            np.array([0.034270798550482096, 0.10602051106179564, 0.14357217502739192, 0.9833474432563557]),
        )

    def test_equality(self):
        """Test that the SE3 can be compared for equality."""
        se3_1 = SE3()
        se3_2 = SE3()
        assert se3_1 == se3_2
        se3_1 = SE3(t=np.array([1, 0, 0]))
        se3_2 = SE3(t=np.array([1, 0, 0]))
        assert se3_1 == se3_2
        se3_1 = SE3(r=Rotation.from_quat(np.array([0, 0, 0, 1])))
        se3_2 = SE3(r=Rotation.from_quat(np.array([0, 0, 0, 1])))
        assert se3_1 == se3_2
