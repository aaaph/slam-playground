import gtsam
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


class SE3:
    """Special Euclidean Group in 3 dimensions."""

    def __init__(self, r: Rotation | None = None, t: np.ndarray | None = None) -> None:
        """Initialize the SE3 object."""
        if r is None:
            r = Rotation.identity()
        if t is None:
            t = np.zeros(3, dtype=np.float64)

        self._rot = r
        self._translation = np.array(t, dtype=np.float64, copy=True)

    def act_on_vector(self, vector: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Apply SE3 transform to a 3D vector.

        Args:
            vector: (3,) numpy array in source frame

        Returns:
            (3,) numpy array in target frame

        """
        return self._rot.apply(vector) + self._translation

    def inverse(self) -> "SE3":
        """Inverse the SE3 transformation."""
        new_rot = self._rot.inv()
        new_translation = -new_rot.as_matrix() @ self._translation
        return SE3(new_rot, new_translation)

    def rotation(self) -> Rotation:
        """Get the rotation of the SE3 transformation."""
        return self._rot

    def translation(self) -> NDArray[np.float64]:  # shape: (3,)
        """Get the translation of the SE3 transformation."""
        return self._translation

    def __mul__(self, other: "SE3") -> "SE3":
        """Compose two SE3 transformations."""
        new_rot = self._rot * other._rot
        new_translation = self._rot.apply(other._translation) + self._translation
        return SE3(new_rot, new_translation)

    def __matmul__(self, v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply SE3 transform to a 3D vector."""
        return self.act_on_vector(v)

    def __eq__(self, other: object) -> bool:
        """Compare two SE3 transformations for equality."""
        if not isinstance(other, SE3):
            return False
        return self._rot.approx_equal(other._rot) and np.allclose(self._translation, other._translation)

    def __hash__(self) -> int:
        """Compute a hash based on rotation (quaternion) and translation."""
        quat = tuple(np.round(self._rot.as_quat().astype(np.float64), decimals=12))
        translation = tuple(np.round(self._translation.astype(np.float64), decimals=12))
        return hash((quat, translation))

    def __repr__(self) -> str:
        """Return a string representation of the SE3 transformation."""
        rot = self.rotation()
        translation = self.translation()
        quat = rot.as_quat()
        return f"SE3(quat_xyzw={quat}, vec={translation})"

    def copy(self) -> "SE3":
        """Copy the SE3 transformation."""
        matrix = self.as_matrix()
        return SE3.from_matrix(matrix.copy())

    def as_matrix(self) -> NDArray[np.float64]:  # shape: (4, 4)
        """Get the matrix representation of the SE3 transformation."""
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self._rot.as_matrix()
        matrix[:3, 3] = self._translation
        return matrix

    def as_gtsam_pose(self) -> "gtsam.Pose3":
        """Convert to GTSAM Pose3."""
        rot = gtsam.Rot3(self._rot.as_matrix())
        vec = gtsam.Point3(*self._translation)
        return gtsam.Pose3(rot, vec)

    def as_flat_ndarray(self) -> NDArray[np.float64]:
        """Convert to a flat numpy array."""
        quat = np.array(self._rot.as_quat(), dtype=np.float64)
        translation = np.array(self._translation, dtype=np.float64)
        return np.concatenate([quat, translation])

    @staticmethod
    def from_flat_ndarray(array: NDArray[np.float64]) -> "SE3":
        """Create an SE3 from a flat numpy array."""
        quat = array[:4]
        translation = array[4:7]
        return SE3.from_quat_and_translation(quat, translation)

    @staticmethod
    def from_matrix(matrix: NDArray[np.float64]) -> "SE3":  # shape: (4, 4)
        """Create an SE3 transformation from a matrix."""
        matrix = np.asarray(matrix, dtype=np.float64)
        rot = Rotation.from_matrix(np.array(matrix[:3, :3], dtype=np.float64, copy=True))
        translation = np.array(matrix[:3, 3], dtype=np.float64, copy=True)
        return SE3(rot, translation)

    @staticmethod
    def from_quat_and_translation(quat: NDArray[np.float64], translation: NDArray[np.float64]) -> "SE3":
        """Create an SE3 from a quaternion and a translation."""
        quat = np.array(quat, dtype=np.float64)
        translation = np.array(translation, dtype=np.float64)
        if quat.shape != (4,):
            raise ValueError("Quaternion must be a 4-element array.")
        rot = Rotation.from_quat(quat)
        return SE3(rot, translation)

    @staticmethod
    def from_rpy_xyz(rpy: NDArray[np.float64], translation: NDArray[np.float64]) -> "SE3":
        """Create an SE3 from a roll, pitch, yaw and a translation."""
        rpy = np.array(rpy, dtype=np.float64)
        translation = np.array(translation, dtype=np.float64)
        if rpy.shape != (3,):
            raise ValueError("RPY must be a 3-element array.")
        rot = Rotation.from_euler("xyz", rpy)
        return SE3(rot, translation)

    @staticmethod
    def from_gtsam_pose(pose: "gtsam.Pose3") -> "SE3":
        """Create an SE3 from a GTSAM Pose3."""
        rot = Rotation.from_matrix(pose.rotation().matrix())
        translation = pose.translation()
        return SE3(rot, translation)

    @staticmethod
    def identity() -> "SE3":
        """Create an identity SE3 transformation."""
        return SE3(Rotation.identity(), np.zeros(3, dtype=np.float64))
