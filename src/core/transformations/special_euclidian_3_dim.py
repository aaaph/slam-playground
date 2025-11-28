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
        self._translation = t

    def act_on_vector(self, vector: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Apply SE3 transform to a 3D vector.

        Args:
            vector: (3,) numpy array in source frame

        Returns:
            (3,) numpy array in target frame

        """
        return self._rot.apply(vector) + self._translation

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

    def __repr__(self) -> str:
        """Return a string representation of the SE3 transformation."""
        quat = Rotation.from_matrix(self.transform[:3, :3]).as_quat()
        translation = self.transform[:3, 3]
        return f"SE3(quaternion={quat}, translation={translation})"

    def as_matrix(self) -> NDArray[np.float64]:  # shape: (4, 4)
        """Get the matrix representation of the SE3 transformation."""
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self._rot.as_matrix()
        matrix[:3, 3] = self._translation
        return matrix

    @staticmethod
    def from_matrix(matrix: NDArray[np.float64]) -> "SE3":  # shape: (4, 4)
        """Create an SE3 transformation from a matrix."""
        rot = Rotation.from_matrix(matrix[:3, :3])
        translation = matrix[:3, 3]
        return SE3(rot, translation)

    @staticmethod
    def from_quat_and_translation(quat: NDArray[np.float64], translation: NDArray[np.float64]) -> "SE3":
        """Create an SE3 from a quaternion and a translation."""
        if quat.shape != (4,):
            raise ValueError("Quaternion must be a 4-element array.")
        rot = Rotation.from_quat(quat)
        return SE3(rot, translation)

    @staticmethod
    def identity() -> "SE3":
        """Create an identity SE3 transformation."""
        return SE3(Rotation.identity(), np.zeros(3, dtype=np.float64))
