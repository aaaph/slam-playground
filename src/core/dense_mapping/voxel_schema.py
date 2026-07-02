class VoxelSchema:
    """Voxel schema."""

    VOXEL_KEY_X = 0
    VOXEL_KEY_Y = 1
    VOXEL_KEY_Z = 2
    VOXEL_HITS = 3
    VOXEL_OBSERVATIONS = 4
    VOXEL_COLOR_R = 5
    VOXEL_COLOR_G = 6
    VOXEL_COLOR_B = 7
    VOXEL_CENTER_X = 8
    VOXEL_CENTER_Y = 9
    VOXEL_CENTER_Z = 10
    VOXEL_STATUS = 11

    VOXEL_KEY = slice(VOXEL_KEY_X, VOXEL_KEY_Z + 1)
    VOXEL_CENTER = slice(VOXEL_CENTER_X, VOXEL_CENTER_Z + 1)
    VOXEL_COLOR = slice(VOXEL_COLOR_R, VOXEL_COLOR_B + 1)

    @classmethod
    def count(cls) -> int:
        """Get the count of the voxel schema."""
        return 12
