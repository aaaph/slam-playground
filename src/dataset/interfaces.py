from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from datasets import Dataset


@runtime_checkable
class MonocularDataset(Protocol):
    """Dataset that can provide a monocular camera stream."""

    def monocular(self, *, camera: str = "cam0", decode_images: bool = True) -> Dataset:
        """Return a monocular image stream."""


@runtime_checkable
class StereoDataset(Protocol):
    """Dataset that can provide synchronized stereo frames."""

    def stereo(self) -> Dataset:
        """Return a stereo image stream."""


@runtime_checkable
class VioDataset(StereoDataset, Protocol):
    """Dataset that can provide the VIO input stream."""

    def imu_and_stereo(self, *, decode_images: bool = True) -> Dataset:
        """Return stereo frames grouped with IMU samples between frames."""
