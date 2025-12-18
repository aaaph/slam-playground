from dataclasses import dataclass

from dataset.dataset_config import CameraConfig


@dataclass
class RawStereoConfigDto:
    """Raw stereo configuration DTO."""

    cam0: CameraConfig
    cam1: CameraConfig
