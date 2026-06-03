from dataclasses import dataclass

from dataset.sensor_config import CameraSensor


@dataclass
class RawStereoConfigDto:
    """Raw stereo configuration DTO."""

    cam0: CameraSensor
    cam1: CameraSensor
