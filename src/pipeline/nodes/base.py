import os
from pathlib import Path

from core.camera_model.stereo_camera_model import StereoCameraModel
from core.camera_model.vio_context import ImuContext, VioContext
from dataset.manifest import DatasetRigConfig
from dataset.registry import DatasetRegistry
from dataset.sensor_config import CameraSensor, IMUSensor
from pipeline.runtime_config import NodePipelineConfig


class PipelineNode:
    """Shared helpers for loading pipeline node runtime config from env."""

    @classmethod
    def required_env(cls, name: str) -> str:
        """Return a required env value or raise a clear error."""
        value = os.getenv(name)
        if value is None:
            msg = f"{name} is not set"
            raise ValueError(msg)
        return value

    @classmethod
    def runtime_config(cls) -> NodePipelineConfig:
        """Return runtime config embedded into the materialized dataflow."""
        return NodePipelineConfig.from_env_variable(default_node_id=cls.__name__)

    @classmethod
    def runtime_config_as[T: NodePipelineConfig](cls, config_type: type[T]) -> T:
        """Return runtime config using a node-specific schema."""
        return config_type.from_env_variable(default_node_id=cls.__name__)

    @classmethod
    def repo_root_from_env(cls) -> Path:
        """Return REPO_ROOT from env, or cwd when the launcher did not provide it."""
        config = cls.runtime_config()
        if config.repo_root is not None:
            return config.repo_root.resolve()

        value = os.getenv("REPO_ROOT")
        return Path(value).resolve() if value is not None else Path.cwd().resolve()

    @classmethod
    def dataset_registry_from_env(cls) -> DatasetRegistry:
        """Create a dataset registry using the node runtime repo root."""
        return DatasetRegistry(repo_root=cls.repo_root_from_env())

    @classmethod
    def load_dataset_rig_from_env(cls) -> DatasetRigConfig:
        """Load DATASET_RIG_PATH relative to the runtime repo root."""
        config = cls.runtime_config()
        rig_path = config.dataset_rig_path or Path(cls.required_env("DATASET_RIG_PATH"))
        return cls.dataset_registry_from_env().load_rig(rig_path)

    @classmethod
    def create_vio_ctx(cls) -> VioContext:
        """Create a VIO context."""
        rig = cls.load_dataset_rig_from_env()
        cam0_sensor = CameraSensor.from_rig_config(rig.cam0)
        cam1_sensor = CameraSensor.from_rig_config(rig.cam1)
        imu_sensor = IMUSensor.from_rig_config(rig.imu0)
        camera_model = StereoCameraModel.from_cameras_config(cam0_sensor, cam1_sensor)
        stereo_ctx = camera_model.as_stereo_ctx()
        imu_ctx = ImuContext.from_imu_config(imu_sensor)
        return VioContext.from_stereo_and_imu_config(stereo_ctx, imu_ctx)

    @classmethod
    def create_stereo_camera_model(cls) -> StereoCameraModel:
        """Create a stereo camera model."""
        rig = cls.load_dataset_rig_from_env()
        cam0_sensor = CameraSensor.from_rig_config(rig.cam0)
        cam1_sensor = CameraSensor.from_rig_config(rig.cam1)
        return StereoCameraModel.from_cameras_config(cam0_sensor, cam1_sensor)

    def run(self) -> None: ...  # noqa: D102
