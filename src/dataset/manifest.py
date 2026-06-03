from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from yaml import safe_load


class TransformMatrixConfig(BaseModel):
    """Homogeneous transform matrix stored in dataset rig YAML."""

    rows: int
    cols: int
    data: list[float]

    @model_validator(mode="after")
    def validate_shape(self) -> TransformMatrixConfig:
        """Validate that flat matrix data matches rows and columns."""
        expected_items = self.rows * self.cols
        if len(self.data) != expected_items:
            msg = f"Transform matrix has {len(self.data)} values, expected {expected_items}"
            raise ValueError(msg)
        return self


class CameraRigConfig(BaseModel):
    """Camera calibration in normalized rig config."""

    model_config = ConfigDict(populate_by_name=True)

    rate_hz: float
    resolution: tuple[int, int]
    camera_model: str
    intrinsics: tuple[float, float, float, float]
    distortion_model: str
    distortion_coefficients: list[float]
    body_sensor_transform: TransformMatrixConfig = Field(alias="T_BS")


class ImuRigConfig(BaseModel):
    """IMU calibration and noise parameters in normalized rig config."""

    model_config = ConfigDict(populate_by_name=True)

    rate_hz: float
    body_sensor_transform: TransformMatrixConfig = Field(alias="T_BS")
    gyroscope_noise_density: float
    gyroscope_random_walk: float
    accelerometer_noise_density: float
    accelerometer_random_walk: float


class DatasetRigConfig(BaseModel):
    """Normalized sensor rig config for a supported dataset family."""

    name: str
    cam0: CameraRigConfig
    cam1: CameraRigConfig
    imu0: ImuRigConfig


class DatasetStreamsConfig(BaseModel):
    """Dataset stream file layout relative to dataset root."""

    cam0: Path
    cam1: Path
    imu0: Path
    ground_truth: Path | None = None


class DatasetManifest(BaseModel):
    """Dataset manifest resolved by name."""

    name: str
    type: str
    root: Path
    rig: Path
    streams: DatasetStreamsConfig
    cache: Path | None = None


class ResolvedDatasetManifest(BaseModel):
    """Dataset manifest resolved together with its sensor rig."""

    dataset: DatasetManifest
    rig: DatasetRigConfig


class DatasetLocalStatus(BaseModel):
    """Local availability status for a dataset manifest."""

    exists: bool
    verified: bool
    issues: list[Path]


class DatasetManifestLoader:
    """Load dataset manifests and normalized sensor rigs."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        dataset_dir: Path | None = None,
    ) -> None:
        """Create a dataset manifest loader."""
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.dataset_dir = self._resolve_path(dataset_dir or Path("datasets"))

    def resolve(self, name: str) -> ResolvedDatasetManifest:
        """Load a dataset manifest and its referenced sensor rig."""
        dataset = self.load_dataset(name)
        rig = self.load_rig(dataset.rig)
        return ResolvedDatasetManifest(dataset=dataset, rig=rig)

    def load_dataset(self, name: str) -> DatasetManifest:
        """Load a dataset manifest by name."""
        raw_manifest = self._load_yaml(self.dataset_dir / f"{name}.yaml")
        raw_manifest.setdefault("name", name)
        return DatasetManifest.model_validate(raw_manifest)

    def list_datasets(self) -> list[DatasetManifest]:
        """List dataset manifests available in the dataset registry directory."""
        manifests = []
        for manifest_path in sorted(self.dataset_dir.glob("*.yaml")):
            raw_manifest = self._load_yaml(manifest_path)
            raw_manifest.setdefault("name", manifest_path.stem)
            manifests.append(DatasetManifest.model_validate(raw_manifest))
        return manifests

    def local_status(self, manifest: DatasetManifest) -> DatasetLocalStatus:
        """Inspect whether a dataset manifest is available on local disk."""
        root = self._resolve_path(manifest.root)
        missing_streams = [
            path for path in self._resolve_stream_paths(manifest, root).values() if not path.exists()
        ]
        exists = root.exists()
        issues = ([] if exists else [root]) + missing_streams
        return DatasetLocalStatus(
            exists=exists,
            verified=not issues,
            issues=issues,
        )

    def load_rig(self, path: Path) -> DatasetRigConfig:
        """Load normalized sensor rig config."""
        rig_path = self._resolve_path(path)
        return DatasetRigConfig.model_validate(self._load_yaml(rig_path))

    def resolve_path(self, path: Path) -> Path:
        """Resolve a manifest-level path against the repository root."""
        return self._resolve_path(path)

    def _load_yaml(self, config_path: Path) -> dict[str, Any]:
        if not config_path.exists():
            msg = f"Config file not found: {config_path}"
            raise FileNotFoundError(msg)
        with config_path.open("r", encoding="utf-8") as f:
            raw_config = safe_load(f) or {}
        if not isinstance(raw_config, dict):
            msg = f"Dataset config must be a mapping: {config_path}"
            raise TypeError(msg)
        return raw_config

    def _resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else self.repo_root / path

    @staticmethod
    def _resolve_stream_path(root: Path, path: Path) -> Path:
        return path if path.is_absolute() else root / path

    def _resolve_stream_paths(self, manifest: DatasetManifest, root: Path) -> dict[str, Path]:
        raw_streams = manifest.streams.model_dump(exclude_none=True)
        return {name: self._resolve_stream_path(root, Path(path)) for name, path in raw_streams.items()}
