from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator
from yaml import safe_load

from dataset.manifest import DatasetManifest, DatasetRigConfig  # noqa: TC001 - pydantic model annotations.
from dataset.registry import DatasetRegistry


class VisualizationSink(StrEnum):
    """Where the Rerun visualizer should send data."""

    APP = "app"
    FILE = "file"
    BOTH = "both"
    OFF = "off"


class RunMode(StrEnum):
    """How a pipeline run should start."""

    MANUAL = "manual"
    BATCH_FRACTION = "batch_fraction"


class DataflowProfile(BaseModel):
    """Dataflow graph preset resolved by name."""

    name: str
    template: Path
    build: bool = False


class DataflowSelector(BaseModel):
    """Dataflow selector declared in a composite profile."""

    template: str
    build: bool = False


class VisualizationProfile(BaseModel):
    """Visualization behavior for the selected dataflow."""

    sink: VisualizationSink = VisualizationSink.APP
    output: Path | None = None


class RunProfile(BaseModel):
    """Run-mode preset and automation options."""

    mode: RunMode = RunMode.MANUAL
    fraction: float | None = Field(default=None, gt=0.0, le=1.0)
    autostart_after_ready: bool = False
    stop_after_dataset_done: bool = False

    @model_validator(mode="after")
    def validate_fraction_mode(self) -> RunProfile:
        """Require fraction only for batch-fraction runs."""
        if self.mode == RunMode.BATCH_FRACTION and self.fraction is None:
            msg = "run.fraction is required when run.mode is batch_fraction"
            raise ValueError(msg)
        if self.mode != RunMode.BATCH_FRACTION and self.fraction is not None:
            msg = "run.fraction is only valid when run.mode is batch_fraction"
            raise ValueError(msg)
        return self


class CompositeProfile(BaseModel):
    """Named composition of dataset, dataflow, visualization, and run mode."""

    name: str | None = None
    dataset: str | None = None
    dataflow: DataflowSelector | None = None
    visualization: VisualizationProfile = Field(default_factory=VisualizationProfile)
    run: RunProfile = Field(default_factory=RunProfile)

    @field_validator("dataflow", mode="before")
    @classmethod
    def normalize_dataflow(cls, value: object) -> object:
        """Allow legacy `dataflow: vio-dataflow.yml` profile syntax."""
        if isinstance(value, str):
            return {"template": value}
        return value


class ProfileOverrides(BaseModel):
    """CLI overrides applied after a composite profile is loaded."""

    dataset: str | None = None
    dataflow: str | None = None
    visualization_sink: VisualizationSink | None = None
    run_mode: RunMode | None = None
    fraction: float | None = Field(default=None, gt=0.0, le=1.0)


class ResolvedPipelineProfile(BaseModel):
    """Fully resolved profile snapshot ready to materialize into a run config."""

    repo_root: Path
    profile: str | None = None
    dataset: DatasetManifest
    rig: DatasetRigConfig
    dataflow: DataflowProfile
    visualization: VisualizationProfile
    run: RunProfile


class PipelineProfileResolver:
    """Resolve named pipeline profiles and CLI overrides."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        profile_dir: Path | None = None,
        dataset_dir: Path | None = None,
        dataflow_dir: Path | None = None,
    ) -> None:
        """Create a profile resolver."""
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.profile_dir = self._resolve_path(profile_dir or Path("config/profile"))
        self.dataset_registry = DatasetRegistry(repo_root=self.repo_root, dataset_dir=dataset_dir)
        self.dataflow_dir = self._resolve_path(dataflow_dir or Path("pipeline"))

    def resolve(
        self, *, profile: str | None = None, overrides: ProfileOverrides | None = None
    ) -> ResolvedPipelineProfile:
        """Resolve a composite profile with optional CLI overrides."""
        overrides = overrides or ProfileOverrides()
        composite = self.load_profile(profile) if profile is not None else CompositeProfile()

        dataset_name = overrides.dataset or composite.dataset
        dataflow_selector = self._resolve_dataflow_selector(composite.dataflow, overrides)

        if dataset_name is None:
            raise ValueError("dataset must be provided by --dataset or profile.dataset")
        if dataflow_selector is None:
            raise ValueError("dataflow must be provided by --dataflow or profile.dataflow")

        resolved_dataset = self.dataset_registry.resolve(dataset_name)
        dataset = resolved_dataset.dataset
        rig = resolved_dataset.rig
        dataflow = self.load_dataflow(dataflow_selector)
        visualization = self._resolve_visualization(composite.visualization, overrides)
        run = self._resolve_run(composite.run, overrides)

        return ResolvedPipelineProfile(
            repo_root=self.repo_root,
            profile=composite.name,
            dataset=dataset,
            rig=rig,
            dataflow=dataflow,
            visualization=visualization,
            run=run,
        )

    def load_profile(self, name: str) -> CompositeProfile:
        """Load a composite profile by name."""
        raw_profile = self._load_yaml(self.profile_dir / f"{name}.yaml")
        raw_profile.setdefault("name", name)
        return CompositeProfile.model_validate(raw_profile)

    def load_dataset(self, name: str) -> DatasetManifest:
        """Load a dataset manifest by name."""
        return self.dataset_registry.find(name)

    def load_rig(self, path: Path) -> DatasetRigConfig:
        """Load normalized sensor rig config."""
        return self.dataset_registry.load_rig(path)

    def load_dataflow(self, selector: str | DataflowSelector) -> DataflowProfile:
        """Resolve a dataflow by descriptor filename."""
        dataflow_selector = DataflowSelector(template=selector) if isinstance(selector, str) else selector
        dataflow_path = self._resolve_dataflow_path(dataflow_selector.template)
        if not dataflow_path.exists():
            msg = f"Unknown dataflow '{dataflow_selector.template}': {dataflow_path}"
            raise FileNotFoundError(msg)
        return DataflowProfile(
            name=dataflow_path.name,
            template=dataflow_path.relative_to(self.repo_root),
            build=dataflow_selector.build,
        )

    @staticmethod
    def _resolve_dataflow_selector(
        base: DataflowSelector | None,
        overrides: ProfileOverrides,
    ) -> DataflowSelector | None:
        if overrides.dataflow is not None:
            return DataflowSelector(template=overrides.dataflow, build=base.build if base is not None else False)
        return base

    def _resolve_visualization(
        self,
        base: VisualizationProfile,
        overrides: ProfileOverrides,
    ) -> VisualizationProfile:
        raw_visualization = base.model_dump()
        cli_overrides = _without_none(
            {
                "sink": overrides.visualization_sink,
            }
        )
        return VisualizationProfile.model_validate(_deep_merge(raw_visualization, cli_overrides))

    def _resolve_run(self, base: RunProfile, overrides: ProfileOverrides) -> RunProfile:
        raw_run = base.model_dump()
        mode_override = overrides.run_mode
        if overrides.fraction is not None and mode_override is None:
            mode_override = RunMode.BATCH_FRACTION
        cli_overrides = _without_none(
            {
                "mode": mode_override,
                "fraction": overrides.fraction,
            }
        )
        return RunProfile.model_validate(_deep_merge(raw_run, cli_overrides))

    def _load_yaml(self, config_path: Path) -> dict[str, Any]:
        if not config_path.exists():
            msg = f"Config file not found: {config_path}"
            raise FileNotFoundError(msg)
        with config_path.open("r", encoding="utf-8") as f:
            raw_config = safe_load(f) or {}
        if not isinstance(raw_config, dict):
            msg = f"Profile config must be a mapping: {config_path}"
            raise TypeError(msg)
        return raw_config

    def _resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else self.repo_root / path

    def _resolve_dataflow_path(self, name: str) -> Path:
        dataflow_path = Path(name)
        if dataflow_path.is_absolute():
            return dataflow_path
        return self.dataflow_dir / dataflow_path


def _without_none(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(base_value, value)
        else:
            merged[key] = value
    return merged
