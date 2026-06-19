from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, Field, field_validator, model_validator
from yaml import safe_load

from dataset.manifest import DatasetManifest, DatasetRigConfig  # noqa: TC001 - pydantic model annotations.
from dataset.registry import DatasetRegistry
from pipeline.runtime_config import (
    DORA_NODE_ID_ENV,
    PIPELINE_NODE_CONFIG_ENV,
    ControlNodeRuntimeConfig,
    NodePipelineRuntimeConfig,
    RerunNodeRuntimeConfig,
    RerunNodeSink,
)

CONTROL_NODE_ID = "control"
RERUN_NODE_ID = "rerun"
STATUS_OUTPUT_ID = "status"
STARTUP_TICK_INPUT_ID = "startup_tick"
STARTUP_TICK_SOURCE = "dora/timer/millis/100"


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
    runtime_dataflow: dict[str, Any] = Field(default_factory=dict)


class DataflowSelector(BaseModel):
    """Dataflow selector declared in a composite profile."""

    template: str
    build: bool = False


class ParsedDataflowInput(BaseModel):
    """A normalized dataflow input edge."""

    name: str
    source: str | None = None
    queue_size: int | None = None


class ParsedDataflowNode(BaseModel):
    """A normalized dataflow node descriptor."""

    id: str
    path: Path | None = None
    inputs: list[ParsedDataflowInput] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    env: dict[str, Any] = Field(default_factory=dict)
    emits_status: bool = False


class ParsedDataflowStatusRoute(BaseModel):
    """A status output connected into a dataflow node input."""

    target_node: str
    input: str
    source_node: str
    source_output: str


class ParsedDataflow(BaseModel):
    """Normalized dataflow snapshot for orchestration decisions."""

    nodes: list[ParsedDataflowNode] = Field(default_factory=list)
    status_output_nodes: list[str] = Field(default_factory=list)
    status_routes: list[ParsedDataflowStatusRoute] = Field(default_factory=list)


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


@dataclass(frozen=True)
class _NodeConfigResolutionContext:
    profile: str | None
    dataset: DatasetManifest
    dataflow: DataflowProfile
    parsed_dataflow: ParsedDataflow
    visualization: VisualizationProfile
    run: RunProfile


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
        parsed_dataflow = self.load_parsed_dataflow(dataflow.template)
        visualization = self._resolve_visualization(composite.visualization, overrides)
        run = self._resolve_run(composite.run, overrides)
        node_config_context = _NodeConfigResolutionContext(
            profile=composite.name,
            dataset=dataset,
            dataflow=dataflow,
            parsed_dataflow=parsed_dataflow,
            visualization=visualization,
            run=run,
        )
        node_configs = self._resolve_node_configs(node_config_context)
        runtime_dataflow = self.load_runtime_dataflow(
            dataflow.template,
            parsed_dataflow=parsed_dataflow,
            node_configs=node_configs,
        )
        dataflow = dataflow.model_copy(update={"runtime_dataflow": runtime_dataflow})

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
        raw_profile = self._load_yaml(self.profile_dir / f"{name}.yaml", kind="Profile")
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

    def load_parsed_dataflow(self, template: Path) -> ParsedDataflow:
        """Load and normalize a resolved Dora dataflow file."""
        dataflow_path = self._resolve_path(template)
        raw_dataflow = self._load_yaml(dataflow_path, kind="Dataflow")
        raw_nodes = raw_dataflow.get("nodes", [])
        if not isinstance(raw_nodes, list):
            msg = f"Dataflow nodes must be a list: {dataflow_path}"
            raise TypeError(msg)

        nodes = [self._parse_dataflow_node(raw_node, dataflow_path) for raw_node in raw_nodes]
        status_output_nodes = [node.id for node in nodes if node.id != CONTROL_NODE_ID]
        status_routes = self._runtime_status_routes(status_output_nodes)
        return ParsedDataflow(
            nodes=nodes,
            status_output_nodes=status_output_nodes,
            status_routes=status_routes,
        )

    def load_runtime_dataflow(
        self,
        template: Path,
        *,
        parsed_dataflow: ParsedDataflow,
        node_configs: dict[str, NodePipelineRuntimeConfig],
    ) -> dict[str, Any]:
        """Load a dataflow template and inject runtime node config env values."""
        dataflow_path = self._resolve_path(template)
        raw_dataflow = self._load_yaml(dataflow_path, kind="Dataflow")
        raw_nodes = raw_dataflow.get("nodes", [])
        if not isinstance(raw_nodes, list):
            msg = f"Dataflow nodes must be a list: {dataflow_path}"
            raise TypeError(msg)

        parsed_nodes_by_id = {node.id: node for node in parsed_dataflow.nodes}
        for raw_node in raw_nodes:
            if not isinstance(raw_node, dict):
                msg = f"Dataflow node must be a mapping: {dataflow_path}"
                raise TypeError(msg)
            raw_node_config = cast("dict[str, Any]", raw_node)
            node_id = raw_node_config.get("id")
            if not isinstance(node_id, str):
                msg = f"Dataflow node id must be a string: {dataflow_path}"
                raise TypeError(msg)
            raw_env = raw_node_config.setdefault("env", {})
            if not isinstance(raw_env, dict):
                msg = f"Dataflow node env must be a mapping: {node_id}"
                raise TypeError(msg)
            if node_id not in parsed_nodes_by_id:
                msg = f"Parsed dataflow is missing node: {node_id}"
                raise ValueError(msg)
            runtime_config = node_configs.get(node_id)
            if runtime_config is None:
                msg = f"Runtime node config is missing node: {node_id}"
                raise ValueError(msg)
            self._inject_runtime_status_edges(
                node_id=node_id,
                raw_node_config=raw_node_config,
                parsed_dataflow=parsed_dataflow,
            )
            raw_env[PIPELINE_NODE_CONFIG_ENV] = runtime_config.model_dump_json()
            raw_env[DORA_NODE_ID_ENV] = node_id

        return raw_dataflow

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

    def _resolve_node_configs(self, context: _NodeConfigResolutionContext) -> dict[str, NodePipelineRuntimeConfig]:
        return {
            node.id: self._resolve_node_config(
                node=node,
                context=context,
            )
            for node in context.parsed_dataflow.nodes
        }

    def _resolve_node_config(
        self,
        *,
        node: ParsedDataflowNode,
        context: _NodeConfigResolutionContext,
    ) -> NodePipelineRuntimeConfig:
        emit_ready_status = node.id in context.parsed_dataflow.status_output_nodes
        common_config = {
            "node_id": node.id,
            "emit_ready_status": emit_ready_status,
            "repo_root": self.repo_root,
            "profile": context.profile,
            "dataflow_name": context.dataflow.name,
            "dataset_name": context.dataset.name,
            "dataset_root": context.dataset.root,
            "dataset_cache_path": context.dataset.cache or context.dataset.root / "cache",
            "dataset_rig_path": context.dataset.rig,
        }
        if node.id == "control":
            return ControlNodeRuntimeConfig(
                **common_config,
                expected_ready_nodes=context.parsed_dataflow.status_output_nodes,
                ready_inputs={route.input: route.source_node for route in context.parsed_dataflow.status_routes},
                run_mode=context.run.mode.value,
                fraction=context.run.fraction,
                autostart_after_ready=context.run.autostart_after_ready,
                stop_after_dataset_done=context.run.stop_after_dataset_done,
            )
        if node.id == RERUN_NODE_ID:
            return RerunNodeRuntimeConfig(
                **common_config,
                sink=RerunNodeSink(context.visualization.sink.value),
                output=context.visualization.output,
            )
        return NodePipelineRuntimeConfig(**common_config)

    def _load_yaml(self, config_path: Path, *, kind: str) -> dict[str, Any]:
        if not config_path.exists():
            msg = f"Config file not found: {config_path}"
            raise FileNotFoundError(msg)
        with config_path.open("r", encoding="utf-8") as f:
            raw_config = safe_load(f) or {}
        if not isinstance(raw_config, dict):
            msg = f"{kind} config must be a mapping: {config_path}"
            raise TypeError(msg)
        return raw_config

    def _parse_dataflow_node(self, raw_node: object, dataflow_path: Path) -> ParsedDataflowNode:
        if not isinstance(raw_node, dict):
            msg = f"Dataflow node must be a mapping: {dataflow_path}"
            raise TypeError(msg)
        raw_node_config = cast("dict[str, Any]", raw_node)

        node_id = raw_node_config.get("id")
        if not isinstance(node_id, str):
            msg = f"Dataflow node id must be a string: {dataflow_path}"
            raise TypeError(msg)

        raw_path = raw_node_config.get("path")
        raw_inputs = raw_node_config.get("inputs", {})
        raw_outputs = raw_node_config.get("outputs", [])
        raw_env = raw_node_config.get("env", {})

        if not isinstance(raw_inputs, dict):
            msg = f"Dataflow node inputs must be a mapping: {node_id}"
            raise TypeError(msg)
        if not isinstance(raw_outputs, list):
            msg = f"Dataflow node outputs must be a list: {node_id}"
            raise TypeError(msg)
        if not isinstance(raw_env, dict):
            msg = f"Dataflow node env must be a mapping: {node_id}"
            raise TypeError(msg)

        outputs = [str(output) for output in raw_outputs]
        return ParsedDataflowNode(
            id=node_id,
            path=Path(raw_path) if isinstance(raw_path, str) else None,
            inputs=[
                self._parse_dataflow_input(name=str(name), raw_input=raw_input)
                for name, raw_input in raw_inputs.items()
            ],
            outputs=outputs,
            env={str(key): value for key, value in raw_env.items()},
            emits_status=STATUS_OUTPUT_ID in outputs,
        )

    @staticmethod
    def _parse_dataflow_input(name: str, raw_input: object) -> ParsedDataflowInput:
        if isinstance(raw_input, str):
            return ParsedDataflowInput(name=name, source=raw_input)
        if isinstance(raw_input, dict):
            raw_input_config = cast("dict[str, Any]", raw_input)
            source = raw_input_config.get("source")
            queue_size = raw_input_config.get("queue_size")
            normalized_queue_size = (
                queue_size if isinstance(queue_size, int) and not isinstance(queue_size, bool) else None
            )
            return ParsedDataflowInput(
                name=name,
                source=str(source) if source is not None else None,
                queue_size=normalized_queue_size,
            )
        return ParsedDataflowInput(name=name)

    @staticmethod
    def _parse_status_routes(
        *,
        target_node: str,
        inputs: list[ParsedDataflowInput],
    ) -> list[ParsedDataflowStatusRoute]:
        routes: list[ParsedDataflowStatusRoute] = []
        for input_config in inputs:
            if input_config.source is None:
                continue
            source_node, separator, source_output = input_config.source.rpartition("/")
            if separator != "/" or source_output != STATUS_OUTPUT_ID:
                continue
            routes.append(
                ParsedDataflowStatusRoute(
                    target_node=target_node,
                    input=input_config.name,
                    source_node=source_node,
                    source_output=source_output,
                )
            )
        return routes

    @staticmethod
    def _runtime_status_routes(status_output_nodes: list[str]) -> list[ParsedDataflowStatusRoute]:
        return [
            ParsedDataflowStatusRoute(
                target_node=CONTROL_NODE_ID,
                input=f"{node_id}_{STATUS_OUTPUT_ID}",
                source_node=node_id,
                source_output=STATUS_OUTPUT_ID,
            )
            for node_id in status_output_nodes
        ]

    def _inject_runtime_status_edges(
        self,
        *,
        node_id: str,
        raw_node_config: dict[str, Any],
        parsed_dataflow: ParsedDataflow,
    ) -> None:
        if node_id == CONTROL_NODE_ID:
            raw_inputs = self._ensure_node_inputs(raw_node_config, node_id=node_id)
            self._ensure_input(
                raw_inputs,
                input_name=STARTUP_TICK_INPUT_ID,
                source=STARTUP_TICK_SOURCE,
                node_id=node_id,
            )
            for route in parsed_dataflow.status_routes:
                self._ensure_input(
                    raw_inputs,
                    input_name=route.input,
                    source=f"{route.source_node}/{route.source_output}",
                    node_id=node_id,
                )
            return

        if node_id in parsed_dataflow.status_output_nodes:
            raw_outputs = self._ensure_node_outputs(raw_node_config, node_id=node_id)
            if STATUS_OUTPUT_ID not in raw_outputs:
                raw_outputs.append(STATUS_OUTPUT_ID)

    @staticmethod
    def _ensure_node_inputs(raw_node_config: dict[str, Any], *, node_id: str) -> dict[str, Any]:
        raw_inputs = raw_node_config.setdefault("inputs", {})
        if not isinstance(raw_inputs, dict):
            msg = f"Dataflow node inputs must be a mapping: {node_id}"
            raise TypeError(msg)
        return cast("dict[str, Any]", raw_inputs)

    @staticmethod
    def _ensure_node_outputs(raw_node_config: dict[str, Any], *, node_id: str) -> list[Any]:
        raw_outputs = raw_node_config.setdefault("outputs", [])
        if not isinstance(raw_outputs, list):
            msg = f"Dataflow node outputs must be a list: {node_id}"
            raise TypeError(msg)
        return raw_outputs

    @staticmethod
    def _ensure_input(
        raw_inputs: dict[str, Any],
        *,
        input_name: str,
        source: str,
        node_id: str,
    ) -> None:
        existing_source = raw_inputs.get(input_name)
        if existing_source == source:
            return
        if existing_source is not None:
            msg = f"Dataflow node input '{node_id}.{input_name}' already exists with source {existing_source}"
            raise ValueError(msg)
        raw_inputs[input_name] = source

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
