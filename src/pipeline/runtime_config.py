from __future__ import annotations

import json
import os
from enum import StrEnum
from pathlib import Path  # noqa: TC003 - pydantic resolves model field types at runtime.
from typing import TYPE_CHECKING, Any, Self, cast

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Iterator

PIPELINE_NODE_CONFIG_ENV = "PIPELINE_NODE_CONFIG"
DORA_NODE_ID_ENV = "DORA_NODE_ID"


class NodePipelineRuntimeConfig(BaseModel):
    """Runtime config embedded into the materialized dataflow for one node."""

    node_id: str
    emit_ready_status: bool = True
    repo_root: Path | None = None
    profile: str | None = None
    dataflow_name: str | None = None
    dataset_name: str | None = None
    dataset_root: Path | None = None
    dataset_cache_path: Path | None = None
    dataset_rig_path: Path | None = None

    @classmethod
    def from_env_variable(
        cls,
        env_var: str = PIPELINE_NODE_CONFIG_ENV,
        *,
        default_node_id: str = "unknown",
    ) -> Self:
        """
        Load node runtime config from an env variable.

        The env value may be either the node config itself or a nested object
        keyed by the current Dora node id.
        """
        config_data = cls._load_env_config_data(env_var=env_var, default_node_id=default_node_id)
        return cls.model_validate(config_data)

    @classmethod
    def _load_env_config_data(
        cls,
        *,
        env_var: str,
        default_node_id: str,
    ) -> dict[str, Any]:
        node_id = os.getenv(DORA_NODE_ID_ENV) or default_node_id
        raw_config = os.getenv(env_var)
        if not raw_config:
            return {"node_id": node_id}

        config_data = _extract_node_config(
            raw_config,
            env_var=env_var,
            node_id=node_id,
            field_names=set(cls.model_fields),
        )
        if "node_id" not in config_data:
            config_data = {**config_data, "node_id": node_id}
        return config_data


class ControlNodeRuntimeConfig(NodePipelineRuntimeConfig):
    """Runtime config for the control node."""

    emit_ready_status: bool = False
    expected_ready_nodes: list[str] = Field(default_factory=list)
    ready_inputs: dict[str, str] = Field(default_factory=dict)
    run_mode: str = "manual"
    fraction: float | None = None
    autostart_after_ready: bool = False
    stop_after_dataset_done: bool = False


class RerunNodeSink(StrEnum):
    """Where the rerun node should send recording data."""

    APP = "app"
    FILE = "file"
    BOTH = "both"
    OFF = "off"


class RerunNodeRuntimeConfig(NodePipelineRuntimeConfig):
    """Runtime config for the rerun visualization node."""

    sink: RerunNodeSink = RerunNodeSink.APP
    output: Path | None = None

    @property
    def enabled(self) -> bool:
        """Whether the rerun node should log incoming frames."""
        return self.sink != RerunNodeSink.OFF

    @property
    def spawn_viewer(self) -> bool:
        """Whether the rerun viewer should be spawned for this run."""
        return self.sink in {RerunNodeSink.APP, RerunNodeSink.BOTH}

    @property
    def save_recording(self) -> bool:
        """Whether the rerun recording should be written to an RRD file."""
        return self.sink in {RerunNodeSink.FILE, RerunNodeSink.BOTH}


class DatasetNodeConfig(NodePipelineRuntimeConfig):
    """Runtime config for the dataset node."""

    dataset_name: str

    @classmethod
    def from_env_variable(
        cls,
        env_var: str = PIPELINE_NODE_CONFIG_ENV,
        *,
        default_node_id: str = "dataset",
    ) -> Self:
        """
        Load dataset node runtime config and require a dataset selector.

        DATASET_NAME and REPO_ROOT remain supported as legacy fallbacks for
        manual node runs outside a resolved runtime dataflow.
        """
        config_data = cls._load_env_config_data(env_var=env_var, default_node_id=default_node_id)
        dataset_name = os.getenv("DATASET_NAME")
        repo_root = os.getenv("REPO_ROOT")
        if config_data.get("dataset_name") is None and dataset_name is not None:
            config_data = {**config_data, "dataset_name": dataset_name}
        if config_data.get("repo_root") is None and repo_root is not None:
            config_data = {**config_data, "repo_root": repo_root}
        return cls.model_validate(config_data)


def load_node_config_from_env[T: NodePipelineRuntimeConfig](
    config_type: type[T],
    *,
    default_node_id: str = "unknown",
) -> T:
    """Load node runtime config from PIPELINE_NODE_CONFIG with legacy fallbacks."""
    return config_type.from_env_variable(default_node_id=default_node_id)


def _extract_node_config(
    raw_config: object,
    *,
    env_var: str,
    node_id: str,
    field_names: set[str],
) -> dict[str, Any]:
    data = _decode_mapping(raw_config, env_var=env_var)
    if _looks_like_node_config(data, field_names):
        return data

    for nested_config in _nested_config_candidates(data, env_var=env_var, node_id=node_id):
        return _extract_node_config(
            nested_config,
            env_var=env_var,
            node_id=node_id,
            field_names=field_names,
        )

    msg = f"{env_var} does not contain runtime config for node '{node_id}'"
    raise ValueError(msg)


def _nested_config_candidates(
    data: dict[str, Any],
    *,
    env_var: str,
    node_id: str,
) -> Iterator[object]:
    env = data.get("env")
    if isinstance(env, dict) and env_var in env:
        yield env[env_var]

    for key in (env_var, "node_config", "pipeline_node_config", "config"):
        nested = data.get(key)
        if nested is not None:
            yield nested

    node_configs = data.get("node_configs")
    if isinstance(node_configs, dict) and node_id in node_configs:
        yield node_configs[node_id]

    nodes = data.get("nodes")
    if isinstance(nodes, dict) and node_id in nodes:
        yield nodes[node_id]
    if isinstance(nodes, list):
        for node in nodes:
            if isinstance(node, dict) and node.get("id") == node_id:
                yield node

    nested_config = data.get(node_id)
    if nested_config is not None:
        yield nested_config


def _decode_mapping(raw_config: object, *, env_var: str) -> dict[str, Any]:
    decoded_config = raw_config
    if isinstance(decoded_config, str):
        decoded_config = json.loads(decoded_config)
    if not isinstance(decoded_config, dict):
        msg = f"{env_var} must contain a JSON object"
        raise TypeError(msg)
    return cast("dict[str, Any]", decoded_config)


def _looks_like_node_config(data: dict[str, Any], field_names: set[str]) -> bool:
    return any(field_name in data for field_name in field_names)
