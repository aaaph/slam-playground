import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from pipeline.nodes.base import PipelineNode
from pipeline.runtime_config import (
    DORA_NODE_ID_ENV,
    PIPELINE_NODE_CONFIG_ENV,
    ControlNodeConfig,
    DatasetNodeConfig,
)


class TestPipelineNodeEnvMixin:
    """Tests for shared node env helpers."""

    def test_required_env_returns_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Required env helper should return present values."""
        monkeypatch.setenv("SOME_NODE_ENV", "value")

        assert PipelineNode.required_env("SOME_NODE_ENV") == "value"

    def test_required_env_raises_clear_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Required env helper should fail clearly on missing values."""
        monkeypatch.delenv("SOME_NODE_ENV", raising=False)

        with pytest.raises(ValueError, match="SOME_NODE_ENV is not set"):
            PipelineNode.required_env("SOME_NODE_ENV")

    def test_repo_root_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Repo root should come from REPO_ROOT when set."""
        monkeypatch.setenv("REPO_ROOT", str(Path.cwd()))

        assert PipelineNode.repo_root_from_env() == Path.cwd().resolve()

    def test_runtime_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Runtime config should load from embedded node config JSON."""
        monkeypatch.setenv(
            PIPELINE_NODE_CONFIG_ENV,
            json.dumps(
                {
                    "node_id": "dataset",
                    "emit_ready_status": True,
                    "repo_root": str(Path.cwd()),
                    "dataset_rig_path": "config/dataset_rig/euroc.yaml",
                }
            ),
        )

        config = PipelineNode.runtime_config()

        assert config.node_id == "dataset"
        assert config.emit_ready_status is True
        assert config.repo_root == Path.cwd()
        assert config.dataset_rig_path == Path("config/dataset_rig/euroc.yaml")

    def test_runtime_config_from_env_nested_by_node_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Runtime config should load from a nested object keyed by Dora node id."""
        monkeypatch.setenv(DORA_NODE_ID_ENV, "dataset")
        monkeypatch.setenv(
            PIPELINE_NODE_CONFIG_ENV,
            json.dumps(
                {
                    "dataset": {
                        "emit_ready_status": False,
                        "dataset_name": "euroc_mh_01",
                    }
                }
            ),
        )

        config = PipelineNode.runtime_config()

        assert config.node_id == "dataset"
        assert config.emit_ready_status is False
        assert config.dataset_name == "euroc_mh_01"

    def test_runtime_config_from_env_nested_runtime_dataflow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Runtime config should load from a nested runtime dataflow object."""
        monkeypatch.setenv(DORA_NODE_ID_ENV, "control")
        monkeypatch.setenv(
            PIPELINE_NODE_CONFIG_ENV,
            json.dumps(
                {
                    "nodes": [
                        {
                            "id": "control",
                            "env": {
                                PIPELINE_NODE_CONFIG_ENV: {
                                    "emit_ready_status": False,
                                    "expected_ready_nodes": ["dataset", "rerun"],
                                    "run_mode": "batch_fraction",
                                    "fraction": 0.05,
                                }
                            },
                        }
                    ]
                }
            ),
        )

        config = PipelineNode.runtime_config_as(ControlNodeConfig)

        assert config.node_id == "control"
        assert config.emit_ready_status is False
        assert config.expected_ready_nodes == ["dataset", "rerun"]
        assert config.run_mode == "batch_fraction"
        assert config.fraction == 0.05

    def test_dataset_node_config_requires_dataset_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dataset node config should require a dataset selector."""
        monkeypatch.delenv("DATASET_NAME", raising=False)
        monkeypatch.setenv(PIPELINE_NODE_CONFIG_ENV, json.dumps({"node_id": "dataset"}))

        with pytest.raises(ValidationError, match="dataset_name"):
            PipelineNode.runtime_config_as(DatasetNodeConfig)

    def test_dataset_node_config_uses_legacy_dataset_name_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dataset node config should keep DATASET_NAME as a legacy fallback."""
        monkeypatch.setenv("DATASET_NAME", "euroc_mh_01")
        monkeypatch.setenv("REPO_ROOT", str(Path.cwd()))

        config = PipelineNode.runtime_config_as(DatasetNodeConfig)

        assert config.node_id == "PipelineNode"
        assert config.dataset_name == "euroc_mh_01"
        assert config.repo_root == Path.cwd()

    def test_repo_root_from_runtime_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Repo root should prefer runtime config when available."""
        monkeypatch.setenv("REPO_ROOT", "/not/used")
        monkeypatch.setenv(
            PIPELINE_NODE_CONFIG_ENV,
            json.dumps({"node_id": "dataset", "repo_root": str(Path.cwd())}),
        )

        assert PipelineNode.repo_root_from_env() == Path.cwd().resolve()

    def test_load_dataset_rig_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dataset rig should load from DATASET_RIG_PATH relative to REPO_ROOT."""
        monkeypatch.setenv("REPO_ROOT", str(Path.cwd()))
        monkeypatch.setenv("DATASET_RIG_PATH", "config/dataset_rig/euroc.yaml")

        rig = PipelineNode.load_dataset_rig_from_env()

        assert rig.name == "euroc"
        assert rig.cam0.resolution == (752, 480)

    def test_load_dataset_rig_from_runtime_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dataset rig should prefer runtime config when available."""
        monkeypatch.setenv(
            PIPELINE_NODE_CONFIG_ENV,
            json.dumps(
                {
                    "node_id": "dataset",
                    "repo_root": str(Path.cwd()),
                    "dataset_rig_path": "config/dataset_rig/euroc.yaml",
                }
            ),
        )

        rig = PipelineNode.load_dataset_rig_from_env()

        assert rig.name == "euroc"
        assert rig.cam0.resolution == (752, 480)
