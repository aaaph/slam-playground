from pathlib import Path

import pytest

from pipeline.nodes.base import PipelineNode


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

    def test_load_dataset_rig_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dataset rig should load from DATASET_RIG_PATH relative to REPO_ROOT."""
        monkeypatch.setenv("REPO_ROOT", str(Path.cwd()))
        monkeypatch.setenv("DATASET_RIG_PATH", "config/dataset_rig/euroc.yaml")

        rig = PipelineNode.load_dataset_rig_from_env()

        assert rig.name == "euroc"
        assert rig.cam0.resolution == (752, 480)
