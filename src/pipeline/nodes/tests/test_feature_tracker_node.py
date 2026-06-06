from pathlib import Path

import pytest

from pipeline.nodes.feature_tracker_node import load_dataset_rig_from_env


class TestFeatureTrackerNode:
    """Feature tracker node runtime config tests."""

    def test_load_dataset_rig_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dataset rig should load from env path relative to REPO_ROOT."""
        monkeypatch.setenv("REPO_ROOT", str(Path.cwd()))
        monkeypatch.setenv("DATASET_RIG_PATH", "config/dataset_rig/euroc.yaml")

        rig = load_dataset_rig_from_env()

        assert rig.name == "euroc"
        assert rig.cam0.resolution == (752, 480)
        assert rig.imu0.rate_hz == 200.0

    def test_load_dataset_rig_from_env_requires_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dataset rig path should be explicit in node env."""
        monkeypatch.delenv("DATASET_RIG_PATH", raising=False)

        with pytest.raises(ValueError, match="DATASET_RIG_PATH is not set"):
            load_dataset_rig_from_env()
