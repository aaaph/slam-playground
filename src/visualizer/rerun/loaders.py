import os
from pathlib import Path

from yaml import safe_load

from visualizer.rerun.schemas import RerunConfigSchema


class RerunConfigLoader:
    """Rerun config loader."""

    @staticmethod
    def from_env_path(path: str = "VISUALIZE_CONFIG") -> RerunConfigSchema:
        """Load the rerun config from the environment path."""
        config_path = os.getenv(path, "config/rerun_view_config.yaml")
        config_path = Path(config_path)
        with Path.open(config_path, "r") as f:
            return RerunConfigSchema(**safe_load(f))
