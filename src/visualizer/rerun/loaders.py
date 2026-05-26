import os
from pathlib import Path
from typing import Any

from yaml import safe_load

from visualizer.rerun.schemas import RerunConfigSchema


class RerunConfigLoader:
    """Rerun config loader."""

    @staticmethod
    def from_env_path(path: str = "VISUALIZE_CONFIG") -> RerunConfigSchema:
        """Load the rerun config from the environment path."""
        config_path = os.getenv(path, "config/rerun_view_config.yaml")
        return RerunConfigLoader.from_path(Path(config_path))

    @staticmethod
    def from_path(config_path: Path) -> RerunConfigSchema:
        """Load the rerun config from a YAML file and its includes."""
        raw_config = RerunConfigLoader._load_raw_config(config_path.resolve(), seen=set())
        return RerunConfigSchema(**raw_config)

    @staticmethod
    def _load_raw_config(config_path: Path, seen: set[Path]) -> dict[str, Any]:
        """Load a raw config dict, recursively merging included config fragments."""
        if config_path in seen:
            msg = f"Cyclic rerun config include detected: {config_path}"
            raise ValueError(msg)
        seen.add(config_path)

        with Path.open(config_path, "r") as f:
            raw_config = safe_load(f) or {}
        if not isinstance(raw_config, dict):
            msg = f"Rerun config must be a mapping: {config_path}"
            raise TypeError(msg)

        includes = raw_config.pop("includes", [])
        if isinstance(includes, str):
            includes = [includes]
        if not isinstance(includes, list):
            msg = f"Rerun config includes must be a list: {config_path}"
            raise TypeError(msg)

        merged_config: dict[str, Any] = {}
        for include_path in includes:
            if not isinstance(include_path, str):
                msg = f"Rerun config include path must be a string: {config_path}"
                raise TypeError(msg)
            included_config_path = (config_path.parent / include_path).resolve()
            included_config = RerunConfigLoader._load_raw_config(included_config_path, seen=seen)
            merged_config = RerunConfigLoader._merge_raw_configs(merged_config, included_config)

        result = RerunConfigLoader._merge_raw_configs(merged_config, raw_config)
        seen.remove(config_path)
        return result

    @staticmethod
    def _merge_raw_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Merge two raw rerun config dicts."""
        merged = dict(base)
        for key, value in override.items():
            if key == "views":
                merged[key] = [*merged.get(key, []), *value]
            elif key == "colors":
                merged[key] = {**merged.get(key, {}), **value}
            else:
                merged[key] = value
        return merged
