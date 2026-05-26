from pathlib import Path

import pytest

from visualizer.rerun.loaders import RerunConfigLoader
from visualizer.rerun.schemas import ModuleType, ViewType


class TestRerunConfigLoader:
    """Unit tests for RerunConfigLoader."""

    def test_from_path_merges_included_views_and_root_colors(self, tmp_path: Path) -> None:
        """Included view fragments should be appended before root-level views."""
        include_path = tmp_path / "views" / "image.yaml"
        include_path.parent.mkdir()
        include_path.write_text(
            """
colors:
  x_axis_default: [1, 2, 3]
views:
  - name: Included Image
    type: Spatial2D
    origin: /image
    streams:
      - id: left
        module: image
        entity: .
""",
        )
        root_path = tmp_path / "rerun.yaml"
        root_path.write_text(
            """
includes:
  - views/image.yaml
colors:
  y_axis_default: [4, 5, 6]
views:
  - name: Root Metric
    type: TimeSeries
    origin: /metric
    streams:
      - id: value
        module: plot_scalar
        entity: .
        options:
          label: value
""",
        )

        config = RerunConfigLoader.from_path(root_path)

        assert config.colors.x_axis_default == [1, 2, 3]
        assert config.colors.y_axis_default == [4, 5, 6]
        assert [view.name for view in config.views] == ["Included Image", "Root Metric"]
        assert config.views[0].type == ViewType.SPATIAL_2D
        assert config.views[0].streams[0].module == ModuleType.IMAGE
        assert config.views[1].type == ViewType.TIME_SERIES
        assert config.views[1].streams[0].module == ModuleType.PLOT_SCALAR

    def test_from_path_detects_include_cycles(self, tmp_path: Path) -> None:
        """Recursive includes should fail with a clear error."""
        first_path = tmp_path / "first.yaml"
        second_path = tmp_path / "second.yaml"
        first_path.write_text("includes: [second.yaml]\n")
        second_path.write_text("includes: [first.yaml]\n")

        with pytest.raises(ValueError, match="Cyclic rerun config include detected"):
            RerunConfigLoader.from_path(first_path)
