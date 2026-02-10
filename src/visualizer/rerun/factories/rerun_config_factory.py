import rerun.blueprint as rrb

from visualizer.rerun.modules.features_module import FeaturesModule
from visualizer.rerun.modules.image_module import ImageModule
from visualizer.rerun.rerun_viz_config import VisualizerConfig
from visualizer.rerun.rerun_vizualizer import RerunVizualizer


class RerunConfigFactory:
    """Rerun config factory."""

    @staticmethod
    def from_config(config: VisualizerConfig) -> RerunVizualizer:
        """Create a rerun config from a visualizer config."""
        rerun_vizualizer = RerunVizualizer(app_name=config.app_name)

        if config.image_stream_enabled:
            for image_stream_name, image_stream_path in config.image_streams.items():
                rerun_vizualizer.add_bluepint_part(
                    rrb.Spatial2DView(name=image_stream_name, origin=image_stream_path)
                )
                rerun_vizualizer.add_module(
                    ImageModule(
                        property_name=image_stream_name,
                        entity_path=image_stream_path,
                    )
                )
        if config.features_stream_enabled:
            for feature_stream_name, feature_stream_path in config.features_streams.items():
                rerun_vizualizer.add_bluepint_part(
                    rrb.Spatial2DView(name=feature_stream_name, origin=feature_stream_path)
                )
                rerun_vizualizer.add_module(
                    FeaturesModule(
                        property_name=feature_stream_name,
                        entity_path=feature_stream_path,
                    )
                )

        return rerun_vizualizer
