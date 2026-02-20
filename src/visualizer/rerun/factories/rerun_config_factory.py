import numpy as np
import rerun.blueprint as rrb

from visualizer.rerun.modules.features_module import FeaturesModule
from visualizer.rerun.modules.image_module import ImageModule
from visualizer.rerun.modules.imu_module import ImuModule
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
            resolution = config.image_resolution
            for feature_stream_name, feature_stream_path in config.features_streams.items():
                rerun_vizualizer.add_bluepint_part(
                    rrb.Spatial2DView(
                        name=feature_stream_name,
                        origin=feature_stream_path,
                        visual_bounds=rrb.VisualBounds2D(
                            x_range=np.array([0, resolution[0]]),  # ty: ignore
                            y_range=np.array([0, resolution[1]]),  # ty: ignore
                        ),
                    )
                )
                rerun_vizualizer.add_module(
                    FeaturesModule(
                        property_name=feature_stream_name,
                        entity_path=feature_stream_path,
                    )
                )

        if config.imu_stream_enabled:
            contents = [
                rrb.TimeSeriesView(
                    name=f"{field}_stream",
                    origin=f"{config.imu_path}/{field}",
                )
                for field in config.imu_streams
            ]
            rerun_vizualizer.add_bluepint_part(
                rrb.Vertical(
                    name="imu_stream",
                    contents=contents,
                ),
            )
            rerun_vizualizer.add_module(
                ImuModule(
                    entity_path=config.imu_path,
                    fields=config.imu_streams,
                )
            )
        return rerun_vizualizer
