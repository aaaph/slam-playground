import numpy as np
import rerun.blueprint as rrb

from visualizer.rerun.modules.features_module import FeaturesModule, FeaturesModuleOptions
from visualizer.rerun.modules.image_module import ImageModule
from visualizer.rerun.modules.imu_module import ImuModule
from visualizer.rerun.modules.plot_module import PlotModule
from visualizer.rerun.modules.pointcloud_module import PointcloudModule
from visualizer.rerun.modules.pose_module import PoseModule
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
            for feature_stream_name in config.features_streams:
                feat_stream_config = config.feature_stream(feature_stream_name)
                rerun_vizualizer.add_bluepint_part(
                    rrb.Spatial2DView(
                        name=feature_stream_name,
                        origin=feat_stream_config.path,
                        visual_bounds=rrb.VisualBounds2D(
                            x_range=np.array([0, resolution[0]]),
                            y_range=np.array([0, resolution[1]]),
                        ),
                    )
                )
                rerun_vizualizer.add_module(
                    FeaturesModule(
                        property_name=feature_stream_name,
                        entity_path=feat_stream_config.path,
                        options=FeaturesModuleOptions(
                            show_stereo_baseline=feat_stream_config.show_stereo_baseline,
                        ),
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

        if config.pose_stream_enabled:
            rerun_vizualizer.add_bluepint_part(
                rrb.Spatial3DView(
                    name="3d_view",
                    origin="/",
                    background=rrb.Background(color=np.array([0.0, 0.0, 0.0])),
                    line_grid=rrb.LineGrid3D(stroke_width=1.5),
                )
            )
            for pose_stream_name, pose_stream_path in config.pose_streams.items():
                rerun_vizualizer.add_module(
                    PoseModule(property_name=pose_stream_name, entity_path=pose_stream_path)
                )
        rerun_vizualizer.add_module(
            PointcloudModule(
                entity_path="/",
                property_name="points",
            )
        )
        if config.plot_stream_enabled:
            rerun_vizualizer.add_bluepint_part(
                rrb.TimeSeriesView(name="plot_stream", origin="/world/odom/base_link/metrics/")
            )
            for plot_stream_name, plot_stream_path in config.plot_streams.items():
                rerun_vizualizer.add_module(
                    PlotModule(property_name=plot_stream_name, entity_path=plot_stream_path)
                )
        return rerun_vizualizer
