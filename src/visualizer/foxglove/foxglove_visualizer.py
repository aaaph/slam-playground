from collections.abc import Callable, Generator
from functools import wraps
from pathlib import Path

import foxglove
from foxglove.channels import FrameTransformsChannel
from foxglove.schemas import FrameTransform, FrameTransforms
from foxglove.websocket import Capability

from logger import spawn_logger
from visualizer.foxglove.foxglove_listener import FoxgloveServerListener
from visualizer.foxglove.modules.abc_module import IVizModule
from visualizer.visualizer_context import VisualizerContext


def coroutine(
    func: Callable[..., Generator[None, VisualizerContext]],
) -> Callable[..., Generator[None, VisualizerContext]]:
    """Coroutine decorator."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Generator[None, VisualizerContext]:  # noqa: ANN002, ANN003
        """Wrap the coroutine."""
        gen = func(*args, **kwargs)
        next(gen)
        return gen

    return wrapper


class FoxgloveVisualizer[TContext]:
    """Foxglove visualizer."""

    def __init__(self, *, wait_for_client: bool = False) -> None:
        """Initialize the Foxglove visualizer."""
        self.logger = spawn_logger(app="foxglove_visualizer")
        self.modules: list[IVizModule] = []
        self.wait_for_client = wait_for_client

    def add_module(self, module: IVizModule) -> None:
        """Add a module to the visualizer."""
        module.setup()
        self.modules.append(module)

    @coroutine
    def websocket_viz_gen(self) -> Generator[None, TContext]:
        """Run the websocket foxglove-sdk server visualizer. The WS config is default."""
        logger = spawn_logger(app="foxglove_websocket_viz")
        listener = FoxgloveServerListener()
        if hasattr(foxglove, "start_server"):
            server = foxglove.start_server(
                server_listener=listener,
                capabilities=[Capability.ClientPublish],
                supported_encodings=["json"],
            )
        else:
            raise AttributeError("start_server is not available in the foxglove module")

        if self.wait_for_client:
            logger.info("Waiting for client to connect...")
            listener.client_connected_event.wait()
            logger.info("Client connected")

        frame_transforms_channel = FrameTransformsChannel("/tf")
        logger.info("Foxglove visualizer started")
        try:
            while True:
                incoming_data: TContext = yield
                if incoming_data is None:
                    break
                transforms: list[FrameTransform] = []
                for module in self.modules:
                    transform_list = module.process(incoming_data)
                    transforms.extend(transform_list)
                frame_transforms_message = FrameTransforms(transforms=transforms)
                frame_transforms_channel.log(frame_transforms_message)
        finally:
            logger.info("Foxglove visualizer stopping")
            server.stop()

    @coroutine
    def mcap_gen(self, mcap_file_name: str) -> Generator[None, TContext]:
        """Run the mcap foxglove-sdk writer. The mcap file is saved in the same directory as the script."""
        logger = spawn_logger(app="foxglove_mcap_viz")

        mcap_file = f"{mcap_file_name}.mcap"
        mcap_path = Path(__file__).parent.parent.parent.parent / mcap_file
        writer = foxglove.open_mcap(mcap_path, allow_overwrite=True)

        frame_transforms_channel = FrameTransformsChannel("/tf")
        logger.info("Foxglove writer started")
        try:
            while True:
                incoming_data = yield
                if incoming_data is None:
                    break
                transforms: list[FrameTransform] = []
                for module in self.modules:
                    transform_list = module.process(incoming_data)
                    transforms.extend(transform_list)
                frame_transforms_message = FrameTransforms(transforms=transforms)
                frame_transforms_channel.log(frame_transforms_message)
        finally:
            logger.info("Foxglove visualizer stopping")
            writer.close()
