import cv2
from foxglove.channels import RawImageChannel
from foxglove.schemas import FrameTransform, Quaternion, RawImage, Vector3

from logger import spawn_logger
from visualizer.foxglove.modules.abc_module import IVizModule
from visualizer.visualizer_context import VisualizerContext


class ImageModule(IVizModule):
    """Foxglove Image module."""

    def __init__(self) -> None:
        """Initialize the Foxglove Image module."""
        self.logger = spawn_logger(app="foxglove_image_module")

    def setup(self) -> None:
        """Set up the Foxglove Image module."""
        self.image_channel = RawImageChannel("/frame")
        # self.camera_calibration_channel = None

    def process(self, context: VisualizerContext) -> list[FrameTransform]:
        """Process the Foxglove Image module."""
        if context.frame is None:
            self.logger.warning("Frame data not found")
            return []
        image = context.frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.tobytes()
        raw_image_message = RawImage(
            frame_id="frame",
            data=image,
            step=752 * 3,
            width=752,
            height=480,
            encoding="rgb8",
        )
        self.image_channel.log(raw_image_message)

        """camera_calibration_message = CameraCalibration(
            frame_id="frame",
            width=752,
            height=480,
            distortion_model="plumb_bob",
            # Foxglove schemas expect plain (flattened) lists, not NumPy arrays.
            K=self.camera_model.stereo_k.flatten().tolist(),
            D=self.camera_model.distortion_coefficients.flatten().tolist(),
            R=self.camera_model.r1.flatten().tolist(),
            P=self.camera_model.p1.flatten().tolist(),
        )
        self.camera_calibration_channel.log(camera_calibration_message)"""

        return [
            FrameTransform(
                parent_frame_id="cam0",
                child_frame_id="frame",
                translation=Vector3(x=0, y=0, z=0),
                rotation=Quaternion(x=0, y=0, z=0, w=1),
            )
        ]
