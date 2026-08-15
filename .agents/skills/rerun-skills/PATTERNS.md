# Common Patterns and Complete Examples

This guide provides complete, end-to-end examples of common Rerun use cases.

## Object Detection and Tracking

Complete pipeline for detecting and tracking objects in video.

```python
import argparse
import cv2
import json
import numpy as np
import rerun as rr
from transformers import DetrImageProcessor, DetrForSegmentation


def setup_annotations(categories_path: str) -> None:
    """Set up class annotations."""
    with open(categories_path) as f:
        categories = json.load(f)

    class_descriptions = [
        rr.AnnotationInfo(id=cat["id"], label=cat["name"], color=cat["color"]) for cat in categories
    ]
    rr.log("/", rr.AnnotationContext(class_descriptions), static=True)


def detect_and_track(video_path: str) -> None:
    """Detect and track objects in video."""
    # Load video
    video_asset = rr.AssetVideo(path=video_path)
    frame_timestamps_ns = video_asset.read_frame_timestamps_nanos()
    rr.log("video", video_asset, static=True)

    # Initialize detector
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50")

    # Process video
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, bgr = cap.read()
        if not ret:
            break

        rr.set_time("frame", sequence=frame_idx)

        # Reference video frame
        rr.log("video", rr.VideoFrameReference(nanoseconds=frame_timestamps_ns[frame_idx]))

        # Run detection every 10 frames
        if frame_idx % 10 == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # Detect objects
            inputs = processor(images=rgb, return_tensors="pt")
            outputs = model(**inputs)
            results = processor.post_process_object_detection(
                outputs, threshold=0.8, target_sizes=[(bgr.shape[0], bgr.shape[1])]
            )[0]

            # Log detections
            boxes = results["boxes"].detach().cpu().numpy()
            class_ids = results["labels"].detach().cpu().numpy()

            rr.log(
                "video/detections", rr.Boxes2D(array=boxes, array_format=rr.Box2DFormat.XYXY, class_ids=class_ids)
            )

        frame_idx += 1

    cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Object detection and tracking example")
    rr.script_add_args(parser)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--categories", type=str, required=True)
    args = parser.parse_args()

    rr.script_setup(args, "object_detection")
    setup_annotations(args.categories)
    detect_and_track(args.video)
    rr.script_teardown(args)


if __name__ == "__main__":
    main()
```

## Pose Estimation with Keypoints

Face tracking with MediaPipe and keypoint visualization.

```python
import argparse
import cv2
import math
import mediapipe as mp
import rerun as rr
import rerun.blueprint as rrb
from mediapipe.tasks.python import vision


class FaceLandmarker:
    """Face landmark detection with Rerun logging."""

    def __init__(self, model_path: str, video_mode: bool = False) -> None:
        self.video_mode = video_mode

        # Initialize MediaPipe
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            running_mode=(mp.tasks.vision.RunningMode.VIDEO if video_mode else mp.tasks.vision.RunningMode.IMAGE),
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # Set up keypoint connections
        self._setup_keypoint_connections()

    def _setup_keypoint_connections(self) -> None:
        """Define face mesh structure."""
        classes = [
            mp.solutions.face_mesh.FACEMESH_LIPS,
            mp.solutions.face_mesh.FACEMESH_LEFT_EYE,
            mp.solutions.face_mesh.FACEMESH_LEFT_IRIS,
            mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW,
            mp.solutions.face_mesh.FACEMESH_RIGHT_EYE,
            mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW,
            mp.solutions.face_mesh.FACEMESH_RIGHT_IRIS,
            mp.solutions.face_mesh.FACEMESH_FACE_OVAL,
            mp.solutions.face_mesh.FACEMESH_NOSE,
        ]

        class_descriptions = [
            rr.ClassDescription(info=rr.AnnotationInfo(id=i), keypoint_connections=klass)
            for i, klass in enumerate(classes)
        ]

        rr.log("video/face", rr.AnnotationContext(class_descriptions), static=True)

    def detect_and_log(self, image: np.ndarray, timestamp_ms: int) -> None:
        """Detect face landmarks and log to Rerun."""
        height, width, _ = image.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Detect
        result = (
            self.detector.detect_for_video(mp_image, timestamp_ms)
            if self.video_mode
            else self.detector.detect(mp_image)
        )

        # Log results
        for i, landmarks in enumerate(result.face_landmarks):
            # Convert normalized coords to pixels
            keypoints = [(math.floor(lm.x * width), math.floor(lm.y * height)) for lm in landmarks]

            # Log 2D keypoints
            rr.log(
                f"video/face/{i}/keypoints",
                rr.Points2D(keypoints, radii=2, keypoint_ids=list(range(len(keypoints)))),
            )

            # Log 3D reconstruction
            rr.log(
                f"reconstruction/face/{i}",
                rr.Points3D([(lm.x, lm.y, lm.z) for lm in landmarks], keypoint_ids=list(range(len(landmarks)))),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Face tracking example")
    rr.script_add_args(parser)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    # Blueprint with 3D and 2D views
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="reconstruction", name="3D"), rrb.Spatial2DView(origin="video", name="Video")
        )
    )

    rr.script_setup(args, "face_tracking", default_blueprint=blueprint)

    # Set coordinate system
    rr.log("reconstruction", rr.ViewCoordinates.RDF, static=True)

    # Process video
    landmarker = FaceLandmarker(args.model, video_mode=True)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(frame_idx * 1000 / fps)

        rr.set_time("frame", sequence=frame_idx)
        rr.log("video/image", rr.Image(frame, color_model="BGR"))

        landmarker.detect_and_log(frame, timestamp_ms)
        frame_idx += 1

    cap.release()
    rr.script_teardown(args)


if __name__ == "__main__":
    main()
```

## 3D Point Cloud Visualization (LIDAR)

Visualize LIDAR point clouds with distance-based coloring.

```python
import argparse
import matplotlib
import numpy as np
import rerun as rr
from nuscenes import nuscenes
from pathlib import Path


def visualize_lidar(dataset_dir: Path, scene_name: str) -> None:
    """Visualize LIDAR scans from nuScenes dataset."""
    # Load dataset
    nusc = nuscenes.NuScenes(version="v1.0-mini", dataroot=dataset_dir, verbose=True)

    # Find scene
    scene = next(s for s in nusc.scene if s["name"] == scene_name)

    # Set coordinate system
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Color map for distance
    cmap = matplotlib.colormaps["turbo_r"]
    norm = matplotlib.colors.Normalize(vmin=3.0, vmax=75.0)

    # Process LIDAR scans
    first_sample = nusc.get("sample", scene["first_sample_token"])
    current_token = first_sample["data"]["LIDAR_TOP"]

    while current_token != "":
        sample_data = nusc.get("sample_data", current_token)

        # Load point cloud
        data_file = nusc.dataroot / sample_data["filename"]
        pointcloud = nuscenes.LidarPointCloud.from_file(str(data_file))
        points = pointcloud.points[:3].T  # (N, 3)

        # Color by distance
        distances = np.linalg.norm(points, axis=1)
        colors = cmap(norm(distances))

        # Log with timestamp
        rr.set_time("timestamp", timestamp=sample_data["timestamp"] * 1e-6)
        rr.log("world/lidar", rr.Points3D(points, colors=colors))

        current_token = sample_data["next"]


def main() -> None:
    parser = argparse.ArgumentParser(description="LIDAR visualization example")
    rr.script_add_args(parser)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--scene", type=str, default="scene-0061")
    args = parser.parse_args()

    rr.script_setup(args, "lidar_visualization")
    visualize_lidar(args.dataset_dir, args.scene)
    rr.script_teardown(args)


if __name__ == "__main__":
    main()
```

## Training Metrics Dashboard

Real-time training metrics visualization.

```python
import argparse
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from torch.utils.data import DataLoader


def train_with_logging(model, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> None:
    """Train model with Rerun logging."""

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        train_accs = []

        for batch_idx, (data, target) in enumerate(train_loader):
            loss, acc = train_step(model, data, target)
            train_losses.append(loss)
            train_accs.append(acc)

            # Log batch metrics
            step = epoch * len(train_loader) + batch_idx
            rr.set_time("step", sequence=step)
            rr.log("training/batch/loss", rr.Scalars(loss))
            rr.log("training/batch/accuracy", rr.Scalars(acc))

        # Validation phase
        model.eval()
        val_loss, val_acc = validate(model, val_loader)

        # Log epoch metrics
        rr.set_time("epoch", sequence=epoch)
        rr.log("training/epoch/loss", rr.Scalars(np.mean(train_losses)))
        rr.log("training/epoch/accuracy", rr.Scalars(np.mean(train_accs)))
        rr.log("validation/loss", rr.Scalars(val_loss))
        rr.log("validation/accuracy", rr.Scalars(val_acc))

        # Log learning rate
        lr = optimizer.param_groups[0]["lr"]
        rr.log("training/learning_rate", rr.Scalars(lr))

        # Log sample predictions
        log_sample_predictions(model, val_loader, epoch)


def log_sample_predictions(model, loader: DataLoader, epoch: int) -> None:
    """Log sample predictions for visualization."""
    model.eval()
    data, targets = next(iter(loader))
    predictions = model(data)

    rr.set_time("epoch", sequence=epoch)

    # Log a few samples
    for i in range(min(4, len(data))):
        # Input image
        rr.log(f"samples/{i}/image", rr.Image(data[i]))

        # Ground truth
        rr.log(f"samples/{i}/ground_truth", rr.TextDocument(f"True: {targets[i]}"))

        # Prediction
        pred_class = predictions[i].argmax()
        confidence = predictions[i].max()
        rr.log(f"samples/{i}/prediction", rr.TextDocument(f"Pred: {pred_class} ({confidence:.2%})"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Training with Rerun")
    rr.script_add_args(parser)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    # Blueprint
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            # Metrics row
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    origin="training",
                    name="Training Metrics",
                    overrides={
                        "training/epoch/loss": rr.SeriesLines.from_fields(colors=[255, 0, 0], names="Train Loss"),
                        "validation/loss": rr.SeriesLines.from_fields(colors=[0, 255, 0], names="Val Loss"),
                    },
                ),
                rrb.TimeSeriesView(origin="training/epoch/accuracy", name="Accuracy"),
                rrb.TimeSeriesView(origin="training/learning_rate", name="Learning Rate"),
            ),
            # Samples row
            rrb.Grid(
                rrb.Vertical(
                    rrb.Spatial2DView(origin="samples/0/image"),
                    rrb.TextDocumentView(origin="samples/0/ground_truth"),
                    rrb.TextDocumentView(origin="samples/0/prediction"),
                ),
                rrb.Vertical(
                    rrb.Spatial2DView(origin="samples/1/image"),
                    rrb.TextDocumentView(origin="samples/1/ground_truth"),
                    rrb.TextDocumentView(origin="samples/1/prediction"),
                ),
                rrb.Vertical(
                    rrb.Spatial2DView(origin="samples/2/image"),
                    rrb.TextDocumentView(origin="samples/2/ground_truth"),
                    rrb.TextDocumentView(origin="samples/2/prediction"),
                ),
                rrb.Vertical(
                    rrb.Spatial2DView(origin="samples/3/image"),
                    rrb.TextDocumentView(origin="samples/3/ground_truth"),
                    rrb.TextDocumentView(origin="samples/3/prediction"),
                ),
            ),
            row_shares=[1, 2],
        ),
        rrb.TimePanel(state="expanded"),
    )

    rr.script_setup(args, "training_dashboard", default_blueprint=blueprint)

    # Load data and train
    train_loader, val_loader = load_data()
    model = create_model()
    train_with_logging(model, train_loader, val_loader, args.epochs)

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
```

## Sensor Data Logging (IMU)

Log multi-axis sensor data from IMU.

```python
import argparse
import numpy as np
import rerun as rr
import rerun.blueprint as rrb


def log_imu_data(imu_file: str) -> None:
    """Log IMU sensor data."""
    # Load IMU data
    data = np.load(imu_file)

    timestamps = data["timestamps"]
    accel_x = data["accel_x"]
    accel_y = data["accel_y"]
    accel_z = data["accel_z"]
    gyro_x = data["gyro_x"]
    gyro_y = data["gyro_y"]
    gyro_z = data["gyro_z"]

    # Method 1: Log in loop
    for i, t in enumerate(timestamps):
        rr.set_time("timestamp", timestamp=t)

        # Accelerometer
        rr.log("sensors/imu/accel", rr.Scalars([accel_x[i], accel_y[i], accel_z[i]]))

        # Gyroscope
        rr.log("sensors/imu/gyro", rr.Scalars([gyro_x[i], gyro_y[i], gyro_z[i]]))

        # Magnitudes
        accel_mag = np.sqrt(accel_x[i] ** 2 + accel_y[i] ** 2 + accel_z[i] ** 2)
        rr.log("sensors/imu/accel_magnitude", rr.Scalars(accel_mag))

    # Method 2: Batch log with send_columns (faster)
    accel_data = np.column_stack([accel_x, accel_y, accel_z])
    rr.send_columns(
        "sensors/imu/accel_batch",
        indexes=[rr.TimeColumn("timestamp", timestamps)],
        columns=[*rr.Scalars.columns(scalars=accel_data)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="IMU data visualization")
    rr.script_add_args(parser)
    parser.add_argument("--imu-file", type=str, required=True)
    args = parser.parse_args()

    # Blueprint
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.TimeSeriesView(
                origin="sensors/imu/accel",
                name="Accelerometer",
                overrides={"sensors/imu/accel": rr.SeriesLines.from_fields(names=["X", "Y", "Z"])},
            ),
            rrb.TimeSeriesView(
                origin="sensors/imu/gyro",
                name="Gyroscope",
                overrides={"sensors/imu/gyro": rr.SeriesLines.from_fields(names=["X", "Y", "Z"])},
            ),
            rrb.TimeSeriesView(origin="sensors/imu/accel_magnitude", name="Acceleration Magnitude"),
        ),
        rrb.TimePanel(state="expanded"),
    )

    rr.script_setup(args, "imu_visualization", default_blueprint=blueprint)
    log_imu_data(args.imu_file)
    rr.script_teardown(args)


if __name__ == "__main__":
    main()
```

## Python Logging Integration

Integrate Python's logging module with Rerun.

```python
import argparse
import logging
import rerun as rr


def setup_logging() -> None:
    """Set up Python logging to Rerun."""
    logger = logging.getLogger()

    # Add Rerun handler
    rerun_handler = rr.LoggingHandler("logs")
    rerun_handler.setLevel(logging.DEBUG)
    logger.addHandler(rerun_handler)

    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    logger.setLevel(logging.DEBUG)


def process_data():
    """Example function that logs to Python logging."""
    logging.info("Starting data processing")

    for i in range(100):
        rr.set_time("step", sequence=i)

        if i % 10 == 0:
            logging.info(f"Processing step {i}")

        try:
            # Simulate processing
            result = complex_computation(i)
            rr.log("results/value", rr.Scalars(result))

            logging.debug(f"Step {i}: result = {result}")

        except Exception as e:
            logging.error(f"Error at step {i}: {e}")

    logging.info("Data processing complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Logging integration example")
    rr.script_add_args(parser)
    args = parser.parse_args()

    rr.script_setup(args, "logging_example")
    setup_logging()

    process_data()

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
```

## Multi-Process Logging

Log from multiple processes to the same Rerun session.

```python
import argparse
import multiprocessing as mp
import time
import rerun as rr


def worker_process(worker_id: int, queue: mp.Queue) -> None:
    """Worker that generates data."""
    for i in range(100):
        # Generate some data
        value = worker_id * 100 + i
        timestamp = time.time()

        # Send to queue
        queue.put({"worker_id": worker_id, "step": i, "value": value, "timestamp": timestamp})

        time.sleep(0.1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-process logging")
    rr.script_add_args(parser)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    rr.script_setup(args, "multiprocess_logging")

    # Create queue for communication
    queue = mp.Queue()

    # Start worker processes
    processes = []
    for worker_id in range(args.workers):
        p = mp.Process(target=worker_process, args=(worker_id, queue))
        p.start()
        processes.append(p)

    # Main process logs data from queue
    active_workers = args.workers
    while active_workers > 0:
        try:
            data = queue.get(timeout=1.0)

            rr.set_time("step", sequence=data["step"])
            rr.set_time("timestamp", timestamp=data["timestamp"])

            rr.log(f"workers/{data['worker_id']}/value", rr.Scalars(data["value"]))

        except:
            # Check if all processes are done
            active_workers = sum(p.is_alive() for p in processes)

    # Wait for all processes
    for p in processes:
        p.join()

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
```

## Best Practices Summary

1. **Always set up annotations** for classified data (detections, segmentations)
2. **Use static=True** for configuration data (coordinate systems, class descriptions)
3. **Compress images** in high-frequency logging scenarios
4. **Batch log** time series data with `send_columns` for performance
5. **Create blueprints** for consistent viewer layouts
6. **Use hierarchical entity paths** to organize related data
7. **Set multiple timelines** when you have different time bases (frames, wall clock, etc.)
8. **Clear old data** that's no longer relevant
9. **Log metadata** with `rr.AnyValues` for debugging
10. **Test incrementally** - start simple, add complexity gradually
