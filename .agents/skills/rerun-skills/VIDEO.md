# Video Logging

Detailed guide for logging and visualizing video data with Rerun.

## AssetVideo - Video File Reference

Log a video file once as a static asset, then reference frames by timestamp.

### Basic Usage

```python
import rerun as rr

# Log video file once
video_path = "video.mp4"
video_asset = rr.AssetVideo(path=video_path)
rr.log("video", video_asset, static=True)

# Get frame timestamps
frame_timestamps_ns = video_asset.read_frame_timestamps_nanos()

# Reference frames as you process them
for frame_idx in range(len(frame_timestamps_ns)):
    rr.set_time("frame", sequence=frame_idx)
    rr.log("video", rr.VideoFrameReference(
        nanoseconds=frame_timestamps_ns[frame_idx]
    ))
```

### Complete Example

```python
import cv2
import rerun as rr

def process_video(video_path: str) -> None:
    # Log video asset
    video_asset = rr.AssetVideo(path=video_path)
    frame_timestamps_ns = video_asset.read_frame_timestamps_nanos()
    rr.log("video", video_asset, static=True)

    # Process video with OpenCV
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, bgr = cap.read()
        if not ret:
            break

        # Set time
        rr.set_time("frame", sequence=frame_idx)

        # Reference the video frame
        rr.log("video", rr.VideoFrameReference(
            nanoseconds=frame_timestamps_ns[frame_idx]
        ))

        # Process frame
        detections = detect_objects(bgr)

        # Log detections on top of video
        rr.log("video/detections", rr.Boxes2D(
            array=[d.bbox for d in detections],
            array_format=rr.Box2DFormat.XYWH,
            class_ids=[d.class_id for d in detections]
        ))

        frame_idx += 1

    cap.release()
```

## Video + Processing Pipeline

### Object Detection on Video

```python
def track_objects(video_path: str) -> None:
    # Set up annotation context
    class_descriptions = [
        rr.AnnotationInfo(id=0, label="person", color=[255, 0, 0]),
        rr.AnnotationInfo(id=1, label="car", color=[0, 255, 0]),
        rr.AnnotationInfo(id=2, label="bicycle", color=[0, 0, 255])
    ]
    rr.log("/", rr.AnnotationContext(class_descriptions), static=True)

    # Log video
    video_asset = rr.AssetVideo(path=video_path)
    frame_timestamps_ns = video_asset.read_frame_timestamps_nanos()
    rr.log("video", video_asset, static=True)

    # Process each frame
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, bgr = cap.read()
        if not ret:
            break

        rr.set_time("frame", sequence=frame_idx)

        # Reference video frame
        rr.log("video", rr.VideoFrameReference(
            nanoseconds=frame_timestamps_ns[frame_idx]
        ))

        # Run detection every frame
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        detections = detector.detect(rgb)

        # Log detections
        rr.log("video/detections", rr.Boxes2D(
            array=[d.bbox for d in detections],
            array_format=rr.Box2DFormat.XYWH,
            class_ids=[d.class_id for d in detections]
        ))

        frame_idx += 1

    cap.release()
```

### Video with Segmentation

```python
def segment_video(video_path: str) -> None:
    # Log video
    video_asset = rr.AssetVideo(path=video_path)
    frame_timestamps_ns = video_asset.read_frame_timestamps_nanos()
    rr.log("video/rgb", video_asset, static=True)

    # Process frames
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, bgr = cap.read()
        if not ret:
            break

        rr.set_time("frame", sequence=frame_idx)

        # Reference video frame
        rr.log("video/rgb", rr.VideoFrameReference(
            nanoseconds=frame_timestamps_ns[frame_idx]
        ))

        # Run segmentation
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = segmentation_model(rgb)

        # Log segmentation overlay
        rr.log("video/segmentation", rr.SegmentationImage(mask))

        frame_idx += 1

    cap.release()
```

## Alternative: Frame-by-Frame Logging

If you don't want to use AssetVideo, you can log frames individually:

### Log Individual Frames

```python
cap = cv2.VideoCapture(video_path)
frame_idx = 0

while cap.isOpened():
    ret, bgr = cap.read()
    if not ret:
        break

    rr.set_time("frame", sequence=frame_idx)

    # Log frame as image (no video asset)
    rr.log("camera/image", rr.Image(bgr, color_model="BGR"))

    # Process and log other data
    detections = detect_objects(bgr)
    rr.log("camera/detections", rr.Boxes2D(...))

    frame_idx += 1

cap.release()
```

### With Compression

```python
# Compress frames for efficiency
rr.log("camera/image",
    rr.Image(bgr, color_model="BGR").compress(jpeg_quality=85)
)
```

## Live Video Streams

### Camera Stream

```python
def stream_camera(camera_id: int = 0) -> None:
    cap = cv2.VideoCapture(camera_id)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate frame time
            frame_time_nano = int(frame_idx * 1000 / fps * 1e6)

            # Set time
            rr.set_time("frame_nr", sequence=frame_idx)
            rr.set_time("frame_time", duration=1e-9 * frame_time_nano)

            # Log frame
            rr.log("camera/image", rr.Image(frame, color_model="BGR"))

            # Process frame
            process_frame(frame)

            frame_idx += 1

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
```

### With Frame Rate Control

```python
import time

def stream_camera_controlled(camera_id: int = 0, target_fps: int = 30) -> None:
    cap = cv2.VideoCapture(camera_id)
    frame_period = 1.0 / target_fps

    frame_idx = 0
    try:
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            rr.set_time("frame", sequence=frame_idx)
            rr.log("camera/image", rr.Image(frame, color_model="BGR"))

            # Process frame
            process_frame(frame)

            # Sleep to maintain frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_period - elapsed)
            time.sleep(sleep_time)

            frame_idx += 1

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
```

## Multi-Camera Video

```python
def log_multi_camera(video_paths: list[str]) -> None:
    # Load all videos
    assets = []
    timestamps = []
    for i, path in enumerate(video_paths):
        asset = rr.AssetVideo(path=path)
        rr.log(f"camera_{i}/video", asset, static=True)
        assets.append(asset)
        timestamps.append(asset.read_frame_timestamps_nanos())

    # Assume synchronized
    num_frames = min(len(ts) for ts in timestamps)

    for frame_idx in range(num_frames):
        rr.set_time("frame", sequence=frame_idx)

        # Reference frame from each camera
        for i, (asset, ts) in enumerate(zip(assets, timestamps)):
            rr.log(f"camera_{i}/video",
                rr.VideoFrameReference(nanoseconds=ts[frame_idx])
            )
```

## Video with 3D Overlay

Combine video with 3D visualizations:

```python
def video_with_3d(video_path: str) -> None:
    # Set up 3D coordinate system
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Log video
    video_asset = rr.AssetVideo(path=video_path)
    frame_timestamps_ns = video_asset.read_frame_timestamps_nanos()
    rr.log("world/camera/image", video_asset, static=True)

    # Process video
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, bgr = cap.read()
        if not ret:
            break

        rr.set_time("frame", sequence=frame_idx)

        # Video frame
        rr.log("world/camera/image",
            rr.VideoFrameReference(nanoseconds=frame_timestamps_ns[frame_idx])
        )

        # Camera pose in 3D
        pose = estimate_camera_pose(bgr)
        rr.log("world/camera", rr.Transform3D(
            translation=pose.position,
            rotation=rr.Quaternion(xyzw=pose.quaternion)
        ))

        # 3D points
        points_3d = triangulate_points(bgr)
        rr.log("world/points", rr.Points3D(points_3d))

        frame_idx += 1

    cap.release()
```

## Blueprint Configuration

```python
import rerun.blueprint as rrb

# Layout with video and other views
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        # Video on the left
        rrb.Spatial2DView(origin="video", name="Video"),
        # Other views on the right
        rrb.Vertical(
            rrb.TimeSeriesView(origin="metrics", name="Metrics"),
            rrb.Spatial3DView(origin="world", name="3D Scene")
        ),
        column_shares=[2, 1]
    )
)

rr.script_setup(args, "video_app", default_blueprint=blueprint)
```

## Tips and Best Practices

### When to Use AssetVideo

- **Use AssetVideo when**: You have a video file and want to reference it efficiently
- **Use Image logging when**: Processing live streams or don't have a video file

### Memory Efficiency

```python
# AssetVideo - references video file, low memory
rr.log("video", rr.AssetVideo(path=video_path), static=True)

# Image logging - stores frames in recording, higher memory
rr.log("video", rr.Image(frame))

# Compressed images - middle ground
rr.log("video", rr.Image(frame).compress(jpeg_quality=85))
```

### Synchronization

For synchronized multi-camera setups:

```python
# Use consistent timestamps
rr.set_time("timestamp", timestamp=frame_timestamp_seconds)
rr.set_time("frame", sequence=frame_idx)
```

### Processing Every Nth Frame

```python
# Process heavy operations less frequently
if frame_idx % 10 == 0:
    # Run detection/segmentation
    detections = detector.detect(frame)
    rr.log("video/detections", rr.Boxes2D(...))
else:
    # Just track
    update_trackers(frame)
```
