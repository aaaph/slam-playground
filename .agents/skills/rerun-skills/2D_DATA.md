# 2D Data Visualization

Detailed guide for visualizing 2D spatial data, images, and annotations with Rerun.

## Points2D - 2D Points and Keypoints

Visualize 2D points, keypoints for pose estimation, and feature points.

### Basic Usage

```python
import rerun as rr

# Simple 2D points
points = [(100, 150), (200, 250), (300, 350)]
rr.log("image/points", rr.Points2D(points))

# With radii
rr.log("image/points", rr.Points2D(points, radii=5))
```

### Keypoints for Pose Estimation

```python
import math

# MediaPipe-style keypoints (normalized [0, 1] to pixel coords)
height, width = image.shape[:2]
keypoint_positions = [
    (math.floor(kp.x * width), math.floor(kp.y * height))
    for kp in landmarks
]

rr.log("image/face/keypoints", rr.Points2D(
    keypoint_positions,
    radii=3,
    keypoint_ids=list(range(len(keypoint_positions))),
    class_ids=class_ids  # Links to AnnotationContext
))
```

### With Colors

```python
# Different color per point
colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
rr.log("image/points", rr.Points2D(
    points,
    colors=colors,
    radii=4
))
```

### Expected Data

- `positions`: List of `(x, y)` tuples or `(N, 2)` numpy array
- `radii`: Scalar or `(N,)` array in pixel units
- `keypoint_ids`: `(N,)` array of integers
- `class_ids`: `(N,)` array of integers
- `colors`: `(N, 3)` or `(N, 4)` uint8 array

## Boxes2D - 2D Bounding Boxes

Visualize 2D bounding boxes for object detection and tracking.

### XYWH Format (x, y, width, height)

```python
# Bounding boxes in XYWH format
boxes = [
    [100, 150, 50, 80],   # x, y, width, height
    [200, 250, 60, 90]
]

rr.log("image/boxes", rr.Boxes2D(
    array=boxes,
    array_format=rr.Box2DFormat.XYWH,
    class_ids=[0, 1]
))
```

### XYXY Format (x_min, y_min, x_max, y_max)

```python
# Bounding boxes in XYXY format
boxes = [
    [100, 150, 150, 230],  # x_min, y_min, x_max, y_max
    [200, 250, 260, 340]
]

rr.log("image/boxes", rr.Boxes2D(
    array=boxes,
    array_format=rr.Box2DFormat.XYXY,
    class_ids=[0, 1]
))
```

### Object Detection Example

```python
# With detection scores and class IDs
for frame_idx, detections in enumerate(detection_results):
    rr.set_time("frame", sequence=frame_idx)

    boxes = [det["bbox"] for det in detections]
    class_ids = [det["class_id"] for det in detections]

    rr.log("video/detections/boxes", rr.Boxes2D(
        array=boxes,
        array_format=rr.Box2DFormat.XYWH,
        class_ids=class_ids
    ))

    # Log additional metadata
    scores = [det["score"] for det in detections]
    rr.log("video/detections/metadata", rr.AnyValues(
        scores=scores,
        count=len(detections)
    ))
```

### Tracking with Unique IDs

```python
# Each tracked object gets its own entity path
for tracker in trackers:
    rr.log(
        f"video/tracked/{tracker.id}",
        rr.Boxes2D(
            array=tracker.bbox,
            array_format=rr.Box2DFormat.XYWH,
            class_ids=tracker.class_id
        )
    )

# Clear boxes for lost tracks
rr.log(f"video/tracked/{lost_id}", rr.Boxes2D.cleared())
```

### Expected Data

- `array`: List or `(N, 4)` numpy array
- `array_format`: `rr.Box2DFormat.XYWH` or `rr.Box2DFormat.XYXY`
- `class_ids`: `(N,)` array of integers

## Image - 2D Images

Log 2D images including RGB, grayscale, depth maps, and more.

### RGB Images

```python
import numpy as np

# RGB image (H, W, 3)
rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
rr.log("camera/rgb", rr.Image(rgb_image))
```

### BGR Images (OpenCV)

```python
import cv2

# OpenCV reads images as BGR
bgr_image = cv2.imread("image.jpg")
rr.log("camera/image", rr.Image(bgr_image, color_model="BGR"))
```

### Grayscale Images

```python
# Grayscale image (H, W) or (H, W, 1)
gray_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
rr.log("camera/gray", rr.Image(gray_image))
```

### Depth Images

```python
# Depth map (H, W) with float values
depth_map = np.random.rand(480, 640).astype(np.float32) * 10.0
rr.log("camera/depth", rr.Image(depth_map))
```

### Image Compression

For bandwidth efficiency, compress images:

```python
# JPEG compression (quality 0-100)
rr.log("camera/image", rr.Image(rgb_image).compress(jpeg_quality=85))

# Higher quality
rr.log("camera/hq", rr.Image(rgb_image).compress(jpeg_quality=95))
```

### Expected Shapes

- RGB: `(H, W, 3)` uint8
- RGBA: `(H, W, 4)` uint8
- Grayscale: `(H, W)` or `(H, W, 1)` uint8 or float
- Depth: `(H, W)` float

## SegmentationImage - Segmentation Masks

Visualize semantic and instance segmentation masks where each pixel value represents a class ID.

### Basic Segmentation

```python
# Segmentation mask where pixel values are class IDs
mask = np.zeros((480, 640), dtype=np.uint8)
mask[100:200, 100:200] = 1  # Class 1
mask[300:400, 300:400] = 2  # Class 2

rr.log("segmentation", rr.SegmentationImage(mask))
```

### With Annotation Context

```python
# Define classes first
class_descriptions = [
    rr.AnnotationInfo(id=0, label="background", color=[0, 0, 0]),
    rr.AnnotationInfo(id=1, label="person", color=[255, 0, 0]),
    rr.AnnotationInfo(id=2, label="car", color=[0, 255, 0]),
    rr.AnnotationInfo(id=3, label="building", color=[0, 0, 255])
]
rr.log("/", rr.AnnotationContext(class_descriptions), static=True)

# Now log segmentation
rr.log("segmentation", rr.SegmentationImage(mask))
```

### From Model Output

```python
# Typical semantic segmentation model output
model_output = model.predict(image)  # Shape (H, W) with class indices
mask = model_output.argmax(axis=-1).astype(np.uint8)

rr.log("segmentation/prediction", rr.SegmentationImage(mask))
```

### Multi-Scale Segmentation

```python
# Log segmentation at different scales
for scale_name, segmentation in segmentations.items():
    rr.log(f"segmentation/{scale_name}", rr.SegmentationImage(segmentation))
```

## AnnotationContext - Class Definitions

Define classes with colors, labels, and keypoint connections.

### Basic Class Definitions

```python
class_descriptions = [
    rr.AnnotationInfo(id=0, label="background", color=[0, 0, 0]),
    rr.AnnotationInfo(id=1, label="person", color=[255, 0, 0]),
    rr.AnnotationInfo(id=2, label="vehicle", color=[0, 255, 0]),
    rr.AnnotationInfo(id=3, label="building", color=[0, 0, 255])
]

rr.log("/", rr.AnnotationContext(class_descriptions), static=True)
```

### From COCO Categories

```python
import json

with open("coco_categories.json") as f:
    coco_categories = json.load(f)

class_descriptions = [
    rr.AnnotationInfo(
        id=cat["id"],
        label=cat["name"],
        color=cat["color"]
    )
    for cat in coco_categories
]

rr.log("/", rr.AnnotationContext(class_descriptions), static=True)
```

### With Keypoint Connections

Define skeleton structure for pose estimation:

```python
# Face detection with keypoint connections
rr.log("video/detector", rr.ClassDescription(
    info=rr.AnnotationInfo(id=0),
    keypoint_connections=[
        (0, 1),  # Right eye to left eye
        (1, 2),  # Left eye to nose
        (2, 0),  # Nose to right eye
        (2, 3),  # Nose to mouth right
        (0, 4),  # Right eye to right ear
        (1, 5)   # Left eye to left ear
    ]
), static=True)
```

### MediaPipe Face Mesh Example

```python
import mediapipe as mp

# Extract keypoint connections from MediaPipe
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

class_descriptions = []
for i, klass in enumerate(classes):
    class_descriptions.append(
        rr.ClassDescription(
            info=rr.AnnotationInfo(id=i),
            keypoint_connections=klass
        )
    )

rr.log("video/face", rr.AnnotationContext(class_descriptions), static=True)
```

## TextDocument - Text and Markdown

Display text, markdown documentation, and descriptions.

### Markdown Documents

```python
markdown_text = """
# My Application

This visualization shows:
- Object detection results
- Tracking over time
- 3D reconstruction

## Controls
Use the timeline to scrub through frames.
"""

rr.log("description", rr.TextDocument(
    markdown_text,
    media_type=rr.MediaType.MARKDOWN
), static=True)
```

### Plain Text

```python
rr.log("readme", rr.TextDocument(
    "Plain text content here",
    media_type=rr.MediaType.PLAIN_TEXT
), static=True)
```

### Dynamic Text

```python
# Update text over time
for frame_idx, status in enumerate(statuses):
    rr.set_time("frame", sequence=frame_idx)
    rr.log("status", rr.TextDocument(f"Processing frame {frame_idx}: {status}"))
```

## AnyValues - Arbitrary Metadata

Log arbitrary key-value metadata.

```python
# Log metadata alongside other data
rr.log("detection",
    rr.Boxes2D(boxes, array_format=rr.Box2DFormat.XYWH),
    rr.AnyValues(
        score=0.95,
        index=42,
        name="pedestrian",
        confidence_threshold=0.8
    )
)
```

### Logging Scores and Metrics

```python
# Per-detection metadata
for i, detection in enumerate(detections):
    rr.log(f"detections/{i}",
        rr.Boxes2D([detection.bbox], array_format=rr.Box2DFormat.XYWH),
        rr.AnyValues(
            score=detection.score,
            class_name=detection.class_name
        )
    )
```

## Combining 2D Data Types

### Object Detection Pipeline

```python
import cv2

# Set up classes
class_descriptions = [
    rr.AnnotationInfo(id=0, label="person", color=[255, 0, 0]),
    rr.AnnotationInfo(id=1, label="car", color=[0, 255, 0])
]
rr.log("/", rr.AnnotationContext(class_descriptions), static=True)

# Process video
cap = cv2.VideoCapture("video.mp4")
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rr.set_time("frame", sequence=frame_idx)

    # Log frame
    rr.log("video/image", rr.Image(frame, color_model="BGR"))

    # Run detection
    detections = detector.detect(frame)

    # Log boxes
    boxes = [d.bbox for d in detections]
    class_ids = [d.class_id for d in detections]
    rr.log("video/detections", rr.Boxes2D(
        array=boxes,
        array_format=rr.Box2DFormat.XYWH,
        class_ids=class_ids
    ))

    frame_idx += 1
```

### Pose Estimation with Keypoints

```python
# Define skeleton
rr.log("pose", rr.ClassDescription(
    info=rr.AnnotationInfo(id=0),
    keypoint_connections=[
        (0, 1), (1, 2), (2, 3),  # Right arm
        (0, 4), (4, 5), (5, 6),  # Left arm
        (0, 7), (7, 8), (8, 9),  # Right leg
        (0, 10), (10, 11), (11, 12)  # Left leg
    ]
), static=True)

# Log keypoints
for frame_idx, pose in enumerate(poses):
    rr.set_time("frame", sequence=frame_idx)

    rr.log("video/image", rr.Image(frames[frame_idx]))

    rr.log("video/pose/keypoints", rr.Points2D(
        pose.keypoints,
        keypoint_ids=list(range(len(pose.keypoints))),
        radii=5,
        colors=pose.confidences_to_colors()  # Color by confidence
    ))
```

### Image with Overlay

```python
# Log base image
rr.log("camera/rgb", rr.Image(rgb_image))

# Log segmentation overlay
rr.log("camera/segmentation", rr.SegmentationImage(mask))

# Log detection boxes
rr.log("camera/detections", rr.Boxes2D(boxes, array_format=rr.Box2DFormat.XYWH))

# Log feature points
rr.log("camera/features", rr.Points2D(feature_points, radii=3))
```

## Tips and Best Practices

### Image Coordinate System

2D coordinates use standard image coordinates:
- Origin `(0, 0)` at top-left
- X increases to the right
- Y increases downward

### Scaling Detections

When working with resized images:

```python
# Detection on small image
small_image = cv2.resize(original, (320, 240))
detections = model.detect(small_image)

# Scale bounding boxes back to original size
scale_x = original.shape[1] / small_image.shape[1]
scale_y = original.shape[0] / small_image.shape[0]

scaled_boxes = [
    [bbox[0] * scale_x, bbox[1] * scale_y,
     bbox[2] * scale_x, bbox[3] * scale_y]
    for bbox in detections.bboxes
]

rr.log("video/image", rr.Image(original, color_model="BGR"))
rr.log("video/boxes", rr.Boxes2D(
    array=scaled_boxes,
    array_format=rr.Box2DFormat.XYWH
))
```

### Efficient Image Logging

```python
# For high-frequency logging, use compression
rr.log("camera/image",
    rr.Image(frame, color_model="BGR").compress(jpeg_quality=85))

# For static reference images, no compression needed
rr.log("reference", rr.Image(reference_image), static=True)
```
