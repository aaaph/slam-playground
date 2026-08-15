---
name: rerun-visualization
description: Visualize various data types using the Rerun SDK including 3D point clouds, 2D/3D images, plots, graphs, video, sensor data, meshes, and more. Use when creating visualizations, logging data for analysis, or building interactive visual applications with temporal data.
allowed-tools: Read, Write, Bash, Grep, Glob
---

# Rerun Visualization

A comprehensive skill for creating visualizations using the Rerun SDK. Rerun is a time-series visualization engine for computer vision, robotics, and general multimodal data.

## Quick Start

### Basic Initialization

```python
import rerun as rr

# Initialize Rerun with spawn=True to open the viewer automatically
rr.init("my_application_name", spawn=True)

# Log your data
rr.log("entity/path", rr.Points3D(positions, colors=colors, radii=0.5))
```

### With Arguments Support (Recommended)

```python
import argparse
import rerun as rr

parser = argparse.ArgumentParser(description="My application")
rr.script_add_args(parser)
args = parser.parse_args()

rr.script_setup(args, "my_application_name")
# ... log data ...
rr.script_teardown(args)
```

## Installation

```bash
pip install rerun-sdk
```

## Core Concepts

### Entity Paths
- Data is logged to hierarchical entity paths like `"world/camera/image"`
- Use `/` to create hierarchies for organization
- Entity paths support wildcards in blueprint queries
- Example hierarchy:
  ```
  world/
    camera/
      image
      depth
    lidar/
      points
    detections/
      boxes
  ```

### Time Management

Rerun supports multiple timelines for temporal data:

```python
# Sequence-based timeline (frame numbers)
rr.set_time("frame", sequence=frame_idx)

# Wall-clock time (in seconds since epoch)
rr.set_time("timestamp", timestamp=time.time())

# Duration-based time (elapsed seconds)
rr.set_time("stable_time", duration=elapsed_seconds)
```

You can have multiple active timelines simultaneously. The viewer allows switching between them.

### Static vs Temporal Data

Data logged as `static=True` persists across all time and doesn't get cleared:

```python
# Static data - persists forever, overwritten if logged again
rr.log("description", rr.TextDocument(text, media_type=rr.MediaType.MARKDOWN), static=True)
rr.log("/", rr.AnnotationContext(class_descriptions), static=True)
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

# Temporal data - associated with current timeline values
rr.set_time("frame", sequence=42)
rr.log("world/points", rr.Points3D(positions))  # Only visible at frame 42
```

### Clearing Data

Remove data from entity paths:

```python
# Clear specific entity only
rr.log("entity/path", rr.Clear(recursive=False))

# Clear entity and all children
rr.log("entity/path", rr.Clear(recursive=True))
```

## Data Type Categories

Rerun supports many data types organized by category. Click on the links below for detailed documentation:

### [3D Data](3D_DATA.md)
Visualize 3D spatial data:
- **Points3D** - Point clouds with colors and radii
- **LineStrips3D** - 3D lines and trajectories
- **Boxes3D** - 3D bounding boxes
- **Mesh3D** - Triangle meshes with normals and colors
- **Transform3D** - 3D transformations (translation, rotation, scale)
- **ViewCoordinates** - Coordinate system conventions
- **Pinhole** - Camera projection model (required for showing 2D data in 3D views)

### [2D Data](2D_DATA.md)
Visualize 2D spatial data:
- **Points2D** - 2D points and keypoints
- **Boxes2D** - 2D bounding boxes (XYWH, XYXY formats)
- **Image** - 2D images (RGB, BGR, grayscale, depth)
- **SegmentationImage** - Segmentation masks with class IDs

### [Video](VIDEO.md)
Log and visualize video data:
- **AssetVideo** - Video file references
- **VideoFrameReference** - Reference frames by timestamp

### [Plots and Time Series](PLOTS.md)
Visualize temporal data and charts:
- **Scalars** - Time series line plots
- **SeriesLines** - Line plot styling
- **SeriesPoints** - Scatter plot styling
- **BarChart** - Bar charts
- **send_columns** - Efficient batch logging

### [Graphs](GRAPHS.md)
Visualize graph structures:
- **GraphNodes** - Define nodes with positions, colors, labels
- **GraphEdges** - Define edges between nodes
- Force-based layouts for automatic positioning

### [Annotations and Metadata](2D_DATA.md)
Add context and metadata:
- **AnnotationContext** - Define classes with colors and labels
- **ClassDescription** - Keypoint connections for poses
- **TextDocument** - Markdown and plain text
- **AnyValues** - Arbitrary metadata key-value pairs

## [Blueprints - Viewer Layout](BLUEPRINTS.md)

Blueprints define the viewer's layout and configuration:

```python
import rerun.blueprint as rrb

blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial3DView(origin="world"), rrb.Spatial2DView(origin="camera/image"), column_shares=[2, 1]
    )
)

rr.script_setup(args, "app_name", default_blueprint=blueprint)
```

See [BLUEPRINTS.md](BLUEPRINTS.md) for detailed layout options and view configurations.

## [Common Patterns](PATTERNS.md)

See [PATTERNS.md](PATTERNS.md) for detailed examples of:
- Object detection and tracking
- Pose estimation with keypoints
- 3D reconstruction and point clouds
- LIDAR visualization
- Sensor data logging (IMU, GPS, etc.)
- Python logging integration

## Saving and Loading Recordings

```bash
# Save to .rrd file
python my_script.py --save output.rrd

# View saved recording
rerun output.rrd
```

## Best Practices

1. **Use meaningful entity paths** - Create hierarchies that reflect your data structure
   ```python
   rr.log("world/sensors/lidar/points", ...)
   rr.log("world/sensors/camera/rgb", ...)
   ```

2. **Set coordinate systems early** - Use `ViewCoordinates` for 3D data
   ```python
   rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
   ```

3. **Log static data as static** - Configuration and unchanging data
   ```python
   rr.log("/", rr.AnnotationContext(classes), static=True)
   ```

4. **Compress large images** - Save bandwidth
   ```python
   rr.log("image", rr.Image(rgb).compress(jpeg_quality=85))
   ```

5. **Use blueprints** - Define layouts programmatically
   ```python
   rr.script_setup(args, "app", default_blueprint=blueprint)
   ```

6. **Set appropriate time** - Use multiple timelines when needed
   ```python
   rr.set_time("frame", sequence=i)
   rr.set_time("timestamp", timestamp=t)
   ```

7. **Clear old data** - Remove outdated visualizations
   ```python
   rr.log("detections", rr.Clear(recursive=True))
   ```

## Common Issues

### Memory Usage
Rerun accumulates data until memory budget is reached. For long-running applications:
- Log data as `static=True` to overwrite instead of accumulate
- Use `--memory-limit` argument to control memory usage
- Clear entities that are no longer needed with `rr.Clear()`

### Color Models
- **OpenCV uses BGR** - specify `color_model="BGR"` when logging OpenCV images
- **MediaPipe uses RGB** - no need to specify color model
- **NumPy arrays** - ensure correct shape: (H, W, 3) for RGB/BGR

### Array Shapes
- Points3D expects `(N, 3)` shape
- Points2D expects `(N, 2)` shape
- Colors can be `(N, 3)` for RGB or `(N, 4)` for RGBA
- Use `uint8` for colors (0-255) and `float32` for positions

### Coordinate Systems
Always set coordinate system for 3D data to ensure correct visualization:
```python
# Common conventions
rr.log("world", rr.ViewCoordinates.RDF, static=True)  # Right-Down-Forward
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # Right-Hand Z-Up
```

## Additional Resources

- [Rerun Documentation](https://www.rerun.io/docs)
- [Example Gallery](https://www.rerun.io/examples)
- [GitHub Repository](https://github.com/rerun-io/rerun)
- [Python API Reference](https://ref.rerun.io/docs/python)
