# 3D Data Visualization

Detailed guide for visualizing 3D spatial data with Rerun.

## Points3D - 3D Point Clouds

Visualize point clouds with customizable colors, sizes, and keypoint information.

### Basic Usage

```python
import numpy as np
import rerun as rr

# Simple point cloud
positions = np.random.randn(1000, 3)
rr.log("world/points", rr.Points3D(positions))
```

### With Colors and Radii

```python
# Create colored point cloud
positions = np.vstack([xyz.ravel() for xyz in np.mgrid[3 * [slice(-10, 10, 10j)]]]).T
colors = np.vstack([rgb.ravel() for rgb in np.mgrid[3 * [slice(0, 255, 10j)]]]).astype(np.uint8).T

rr.log("world/points", rr.Points3D(
    positions,       # (N, 3) numpy array
    colors=colors,   # (N, 3) uint8 for RGB or (N, 4) for RGBA
    radii=0.5        # scalar or (N,) array
))
```

### Distance-Based Coloring (LIDAR)

```python
import matplotlib

# Color points by distance from origin
cmap = matplotlib.colormaps["turbo_r"]
norm = matplotlib.colors.Normalize(vmin=3.0, vmax=75.0)

point_distances = np.linalg.norm(points, axis=1)
point_colors = cmap(norm(point_distances))

rr.log("world/lidar", rr.Points3D(points, colors=point_colors))
```

### With Keypoints (Pose Estimation)

```python
# For pose estimation, tracking, etc.
rr.log("person/keypoints", rr.Points3D(
    keypoint_positions,
    keypoint_ids=list(range(num_keypoints)),
    class_ids=class_ids  # Associate with AnnotationContext
))
```

### Expected Data Shapes

- `positions`: `(N, 3)` array of xyz coordinates
- `colors`: `(N, 3)` for RGB or `(N, 4)` for RGBA, values 0-255 as uint8
- `radii`: scalar float or `(N,)` array
- `keypoint_ids`: `(N,)` array of integers
- `class_ids`: `(N,)` array of integers

## LineStrips3D - 3D Lines and Trajectories

Visualize continuous lines, trajectories, and connections in 3D space.

### Basic Line Strips

```python
# Single line strip
trajectory = np.array([[0, 0, 0], [1, 1, 1], [2, 0, 1], [3, 1, 0]])
rr.log("path", rr.LineStrips3D([trajectory]))

# Multiple line strips
rr.log("trajectories", rr.LineStrips3D([
    trajectory1,
    trajectory2,
    trajectory3
]))
```

### With Colors

```python
# Different color per strip
rr.log("trajectories", rr.LineStrips3D(
    [trajectory1, trajectory2],
    colors=[[255, 0, 0], [0, 255, 0]]  # Red and green
))

# Single color for all
rr.log("paths", rr.LineStrips3D(
    strips,
    colors=[128, 128, 128]  # Gray for all
))
```

### Connecting Points (Scaffolding)

```python
# Connect two point clouds (e.g., DNA helix)
points1 = build_spiral(num_points)
points2 = build_spiral(num_points, angular_offset=np.pi)

# Stack creates (N, 2, 3) array connecting pairs
connections = np.stack((points1, points2), axis=1)
rr.log("structure/scaffolding", rr.LineStrips3D(connections, colors=[128, 128, 128]))
```

### Data Format

- Input: List of arrays, each array is shape `(M, 3)` for M points in that strip
- Colors: Single `[R, G, B]` or list of `[R, G, B]` per strip

## Boxes3D - 3D Bounding Boxes

Visualize 3D bounding boxes for object detection, tracking, and spatial reasoning.

### Basic Usage

```python
# Define boxes with centers and sizes
centers = np.array([[0, 0, 0], [2, 0, 0]])
sizes = np.array([[1, 1, 1], [0.5, 0.5, 2]])

rr.log("world/boxes", rr.Boxes3D(
    centers=centers,
    sizes=sizes
))
```

### With Rotations and Classes

```python
# With quaternion rotations
rr.log("world/boxes", rr.Boxes3D(
    centers=centers,
    sizes=sizes,
    rotations=quaternions,  # (N, 4) array [x, y, z, w]
    class_ids=class_ids     # Associate with AnnotationContext
))
```

### Expected Data

- `centers`: `(N, 3)` array of box centers
- `sizes`: `(N, 3)` array of box dimensions [width, height, depth]
- `rotations`: `(N, 4)` array of quaternions [x, y, z, w]
- `class_ids`: `(N,)` array of integers

## Mesh3D - Triangle Meshes

Visualize 3D meshes with vertices, faces, colors, and normals.

### Basic Triangle Mesh

```python
# Define vertices and faces (triangles)
vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# Faces as indices into vertices (each row is a triangle)
faces = np.array([
    [0, 1, 2],
    [0, 1, 3],
    [0, 2, 3],
    [1, 2, 3]
], dtype=np.uint32)

rr.log("world/mesh", rr.Mesh3D(
    vertex_positions=vertices,
    indices=faces
))
```

### With Colors and Normals

```python
rr.log("world/mesh", rr.Mesh3D(
    vertex_positions=vertices,
    indices=faces,
    vertex_colors=colors,    # (N, 3) or (N, 4) uint8 colors per vertex
    vertex_normals=normals   # (N, 3) normal vectors
))
```

### Expected Data

- `vertex_positions`: `(N, 3)` array of vertex coordinates
- `indices`: `(M, 3)` array of triangle indices (uint32)
- `vertex_colors`: `(N, 3)` or `(N, 4)` uint8 array
- `vertex_normals`: `(N, 3)` array of normal vectors

## Transform3D - 3D Transformations

Apply transformations to entities in the hierarchy.

### Translation Only

```python
rr.log("world/object", rr.Transform3D(
    translation=[x, y, z]
))
```

### Rotation with Axis-Angle

```python
import math

rr.log("world/object", rr.Transform3D(
    rotation=rr.RotationAxisAngle(
        axis=[0, 0, 1],  # Z-axis
        radians=math.pi / 4  # 45 degrees
    )
))
```

### Rotation with Quaternion

```python
rr.log("world/object", rr.Transform3D(
    rotation=rr.Quaternion(xyzw=[x, y, z, w])
))
```

### Combined Transform

```python
rr.log("world/object", rr.Transform3D(
    translation=[x, y, z],
    rotation=rr.RotationAxisAngle(axis=[0, 0, 1], radians=angle),
    scale=[sx, sy, sz]  # or scalar for uniform scaling
))
```

### Animated Transformations

```python
# Rotate object over time
for i in range(400):
    time = i * 0.01
    rr.set_time("stable_time", duration=time)

    rr.log("helix/structure", rr.Transform3D(
        rotation=rr.RotationAxisAngle(
            axis=[0, 0, 1],
            radians=time / 4.0 * 2 * math.pi
        )
    ))
```

### Transform Hierarchy

Transforms are hierarchical - child entities inherit parent transforms:

```python
# Parent transform
rr.log("world/robot", rr.Transform3D(translation=[1, 0, 0]))

# Child inherits parent transform and adds its own
rr.log("world/robot/arm", rr.Transform3D(rotation=rotation))
rr.log("world/robot/arm/end_effector", rr.Transform3D(translation=[0, 0, 0.5]))
```

## ViewCoordinates - Coordinate System Convention

Define the coordinate system convention for 3D visualization.

### Common Coordinate Systems

```python
# Right-Down-Forward (common in computer vision)
rr.log("world", rr.ViewCoordinates.RDF, static=True)

# Right-Hand Z-Up (common in robotics, CAD)
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

# Forward-Left-Up
rr.log("world", rr.ViewCoordinates.FLU, static=True)

# Forward-Right-Down
rr.log("world", rr.ViewCoordinates.FRD, static=True)
```

### Why This Matters

Different fields use different coordinate conventions:
- **Computer Vision**: Often RDF (Right-Down-Forward) or RUB (Right-Up-Back)
- **Robotics**: Often RIGHT_HAND_Z_UP (RUF - Right-Up-Forward with Z pointing up)
- **Graphics**: Often RUB (Right-Up-Back, Y-up)

Setting `ViewCoordinates` ensures your 3D data is displayed correctly in the viewer.

### Best Practice

Always set coordinate system as static at the root of your spatial hierarchy:

```python
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
# Now all descendants of "world" use this convention
```

## Pinhole - Camera Projection Model

Define camera intrinsics for projecting 2D images into 3D space. **Required** to show 2D images/data in a 3D view.

### Basic Usage

```python
# Define camera intrinsics
rr.log("world/camera", rr.Pinhole(
    resolution=[width, height],
    focal_length=[fx, fy],
    principal_point=[cx, cy]
))

# Now 2D children can be shown in 3D
rr.log("world/camera/image", rr.Image(image))
```

### Common Camera Models

```python
# Standard pinhole camera
rr.log("camera", rr.Pinhole(
    resolution=[640, 480],
    focal_length=525.0,  # Same fx and fy
    principal_point=[320, 240]  # Image center
))

# Different focal lengths
rr.log("camera", rr.Pinhole(
    resolution=[1920, 1080],
    focal_length=[1000.0, 1000.0],  # [fx, fy]
    principal_point=[960.0, 540.0]  # [cx, cy]
))
```

### With Camera Transform

Pinhole must be combined with Transform3D to position the camera in 3D space:

```python
# Camera intrinsics
rr.log("world/camera", rr.Pinhole(
    resolution=[640, 480],
    focal_length=525.0,
    principal_point=[320, 240]
))

# Camera pose (extrinsics)
rr.log("world/camera", rr.Transform3D(
    translation=camera_position,
    rotation=rr.Quaternion(xyzw=camera_quaternion)
))

# Now image is positioned in 3D
rr.log("world/camera/image", rr.Image(rgb_image))
```

### RGB-D Camera Setup

For RGB-D cameras (like depth cameras), log both RGB and depth with the same pinhole:

```python
# Camera intrinsics (shared by RGB and depth)
rr.log("world/camera", rr.Pinhole(
    resolution=[640, 480],
    focal_length=525.0,
    principal_point=[320, 240]
), static=True)

# Camera pose
rr.log("world/camera", rr.Transform3D(
    translation=position,
    rotation=rotation
))

# RGB image
rr.log("world/camera/rgb", rr.Image(rgb_image))

# Depth image
rr.log("world/camera/depth", rr.Image(depth_image))
```

### From Camera Matrix

If you have a camera matrix K:

```python
# K = [[fx,  0, cx],
#      [ 0, fy, cy],
#      [ 0,  0,  1]]

fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]
cy = K[1, 2]

rr.log("camera", rr.Pinhole(
    resolution=[width, height],
    focal_length=[fx, fy],
    principal_point=[cx, cy]
))
```

### Common Camera Intrinsics

```python
# Kinect v1 / TUM RGB-D
rr.log("camera", rr.Pinhole(
    resolution=[640, 480],
    focal_length=525.0,
    principal_point=[319.5, 239.5]
))

# Intel RealSense D435
rr.log("camera", rr.Pinhole(
    resolution=[640, 480],
    focal_length=[615.0, 615.0],
    principal_point=[320.0, 240.0]
))
```

### Why Pinhole is Required

Without a Pinhole ancestor, 2D visualizers (Image, Boxes2D, Points2D, etc.) cannot be shown in a 3D view because Rerun doesn't know how to project them into 3D space. The error message will be:

```
2D visualizers require a pinhole ancestor to be shown in a 3D view.
```

Always add Pinhole to camera entities when you want to:
- Show camera images in a 3D scene
- Display 2D detections/annotations in 3D
- Visualize camera frustums
- Create RGB-D visualizations

## Tips and Tricks

### Efficient Point Cloud Updates

For real-time applications, log point clouds efficiently:

```python
# Good - reuse numpy arrays
points = np.zeros((1000, 3), dtype=np.float32)
colors = np.zeros((1000, 3), dtype=np.uint8)

for frame in frames:
    # Update arrays in-place
    update_points(points)
    update_colors(colors)

    rr.set_time("frame", sequence=frame)
    rr.log("world/points", rr.Points3D(points, colors=colors))
```

### Combining Multiple 3D Data Types

```python
# Typical 3D scene setup
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

# Point cloud
rr.log("world/points", rr.Points3D(points, colors=colors))

# Bounding boxes
rr.log("world/detections/boxes", rr.Boxes3D(centers, sizes, class_ids=ids))

# Camera pose
rr.log("world/camera", rr.Transform3D(translation=pos, rotation=rot))

# Mesh
rr.log("world/environment/mesh", rr.Mesh3D(vertices, faces))

# Trajectory
rr.log("world/trajectory", rr.LineStrips3D([path_points]))
```

### Color Handling

```python
# NumPy uint8 colors (0-255)
colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)

# From matplotlib colormaps
import matplotlib.pyplot as plt
cmap = plt.cm.viridis
colors = (cmap(values)[:, :3] * 255).astype(np.uint8)

# Solid color for all points
rr.log("points", rr.Points3D(positions, colors=[128, 0, 255]))
```
