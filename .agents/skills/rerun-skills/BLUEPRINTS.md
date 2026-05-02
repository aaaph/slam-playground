# Blueprints - Viewer Layout

Detailed guide for creating blueprints to control the Rerun viewer's layout and configuration.

## What are Blueprints?

Blueprints define:
- **Layout**: How views are organized (horizontal, vertical, grid, tabs)
- **Views**: What data is displayed and how (2D, 3D, time series, etc.)
- **Overrides**: Custom styling and behavior per entity
- **Defaults**: Default settings for views

Blueprints are defined programmatically and can be set at startup or sent during runtime.

## Basic Blueprint Structure

```python
import rerun as rr
import rerun.blueprint as rrb

blueprint = rrb.Blueprint(
    # Layout containers
    rrb.Horizontal(
        # Views
        rrb.Spatial3DView(origin="world"),
        rrb.Spatial2DView(origin="camera/image")
    ),
    # Optional: configure panels
    rrb.SelectionPanel(state="collapsed"),
    rrb.TimePanel(state="collapsed")
)

# Apply at startup
rr.script_setup(args, "app_name", default_blueprint=blueprint)

# Or send later
rr.send_blueprint(blueprint)
```

## View Types

### Spatial3DView - 3D Visualization

```python
rrb.Spatial3DView(
    origin="world",  # Root entity path to display
    name="3D Scene"  # Display name
)
```

### Spatial2DView - 2D Visualization

```python
rrb.Spatial2DView(
    origin="camera/image",
    name="Camera View"
)
```

### TimeSeriesView - Time Series Plots

```python
rrb.TimeSeriesView(
    origin="metrics",
    name="Training Metrics"
)
```

### BarChartView - Bar Charts

```python
rrb.BarChartView(
    origin="bar_chart",
    name="Distribution"
)
```

### GraphView - Graph Visualizations

```python
rrb.GraphView(
    origin="graph",
    name="Network"
)
```

### TextDocumentView - Text and Markdown

```python
rrb.TextDocumentView(
    origin="description",
    name="README"
)
```

## Layout Containers

### Horizontal Layout

```python
rrb.Horizontal(
    rrb.Spatial3DView(origin="world"),
    rrb.Spatial2DView(origin="camera"),
    column_shares=[2, 1]  # 2:1 ratio
)
```

### Vertical Layout

```python
rrb.Vertical(
    rrb.Spatial2DView(origin="image"),
    rrb.TimeSeriesView(origin="metrics"),
    row_shares=[3, 1]  # 3:1 ratio
)
```

### Grid Layout

```python
rrb.Grid(
    rrb.Spatial2DView(origin="camera1"),
    rrb.Spatial2DView(origin="camera2"),
    rrb.Spatial2DView(origin="camera3"),
    rrb.Spatial2DView(origin="camera4")
    # Automatically arranged in grid
)
```

### Tabs Layout

```python
rrb.Tabs(
    rrb.Spatial3DView(origin="world", name="3D"),
    rrb.Spatial2DView(origin="camera", name="2D"),
    rrb.TimeSeriesView(origin="metrics", name="Metrics")
)
```

## Complete Layout Examples

### Side-by-Side 3D and 2D

```python
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial3DView(origin="world", name="3D View"),
        rrb.Spatial2DView(origin="camera/image", name="Camera"),
        column_shares=[1, 1]
    )
)
```

### Dashboard Layout

```python
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        # Left: stacked views
        rrb.Vertical(
            rrb.Spatial3DView(origin="world"),
            rrb.Spatial2DView(origin="camera/rgb"),
            row_shares=[2, 1]
        ),
        # Right: metrics and info
        rrb.Vertical(
            rrb.TimeSeriesView(origin="metrics"),
            rrb.TextDocumentView(origin="description"),
            row_shares=[3, 1]
        ),
        column_shares=[3, 1]
    ),
    rrb.SelectionPanel(state="collapsed"),
    rrb.TimePanel(state="expanded")
)
```

### Multi-Camera Grid

```python
blueprint = rrb.Blueprint(
    rrb.Grid(
        rrb.Spatial2DView(origin="camera_0", name="Front"),
        rrb.Spatial2DView(origin="camera_1", name="Left"),
        rrb.Spatial2DView(origin="camera_2", name="Right"),
        rrb.Spatial2DView(origin="camera_3", name="Back")
    )
)
```

### Complex Multi-Section Layout

```python
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        # Left column
        rrb.Vertical(
            # Top: grid of charts
            rrb.Grid(
                rrb.BarChartView(origin="bar_chart", name="Distribution"),
                rrb.TimeSeriesView(origin="curves", name="Curves"),
                rrb.TimeSeriesView(origin="trig", name="Trig"),
                rrb.TimeSeriesView(origin="classification", name="Classification")
            ),
            # Bottom: spiral plot
            rrb.TimeSeriesView(origin="spiral", name="Spiral"),
            row_shares=[2, 1]
        ),
        # Right column: description
        rrb.TextDocumentView(origin="description", name="Info"),
        column_shares=[3, 1]
    ),
    rrb.SelectionPanel(state="collapsed"),
    rrb.TimePanel(state="collapsed")
)
```

## Overrides - Per-Entity Styling

Customize how specific entities appear in a view.

### Time Series Styling

```python
rrb.TimeSeriesView(
    origin="/metrics",
    overrides={
        "metrics/train_loss": rr.SeriesLines.from_fields(
            colors=[255, 0, 0],
            widths=2.0,
            names="Training Loss"
        ),
        "metrics/val_loss": rr.SeriesLines.from_fields(
            colors=[0, 255, 0],
            widths=2.0,
            names="Validation Loss"
        ),
        "metrics/samples": rr.SeriesPoints()  # Force scatter plot
    }
)
```

### Graph Force Configuration

```python
from rerun.blueprint.archetypes import ForceLink, ForceManyBody, ForceCollisionRadius

rrb.GraphView(
    origin="graph",
    force_link=ForceLink(distance=60),
    force_many_body=ForceManyBody(strength=-60),
    force_collision_radius=ForceCollisionRadius(enabled=True)
)
```

### Visible Time Ranges

```python
rrb.Spatial3DView(
    origin="/",
    overrides={
        "world/particles": rrb.VisibleTimeRanges(
            timeline="stable_time",
            start=rrb.TimeRangeBoundary.cursor_relative(seconds=-0.3),
            end=rrb.TimeRangeBoundary.cursor_relative(seconds=0.3)
        )
    }
)
```

### View Defaults

```python
rrb.GraphView(
    origin="bubble_chart",
    defaults=[
        rr.GraphNodes.from_fields(show_labels=False, radii=10)
    ]
)
```

## Panel Configuration

### Selection Panel

```python
# Collapsed by default
rrb.SelectionPanel(state="collapsed")

# Expanded by default
rrb.SelectionPanel(state="expanded")

# Hidden
rrb.SelectionPanel(state="hidden")
```

### Time Panel

```python
# Collapsed timeline
rrb.TimePanel(state="collapsed")

# Expanded timeline
rrb.TimePanel(state="expanded")

# Hidden timeline
rrb.TimePanel(state="hidden")
```

## Real-World Examples

### Object Detection Application

```python
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        # Video with detections
        rrb.Spatial2DView(
            origin="video",
            name="Detections"
        ),
        # Metrics
        rrb.Vertical(
            rrb.TimeSeriesView(
                origin="metrics/detections",
                name="Detection Count"
            ),
            rrb.TextDocumentView(
                origin="description",
                name="Info"
            ),
            row_shares=[2, 1]
        ),
        column_shares=[3, 1]
    )
)
```

### 3D Reconstruction Pipeline

```python
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        # 3D view
        rrb.Spatial3DView(
            origin="world",
            name="Reconstruction"
        ),
        # Input images and metrics
        rrb.Vertical(
            rrb.Grid(
                rrb.Spatial2DView(origin="camera_0", name="Cam 0"),
                rrb.Spatial2DView(origin="camera_1", name="Cam 1"),
                rrb.Spatial2DView(origin="camera_2", name="Cam 2"),
                rrb.Spatial2DView(origin="camera_3", name="Cam 3")
            ),
            rrb.TimeSeriesView(
                origin="metrics/reconstruction",
                name="Progress"
            ),
            row_shares=[3, 1]
        ),
        column_shares=[2, 1]
    )
)
```

### Training Dashboard

```python
blueprint = rrb.Blueprint(
    rrb.Vertical(
        # Metrics row
        rrb.Horizontal(
            rrb.TimeSeriesView(
                origin="training/loss",
                name="Loss",
                overrides={
                    "training/loss": rr.SeriesLines.from_fields(
                        colors=[255, 0, 0],
                        names="Train"
                    ),
                    "validation/loss": rr.SeriesLines.from_fields(
                        colors=[0, 255, 0],
                        names="Val"
                    )
                }
            ),
            rrb.TimeSeriesView(
                origin="training/accuracy",
                name="Accuracy",
                overrides={
                    "training/accuracy": rr.SeriesLines.from_fields(
                        colors=[255, 0, 0]
                    ),
                    "validation/accuracy": rr.SeriesLines.from_fields(
                        colors=[0, 255, 0]
                    )
                }
            )
        ),
        # Samples row
        rrb.Grid(
            rrb.Spatial2DView(origin="samples/0", name="Sample 1"),
            rrb.Spatial2DView(origin="samples/1", name="Sample 2"),
            rrb.Spatial2DView(origin="samples/2", name="Sample 3"),
            rrb.Spatial2DView(origin="samples/3", name="Sample 4")
        ),
        row_shares=[1, 2]
    ),
    rrb.TimePanel(state="expanded")
)
```

### Graph Visualization Comparison

```python
blueprint = rrb.Blueprint(
    rrb.Grid(
        rrb.GraphView(
            origin="node_link",
            name="Force Layout",
            force_link=ForceLink(distance=60),
            force_many_body=ForceManyBody(strength=-60)
        ),
        rrb.GraphView(
            origin="bubble_chart",
            name="Bubble Chart",
            force_link=ForceLink(enabled=False),
            force_many_body=ForceManyBody(enabled=False),
            force_collision_radius=ForceCollisionRadius(enabled=True),
            defaults=[rr.GraphNodes.from_fields(show_labels=False)]
        ),
        rrb.GraphView(
            origin="lattice",
            name="Lattice",
            force_link=ForceLink(distance=60),
            force_many_body=ForceManyBody(strength=-60),
            defaults=[rr.GraphNodes.from_fields(show_labels=False, radii=10)]
        ),
        rrb.Horizontal(
            rrb.GraphView(
                origin="markov_chain",
                name="Markov Chain"
            ),
            rrb.TextDocumentView(origin="description", name="Description")
        )
    )
)
```

## Dynamic Blueprint Updates

You can send blueprints at any time:

```python
# Initial blueprint
rr.script_setup(args, "app", default_blueprint=simple_blueprint)

# ... do some work ...

# Switch to detailed blueprint
rr.send_blueprint(detailed_blueprint)

# ... do more work ...

# Switch back
rr.send_blueprint(simple_blueprint)
```

## Blueprint Best Practices

### 1. Match Data Hierarchy

Organize views to match your entity hierarchy:

```python
# If your entities are:
# world/lidar/points
# world/camera/rgb
# world/detections/boxes

# Blueprint should reflect this:
rrb.Spatial3DView(origin="world", name="Scene")
```

### 2. Use Descriptive Names

```python
# Good
rrb.TimeSeriesView(origin="metrics/loss", name="Training Loss")

# Less clear
rrb.TimeSeriesView(origin="metrics/loss", name="View 1")
```

### 3. Size Views by Importance

```python
rrb.Horizontal(
    main_view,  # Important
    sidebar,    # Less important
    column_shares=[4, 1]  # 4:1 ratio emphasizes main view
)
```

### 4. Group Related Views

```python
rrb.Vertical(
    # All training metrics together
    rrb.TimeSeriesView(origin="train/loss"),
    rrb.TimeSeriesView(origin="train/accuracy"),
    rrb.TimeSeriesView(origin="train/lr")
)
```

### 5. Consider User Workflow

```python
# Left to right workflow: input -> processing -> output
rrb.Horizontal(
    rrb.Spatial2DView(origin="input/image", name="Input"),
    rrb.Spatial2DView(origin="processing/intermediate", name="Processing"),
    rrb.Spatial2DView(origin="output/result", name="Output")
)
```

### 6. Collapse Panels for Cleaner View

```python
# For presentations or recorded demos
blueprint = rrb.Blueprint(
    layout,
    rrb.SelectionPanel(state="collapsed"),
    rrb.TimePanel(state="collapsed")
)
```
