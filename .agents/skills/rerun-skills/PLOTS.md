# Plots and Time Series

Detailed guide for visualizing temporal data, time series, and charts with Rerun.

## Scalars - Time Series Data

Log scalar values over time to create line plots and time series visualizations.

### Single Scalar Value

```python
import rerun as rr

# Log single metric over time
for t in range(1000):
    rr.set_time("frame_nr", sequence=t)
    loss = compute_loss()
    rr.log("metrics/loss", rr.Scalars(loss))
```

### Multiple Scalars (Multi-Line Plot)

```python
# Log multiple values as separate lines
for epoch in range(100):
    rr.set_time("epoch", sequence=epoch)

    train_acc, val_acc = compute_accuracies()

    # Method 1: Log as array
    rr.log("metrics/accuracy", rr.Scalars([train_acc, val_acc]))

    # Method 2: Log separately
    rr.log("metrics/train_acc", rr.Scalars(train_acc))
    rr.log("metrics/val_acc", rr.Scalars(val_acc))
```

### Mathematical Functions

```python
import math

# Plot trigonometric functions
for t in range(int(2 * math.pi * 100)):
    rr.set_time("frame_nr", sequence=t)

    sin_val = math.sin(float(t) / 100.0)
    rr.log("trig/sin", rr.Scalars(sin_val))

    cos_val = math.cos(float(t) / 100.0)
    rr.log("trig/cos", rr.Scalars(cos_val))
```

### Parabola Example

```python
# Log a parabola over time
for t in range(0, 1000, 10):
    rr.set_time("frame_nr", sequence=t)

    f_of_t = (t * 0.01 - 5) ** 3 + 1
    rr.log("curves/parabola", rr.Scalars(f_of_t))
```

## SeriesLines - Line Plot Styling

Style time series as connected lines with custom colors and widths.

### Basic Line Styling

```python
# Log data with line styling
rr.log("curves/parabola", rr.Scalars(value),
       rr.SeriesLines(widths=2.0, colors=[255, 0, 0]))
```

### Static Styling

```python
# Set styling once, applies to all subsequent data
rr.log("curves/parabola",
    rr.SeriesLines(names="f(t) = (0.01t - 3)³ + 1"),
    static=True
)

# Now log data without styling
for t in range(0, 1000, 10):
    rr.set_time("frame_nr", sequence=t)
    f_of_t = (t * 0.01 - 5) ** 3 + 1
    rr.log("curves/parabola", rr.Scalars(f_of_t))
```

### Dynamic Styling

```python
# Change color and width based on value
for t in range(0, 1000, 10):
    rr.set_time("frame_nr", sequence=t)

    f_of_t = (t * 0.01 - 5) ** 3 + 1
    width = np.clip(abs(f_of_t) * 0.1, 0.5, 10.0)

    # Color changes based on thresholds
    if f_of_t < -10.0:
        color = [255, 0, 0]  # Red
    elif f_of_t > 10.0:
        color = [0, 255, 0]  # Green
    else:
        color = [255, 255, 0]  # Yellow

    rr.log("curves/parabola",
        rr.Scalars(f_of_t),
        rr.SeriesLines(widths=width, colors=color)
    )
```

### Using Blueprint Overrides

```python
import rerun.blueprint as rrb

# Style via blueprint instead of logging
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(
        origin="/trig",
        overrides={
            "/trig/sin": rr.SeriesLines.from_fields(
                colors=[255, 0, 0],
                names="sin(0.01t)"
            ),
            "/trig/cos": rr.SeriesLines.from_fields(
                colors=[0, 255, 0],
                names="cos(0.01t)"
            )
        }
    )
)
```

## SeriesPoints - Scatter Plot Styling

Style time series as scatter points with custom markers and sizes.

### Basic Scatter Points

```python
# Log data as scatter points
for t in range(0, 1000, 2):
    rr.set_time("frame_nr", sequence=t)

    value = compute_value(t)
    rr.log("data/samples",
        rr.Scalars(value),
        rr.SeriesPoints(marker_sizes=5)
    )
```

### Classification Example

```python
import random

# Scatter plot with color-coded classification
for t in range(0, 1000, 2):
    rr.set_time("frame_nr", sequence=t)

    # Decision boundary
    f_of_t = (2 * 0.01 * t) + 2
    rr.log("classification/line", rr.Scalars(f_of_t))

    # Sample with noise
    g_of_t = f_of_t + random.uniform(-5.0, 5.0)

    # Color by classification
    if g_of_t < f_of_t - 1.5:
        color = [255, 0, 0]  # Below threshold
    elif g_of_t > f_of_t + 1.5:
        color = [0, 255, 0]  # Above threshold
    else:
        color = [255, 255, 255]  # Within threshold

    marker_size = abs(g_of_t - f_of_t)

    rr.log("classification/samples",
        rr.Scalars(g_of_t),
        rr.SeriesPoints(colors=color, marker_sizes=marker_size)
    )
```

### Using Blueprint Overrides

```python
# Force SeriesPoints visualizer via blueprint
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(
        origin="/classification",
        overrides={
            "classification/line": rr.SeriesLines.from_fields(
                colors=[255, 255, 0],
                widths=3.0
            ),
            "classification/samples": rr.SeriesPoints()  # Force scatter
        }
    )
)
```

## BarChart - Bar Charts

Visualize data as bar charts.

### Basic Bar Chart

```python
import numpy as np

# Gaussian distribution as bar chart
mean = 0
std = 1
variance = np.square(std)
x = np.arange(-5, 5, 0.1)
y = np.exp(-np.square(x - mean) / 2 * variance) / (np.sqrt(2 * np.pi * variance))

rr.set_time("frame_nr", sequence=0)
rr.log("bar_chart", rr.BarChart(y))
```

### Histogram

```python
# Create histogram from data
data = np.random.randn(1000)
counts, bins = np.histogram(data, bins=50)

rr.log("histogram", rr.BarChart(counts))
```

### Time-Varying Bar Chart

```python
# Animated bar chart
for frame in range(100):
    rr.set_time("frame", sequence=frame)

    # Generate random bars
    values = np.random.rand(10) * frame / 10
    rr.log("bars", rr.BarChart(values))
```

## send_columns - Efficient Batch Logging

For large amounts of time-series data, use columnar logging for efficiency.

### Basic Usage

```python
import numpy as np

# Generate time series data
times = np.arange(1000)
values = np.sin(times / 100.0)

# Log all at once
rr.send_columns(
    "timeseries/signal",
    indexes=[rr.TimeColumn("frame_nr", sequence=times)],
    columns=[*rr.Scalars.columns(scalars=values)]
)
```

### Multiple Series

```python
# Two time series (e.g., X and Y components)
times = np.arange(int(2 * np.pi * 100))
theta = times / 100.0

x = theta * np.cos(theta)
y = theta * np.sin(theta)

# Column-major format: (num_samples, num_series)
scalars = np.array((x, y)).T

rr.send_columns(
    "spiral",
    indexes=[rr.TimeColumn("frame_nr", sequence=times)],
    columns=[*rr.Scalars.columns(scalars=scalars)]
)
```

### Multiple Timelines

```python
# Log with multiple time indices
times_sequence = np.arange(1000)
times_seconds = times_sequence * 0.01  # 10ms per sample

rr.send_columns(
    "sensor/data",
    indexes=[
        rr.TimeColumn("frame_nr", sequence=times_sequence),
        rr.TimeColumn("timestamp", duration=times_seconds)
    ],
    columns=[*rr.Scalars.columns(scalars=sensor_values)]
)
```

### Performance Benefit

`send_columns` is significantly faster than logging individual samples in a loop:

```python
# Slow - 1000 API calls
for t in range(1000):
    rr.set_time("frame", sequence=t)
    rr.log("metric", rr.Scalars(values[t]))

# Fast - 1 API call
rr.send_columns(
    "metric",
    indexes=[rr.TimeColumn("frame", sequence=np.arange(1000))],
    columns=[*rr.Scalars.columns(scalars=values)]
)
```

## Complete Examples

### Training Metrics Dashboard

```python
# Training loop with metrics
for epoch in range(100):
    # Training phase
    train_losses = []
    train_accs = []

    for batch in train_loader:
        loss, acc = train_step(batch)
        train_losses.append(loss)
        train_accs.append(acc)

    # Validation phase
    val_loss, val_acc = validate()

    # Log epoch metrics
    rr.set_time("epoch", sequence=epoch)

    rr.log("training/loss", rr.Scalars(np.mean(train_losses)))
    rr.log("training/accuracy", rr.Scalars(np.mean(train_accs)))
    rr.log("validation/loss", rr.Scalars(val_loss))
    rr.log("validation/accuracy", rr.Scalars(val_acc))

    # Log learning rate
    lr = optimizer.param_groups[0]['lr']
    rr.log("training/learning_rate", rr.Scalars(lr))
```

### Multi-Line Plot with Legend

```python
# Plot multiple related metrics
for t in range(1000):
    rr.set_time("step", sequence=t)

    # Different components
    x_vel = compute_x_velocity(t)
    y_vel = compute_y_velocity(t)
    z_vel = compute_z_velocity(t)

    rr.log("velocity/x", rr.Scalars(x_vel))
    rr.log("velocity/y", rr.Scalars(y_vel))
    rr.log("velocity/z", rr.Scalars(z_vel))

# Configure display with blueprint
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(
        origin="/velocity",
        overrides={
            "velocity/x": rr.SeriesLines.from_fields(colors=[255, 0, 0], names="X"),
            "velocity/y": rr.SeriesLines.from_fields(colors=[0, 255, 0], names="Y"),
            "velocity/z": rr.SeriesLines.from_fields(colors=[0, 0, 255], names="Z")
        }
    )
)
```

### IMU Sensor Data

```python
# Log IMU accelerometer and gyroscope data
for timestamp, imu_reading in imu_data:
    rr.set_time("timestamp", timestamp=timestamp)

    # Accelerometer (3 components)
    rr.log("sensors/imu/accel", rr.Scalars([
        imu_reading.accel_x,
        imu_reading.accel_y,
        imu_reading.accel_z
    ]))

    # Gyroscope (3 components)
    rr.log("sensors/imu/gyro", rr.Scalars([
        imu_reading.gyro_x,
        imu_reading.gyro_y,
        imu_reading.gyro_z
    ]))

    # Magnitude
    accel_mag = np.linalg.norm([
        imu_reading.accel_x,
        imu_reading.accel_y,
        imu_reading.accel_z
    ])
    rr.log("sensors/imu/accel_magnitude", rr.Scalars(accel_mag))
```

### Comparing Algorithms

```python
# Compare performance of different algorithms
algorithms = ["algo_a", "algo_b", "algo_c"]

for iteration in range(1000):
    rr.set_time("iteration", sequence=iteration)

    for algo_name in algorithms:
        metric = run_algorithm(algo_name, iteration)
        rr.log(f"comparison/{algo_name}", rr.Scalars(metric))

# Style with blueprint
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(
        origin="/comparison",
        overrides={
            "comparison/algo_a": rr.SeriesLines.from_fields(
                colors=[255, 0, 0],
                names="Algorithm A"
            ),
            "comparison/algo_b": rr.SeriesLines.from_fields(
                colors=[0, 255, 0],
                names="Algorithm B"
            ),
            "comparison/algo_c": rr.SeriesLines.from_fields(
                colors=[0, 0, 255],
                names="Algorithm C"
            )
        }
    )
)
```

## Tips and Best Practices

### Choosing Between Lines and Points

- Use **SeriesLines** for continuous data (sensor readings, smooth functions)
- Use **SeriesPoints** for discrete samples (events, classifications)
- Mix both for complex visualizations

### Time Granularity

```python
# High-frequency data - use send_columns
rr.send_columns("high_freq", ...)

# Low-frequency data - use individual logs
rr.set_time("event", sequence=i)
rr.log("low_freq", rr.Scalars(value))
```

### Organizing Metrics

Use hierarchical entity paths:

```python
# Good organization
rr.log("model/training/loss", ...)
rr.log("model/training/accuracy", ...)
rr.log("model/validation/loss", ...)
rr.log("model/validation/accuracy", ...)
rr.log("model/hyperparams/learning_rate", ...)

# Creates clear hierarchy in viewer
```

### Multiple Y-Axes

Different metrics with different scales work automatically in Rerun:

```python
# These will auto-scale independently
rr.log("metrics/loss", rr.Scalars(loss))  # Range: 0-10
rr.log("metrics/accuracy", rr.Scalars(acc))  # Range: 0-1
rr.log("metrics/learning_rate", rr.Scalars(lr))  # Range: 1e-5 to 1e-2
```

### Debugging Values

```python
# Log intermediate values for debugging
rr.log("debug/gradient_norm", rr.Scalars(grad_norm))
rr.log("debug/weight_mean", rr.Scalars(weights.mean()))
rr.log("debug/weight_std", rr.Scalars(weights.std()))
```
