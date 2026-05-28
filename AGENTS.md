# Repository Instructions

## Project Context

This repository is a SLAM playground: a research workspace for building visual-inertial navigation systems, experimenting with SLAM ideas, and studying SLAM by implementing it directly.
Python version: 3.13.
Package manager: uv.
Task runner: just.

## Technical Stack

- dora-rs and zenoh for multiprocess and async architecture.
- Rerun for visualization.
- NumPy, Arrow, and SciPy as the core Python data and scientific computing stack.
- Front-end: FAST feature detection, LK optical flow, forward/backward checks, essential matrix RANSAC, stereo LK matching, region-based retracking, keyframe selection, IMU preintegration, zero-velocity/static initialization, and PnP feedback from the backend map.
- Back-end: GTSAM factor graph optimization with explicit landmarks, `IncrementalFixedLagSmoother`, state keys `X(k)` for pose, `V(k)` for velocity, `B(k)` for IMU bias, and `L(id)` for landmarks; factors include pose/velocity/bias priors, IMU factors, bias random walk, stereo projection factors, and ZUPT velocity priors.

## Repository Map

- `src/core` is the algorithmic SLAM/VIO layer: camera model, feature tracker, frontend logic, pose tracking, GTSAM graph optimizer, and transformations.
- `src/pipeline` is the async/event plumbing: dora node decorators, `PipelineContext`, and Arrow serialization.
- `src/pipeline/nodes` contains runnable dora nodes.
- `pipeline/*.yml` defines runnable dataflows.
- `src/dataset` is currently centered on EuRoC `MH_01_easy`.
- `src/visualizer` has Rerun as the active visualization path, with older or secondary Foxglove support.

## Primary Pipeline

The main VIO dataflow is:

```text
zenoh control -> dataset -> vio_frontend -> vio_backend -> rerun
```

Use `pipeline/vio-dataflow.yml` as the primary VIO pipeline entrypoint.

## Pipeline Data Contract

Dora nodes pass state through `PipelineContext`, backed by `pyarrow.StructArray`.
Images and ndarrays are flattened into Arrow arrays.
Complex payloads, such as keyframes and metrics, are passed as Arrow `RecordBatch` values.

## Pipeline Run State

When debugging pipeline logs, first read `pipeline/out/current-run.json`.
The control node updates this runtime manifest at pipeline start and stop; it points to the current `pipeline/out/<run-id>` log directory and lists known `log_*.txt` files.
`pipeline/out/latest` is also maintained as a symlink to the same log directory.
If the manifest is missing, fall back to the newest UUID-named directory in `pipeline/out` by modification time.

## Visualizer Notes

Rerun is the main visualization path.
The main Rerun viewer configuration is `./config/rerun_view_config.yaml`; `pipeline/vio-dataflow.yml` passes it through `VISUALIZE_CONFIG`, and `RerunConfigLoader` defaults to `config/rerun_view_config.yaml`.
Rerun visualization is config-driven:

- `src/visualizer/rerun/schemas.py` defines valid view, layout, and module types.
- `src/visualizer/rerun/factories/rerun_config_factory.py` builds the Rerun blueprint and instantiates modules from YAML streams.
- `src/visualizer/rerun/rerun_vizualizer.py` owns Rerun initialization, timeline setup, and per-frame module dispatch.
- `src/visualizer/rerun/modules/*` contains one module per visualization type, all implementing `IVizModule`.

When adding a new Rerun visualization type, add a dedicated module under `src/visualizer/rerun/modules`, register its `ModuleType` in `schemas.py`, and add it to `MODULE_CLASS_MAP` in `rerun_config_factory.py` before using it in `rerun_view_config.yaml`.

## Dataset Assumptions

The main dataset path is `datasets/euroc_v_01_easy`.
The current loader is built around EuRoC `MH_01_easy` and synchronizes stereo frames with IMU batches between frames.
Timestamps are nanoseconds.
Sensor config paths:

- cam0: `./datasets/euroc_v_01_easy/cam0/sensor.yaml`
- cam1: `./datasets/euroc_v_01_easy/cam1/sensor.yaml`
- imu0: `./datasets/euroc_v_01_easy/imu0/sensor.yaml`

## Known Footguns

- `pipeline/my-slam-dataflow.yml` references `sliding_window_back_end_node.py`, which is not present in `src/pipeline/nodes`.
- Native `gtsam` and `pydbow3` are local/manual dependencies and can disappear after `uv sync`.

## Important Commands

- Sync dependencies: `just dev-sync`
- Sync + reinstall native deps: `just dev-sync-native`
- Run tests: `just test`
- Install GTSAM: `just install-gtsam`
- Install PyDBoW3: `just install-pydbow3`

## Native Dependencies

GTSAM and PyDBoW3 are not tracked by `uv.lock`.
After `uv sync`, run:

```bash
just native-deps
```
