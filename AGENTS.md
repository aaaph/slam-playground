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

## Agent Mode Pipeline Runs

Use `slam_agent_profile` for automated agent runs. It runs `pipeline/slam-dataflow.yml` on `euroc_mh_01`, starts automatically after all nodes are ready, processes 5% of the dataset, stops after the dataset slice is complete, and writes Rerun output to a file instead of opening the app.

Run it as:

```bash
just pipeline run --profile slam_agent_profile
```

Do not include the `.yaml` suffix in the profile name. The CLI resolves profile names under `config/profile/` and appends `.yaml` itself, so `--profile slam_agent_profile.yaml` will look for `slam_agent_profile.yaml.yaml`.

The dataset selected by a profile can be overwritten from the CLI. Use a supported dataset selector from `just dataset list`:

```bash
just pipeline run --profile slam_agent_profile --dataset euroc_mh_01
```

Before a fresh run, `just logs clear` can be used to delete generated files under `pipeline/out` while keeping the directory itself. After a run, read `pipeline/out/current-run.json` first. The latest run directory contains:

- `log_*.txt` files for node logs.
- `data.rrd` for the Rerun recording.
- `rerun_manifest.json` with an agent-readable `stream_index` mapping configured streams to logged Rerun entity paths.
- `rerun_config.json` with the resolved Rerun visualization config.

## CLI Surfaces

Pipeline execution is profile-based. Profiles live in `config/profile/*.yaml` and compose dataset selection, dataflow template, visualization sink, and run mode. Use the profile CLI to inspect the fully resolved runtime dataflow before running:

```bash
just profile resolve --profile slam_agent_profile
```

Use the pipeline CLI to materialize and launch the resolved dataflow:

```bash
just pipeline run --profile slam_agent_profile
```

Profile names are logical names, not file paths, so pass `slam_agent_profile` rather than `slam_agent_profile.yaml`.

Dataset management has its own CLI surface:

```bash
just dataset list
```

Use it to inspect supported dataset manifests and their validation state before choosing a profile or overriding `--dataset`.

## Visualizer Notes

Rerun is the main visualization path.
The main Rerun viewer configuration is `./config/rerun_view_config.yaml`; `pipeline/vio-dataflow.yml` passes it through `VISUALIZE_CONFIG`, and `RerunConfigLoader` defaults to `config/rerun_view_config.yaml`.
Rerun visualization is config-driven:

- `src/visualizer/rerun/schemas.py` defines valid view, layout, and module types.
- `src/visualizer/rerun/factories/rerun_config_factory.py` builds the Rerun blueprint and instantiates modules from YAML streams.
- `src/visualizer/rerun/rerun_vizualizer.py` owns Rerun initialization, timeline setup, and per-frame module dispatch.
- `src/visualizer/rerun/modules/*` contains one module per visualization type, all implementing `IVizModule`.

When adding a new Rerun view that uses existing visualization modules:

1. Add a dedicated YAML file under `config/visualization/rerun/views/<view_name>.yaml`.
2. Include that file from the active top-level visualization config, such as `config/visualization/slam_view_config.yaml` or `config/visualization/vio_view_config.yaml`.
3. Set each stream `branch` to the exact branch name emitted by `RerunNode` (for example `frontend_frame`, `fixedlag_frame`, `trajectory_evaluator_frame`). If this is a new upstream node output, wire it in the dataflow YAML as a `rerun` input and add a matching `@on_input` handler in `src/pipeline/nodes/rerun_node.py`.
4. Set each stream `id` to the `PipelineContext` field name logged by the producing node, and set `entity` to the Rerun entity path where it should appear.
5. Prefer reusing existing modules (`dynamic_transform`, `static_transform`, `pointcloud`, `plot_scalar`, `plot_3d_vector`, `image`, `features`, `trajectory`) before adding a new module type.
6. Validate the config with `RerunConfigLoader`/`RerunConfigFactory` or the relevant Rerun tests before running the full pipeline.

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
- Install third-party native deps: `just install-3rdparty`
- Install only GTSAM: `just install-gtsam`
- Install only PyDBoW3: `just install-pydbow3`

## Native Dependencies

GTSAM and PyDBoW3 are not tracked by `uv.lock`.
After `uv sync`, run:

```bash
just install-3rdparty
```
