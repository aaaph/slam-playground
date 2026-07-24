# Repository Instructions

## Project Context

This repository is a SLAM playground: a research workspace for building visual-inertial navigation systems, experimenting with SLAM ideas, and studying SLAM by implementing it directly.
Python version: 3.13.
Package manager: uv.
Task runner: just.

## Technical Stack

- dora-rs with Zenoh and HTTP control transports for multiprocess and async architecture.
- Rerun for visualization.
- NumPy, Arrow, and SciPy as the core Python data and scientific computing stack.
- Front-end: FAST feature detection, LK optical flow, forward/backward checks, essential matrix RANSAC, stereo LK matching, region-based retracking, keyframe selection, IMU preintegration, zero-velocity/static initialization, and PnP feedback from the backend map.
- Back-end: GTSAM factor graph optimization with explicit landmarks, `IncrementalFixedLagSmoother`, state keys `X(k)` for pose, `V(k)` for velocity, `B(k)` for IMU bias, and `L(id)` for landmarks; factors include pose/velocity/bias priors, IMU factors, bias random walk, stereo projection factors, and ZUPT velocity priors.
- LCD/PGO: ORB/DBoW3 visual place recognition, loop-closure verification, pose graph optimization, and Rerun/OpenCV loop-closure visualization.

## Repository Map

- `src/core` is the algorithmic SLAM/VIO layer: camera model, feature tracker, frontend logic, pose tracking, GTSAM graph optimizer, and transformations.
- `src/pipeline` is the async/event plumbing: dora node decorators, `PipelineContext`, and Arrow serialization.
- `src/pipeline/nodes` contains runnable dora nodes.
- `pipeline/*.yml` defines runnable dataflows.
- `src/dataset` loads manifest-backed datasets from `datasets/*.yaml`, currently focused on EuRoC selectors such as `euroc_v101` and smoke/test variants.
- `src/visualizer` has Rerun as the active visualization path, with older or secondary Foxglove support.

## Primary Pipeline

The main SLAM dataflow is:

```text
control (HTTP or Zenoh) -> dataset -> vio_frontend -> vio_backend -> loop_closure -> pgo -> slam_output -> trajectory_evaluator/rerun
```

Use `pipeline/slam-dataflow.yml` for the full SLAM pipeline. Use `pipeline/vio-dataflow.yml` when only the VIO path is needed.

## Pipeline Data Contract

Dora nodes pass state through `PipelineContext`, backed by `pyarrow.StructArray`.
Images and ndarrays are flattened into Arrow arrays.
Complex payloads, such as keyframes and metrics, are passed as Arrow `RecordBatch` values.

## Code Generation Style

- Keep defensive checks and validation minimal in core code. Internal module boundaries are strict contracts: responsibility for building correct arrays and satisfying shapes, dtypes, and finite-value expectations belongs to the developer and the caller that assembles the data.
- Prefer SoA/table-first data flow. When adding frame-level data, assembling a complete frame table with the expected schema has higher priority than passing loosely described column contracts around and rebuilding tables inside downstream methods.
- Private contracts that organize implementation details inside one module, and are not intended as contracts for using that module, may be marked with a leading underscore, such as `_SomePrivateContract` or `_MotionOnlyBaResult`.

## Pipeline Run State

When debugging pipeline logs, first read `pipeline/out/current-run.json`.
The control node updates this runtime manifest at pipeline start and stop; it points to the current `pipeline/out/<run-id>` log directory and lists known `log_*.txt` files.
`pipeline/out/latest` is also maintained as a symlink to the same log directory.
If the manifest is missing, fall back to the newest UUID-named directory in `pipeline/out` by modification time.

## Agent Mode Pipeline Runs

Use `slam_agent_profile` for automated agent runs. It runs `pipeline/slam-dataflow.yml` on its configured dataset, currently `euroc_v201`, starts automatically after all nodes are ready, processes 5% of the dataset, stops after the dataset slice is complete, and writes Rerun output to a file instead of opening the app.

Run it as:

```bash
just pipeline run --profile slam_agent_profile
```

Do not include the `.yaml` suffix in the profile name. The CLI resolves profile names under `config/profile/` and appends `.yaml` itself, so `--profile slam_agent_profile.yaml` will look for `slam_agent_profile.yaml.yaml`.

The dataset selected by a profile can be overwritten from the CLI. Use a supported dataset selector from `just dataset list`:

```bash
just pipeline run --profile slam_agent_profile --dataset euroc_v101
```

The agent run fraction can also be overwritten directly from the CLI; do not search `--help` just to remember this flag. For a smaller automated run, pass the fraction explicitly:

```bash
just pipeline run --profile slam_agent_profile --dataset euroc_v101 --fraction 0.05
```

For a full dataset run, keep `slam_agent_profile` but override the fraction to `1`:

```bash
just pipeline run --profile slam_agent_profile --dataset euroc_v101 --fraction 1
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

Control commands are sent through the running control node. HTTP is the default command transport, and `--http` can be passed explicitly:

```bash
just pipeline step 20% --http
just pipeline start --http
just pipeline stop --http
```

Use `--zenoh` for the Zenoh command transport when the profile/run is configured for Zenoh control.

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

Dataset selection is manifest-based. `DatasetRegistry` reads `datasets/*.yaml`; use `just dataset list` to inspect supported selectors and local validation state.
Common EuRoC selectors include `euroc_v101`, `euroc_v102`, `euroc_v103`, `euroc_v201`, `euroc_v202`, `euroc_v203`, and `euroc_smoke`.
Dataset manifests provide the dataset root, stream CSV paths, cache path, and rig file, usually `config/dataset_rig/euroc.yaml`.
The EuRoC loader synchronizes stereo frames with IMU batches between frames.
Timestamps are nanoseconds.

## Known Footguns

- Some older profiles or notes can reference dataset selectors that are not present in local `datasets/*.yaml`; verify selectors with `just dataset list` before running.
- `pydbow3` is still a local/manual native dependency for loop closure and can disappear after `uv sync`; reinstall it with `just install-pydbow3` or `just install-3rdparty`.
- Control commands must match the configured control transport: use `--http` for HTTP control and `--zenoh` for Zenoh control.

## Important Commands

- Sync dependencies: `just dev-sync`
- Sync + reinstall local native deps: `just dev-sync-native`
- Run tests: `just test`
- Install third-party native deps: `just install-3rdparty`
- Install only PyDBoW3: `just install-pydbow3`

## Native Dependencies

GTSAM is managed by uv through the `gtsam-develop` dependency in `pyproject.toml`/`uv.lock`; add or update it with `uv add` and install it with `uv sync`.
PyDBoW3 is not tracked by uv and remains a local/manual native binding. After `uv sync`, run this when loop closure needs PyDBoW3 or the binding disappeared:

```bash
just install-pydbow3
```
