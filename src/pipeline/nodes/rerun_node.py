import json
import os
from pathlib import Path

import numpy as np
from dora import Node
from numpy.typing import NDArray

from logger import spawn_logger
from pipeline.annotations import (
    EXECUTION_TIME_MS_METADATA_FIELD,
    Ctx,
    ExecutionTimeMetadata,
)
from pipeline.decorators import on_input, on_stop, reactive
from pipeline.nodes.base import PipelineNode
from pipeline.runtime_config import RerunNodeRuntimeConfig
from visualizer.rerun.factories.rerun_config_factory import RerunConfigFactory
from visualizer.rerun.loaders import RerunConfigLoader
from visualizer.rerun.recording_manifest import build_rerun_stream_index
from visualizer.rerun.schemas import RerunConfigSchema

type Vector3 = NDArray[np.float32]


@reactive
class RerunNode(PipelineNode):
    """Rerun vizualization node."""

    def __init__(
        self,
        config: RerunConfigSchema,
        resolution: tuple[int, int],
        runtime_config: RerunNodeRuntimeConfig | None = None,
    ) -> None:
        """Initialize the rerun node."""
        self.node = Node()
        self.node_runtime_config = runtime_config or self.runtime_config_as(RerunNodeRuntimeConfig)
        self.config = config
        self.config.app_name = f"rerun_{self.node.dataflow_id()}"

        self.config.resolution = resolution

        self.logger = spawn_logger(app="rerun_node")
        self.save_path = self.resolve_save_path(self.node_runtime_config)
        self.write_recording_artifacts(self.save_path)

        self.vizualizer = RerunConfigFactory.from_config(
            self.config,
            spawn=self.node_runtime_config.spawn_viewer,
            save_path=self.save_path,
            enabled=self.node_runtime_config.enabled,
        )

        self.logger.info(self.vizualizer.info())
        self.vizualize = self.vizualizer.pipeline_generator()

    @on_input("dataset_frame")
    def handle_dataset_frame(
        self,
        ctx: Ctx,
        execution_time_metadata: ExecutionTimeMetadata,
    ) -> None:
        """Handle the ctx event. Should visualize the context."""
        self.visualize_branch(
            "dataset_frame",
            ctx,
            execution_time_metadata,
        )

    @on_input("frontend_frame")
    def handle_frontend_frame(
        self,
        ctx: Ctx,
        execution_time_metadata: ExecutionTimeMetadata,
    ) -> None:
        """Handle the ctx event. Should visualize the context."""
        self.visualize_branch(
            "frontend_frame",
            ctx,
            execution_time_metadata,
        )

    @on_input("fixedlag_frame")
    def handle_fixedlag_frame(
        self,
        ctx: Ctx,
        execution_time_metadata: ExecutionTimeMetadata,
    ) -> None:
        """Handle the ctx event. Should visualize the context."""
        self.visualize_branch(
            "fixedlag_frame",
            ctx,
            execution_time_metadata,
        )

    @on_input("tracker_frame")
    def handle_tracker_frame(
        self,
        ctx: Ctx,
        execution_time_metadata: ExecutionTimeMetadata,
    ) -> None:
        """Handle the ctx event. Should visualize the context."""
        self.visualize_branch(
            "frontend_frame",
            ctx,
            execution_time_metadata,
        )

    @on_input("loopclosure_frame")
    def handle_loopclosure_frame(
        self,
        ctx: Ctx,
        execution_time_metadata: ExecutionTimeMetadata,
    ) -> None:
        """Handle the ctx event. Should visualize the context."""
        self.visualize_branch(
            "loopclosure_frame",
            ctx,
            execution_time_metadata,
        )

    @on_input("pgo_frame")
    def handle_pgo_frame(
        self,
        ctx: Ctx,
        execution_time_metadata: ExecutionTimeMetadata,
    ) -> None:
        """Handle the ctx event. Should visualize the context."""
        self.visualize_branch(
            "pgo_frame",
            ctx,
            execution_time_metadata,
        )

    def visualize_branch(
        self,
        branch: str,
        ctx: Ctx,
        execution_time_metadata: ExecutionTimeMetadata,
    ) -> None:
        """Materialize reactive metadata into the context and visualize one branch."""
        ctx.set_record_batch(EXECUTION_TIME_MS_METADATA_FIELD, execution_time_metadata)
        self.vizualize.send((branch, ctx.reassemble()))

    def resolve_save_path(self, runtime_config: RerunNodeRuntimeConfig) -> Path | None:
        """Resolve the RRD output path for file-backed rerun sinks."""
        if not runtime_config.save_recording:
            return None

        repo_root = (runtime_config.repo_root or Path.cwd()).resolve()
        if runtime_config.output is not None:
            output = runtime_config.output
            return output if output.is_absolute() else repo_root / output

        return repo_root / "pipeline" / "out" / str(self.node.dataflow_id()) / "data.rrd"

    def resolve_artifact_dir(self, save_path: Path | None) -> Path:
        """Resolve where rerun sidecar artifacts should be written."""
        if save_path is not None:
            return save_path.parent

        repo_root = (self.node_runtime_config.repo_root or Path.cwd()).resolve()
        return repo_root / "pipeline" / "out" / str(self.node.dataflow_id())

    def write_recording_artifacts(self, save_path: Path | None) -> None:
        """Write agent-readable sidecar files for the rerun recording."""
        artifact_dir = self.resolve_artifact_dir(save_path)
        config_path = artifact_dir / "rerun_config.json"
        manifest_path = artifact_dir / "rerun_manifest.json"
        try:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(self.config.model_dump(mode="json"), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    self.build_recording_manifest(
                        save_path=save_path,
                        artifact_dir=artifact_dir,
                        config_path=config_path,
                        manifest_path=manifest_path,
                    ),
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            self.logger.warning(f"Could not write rerun recording artifacts: {exc}")

    def build_recording_manifest(
        self,
        *,
        save_path: Path | None,
        artifact_dir: Path,
        config_path: Path,
        manifest_path: Path,
    ) -> dict[str, object]:
        """Build an agent-readable manifest for the rerun recording."""
        return {
            "schema_version": 1,
            "dataflow_id": str(self.node.dataflow_id()),
            "app_name": self.config.app_name,
            "source_config_path": os.getenv("VISUALIZE_CONFIG"),
            "artifact_dir": str(artifact_dir),
            "files": {
                "rrd": str(save_path) if save_path is not None else None,
                "rerun_config": str(config_path),
                "rerun_manifest": str(manifest_path),
            },
            "runtime_config": self.node_runtime_config.model_dump(mode="json"),
            "stream_index": build_rerun_stream_index(self.config),
        }

    @on_input("helthcheck")
    def handle_helthcheck(self) -> None:
        """Handle the helthcheck event."""
        self.logger.trace("Still alive")

    @on_stop
    def handle_shutdown(self) -> None:
        """Handle the shutdown event."""
        self.logger.info("Rerun node stopping...")
        self.vizualize.close()
        self.logger.info("Rerun node stopped")


if __name__ == "__main__":
    runtime_config = RerunNode.runtime_config_as(RerunNodeRuntimeConfig)
    RerunNode(
        RerunConfigLoader.from_env_path("VISUALIZE_CONFIG"),
        RerunNode.load_dataset_rig_from_env().cam0.resolution,
        runtime_config=runtime_config,
    ).run()
