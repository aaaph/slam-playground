from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import rerun as rr
import rerun.blueprint as rrb

from logger import spawn_logger
from pipeline.context import PipelineContext
from visualizer.coroutine_decorator import coroutine
from visualizer.rerun.modules.abc_module import IVizModule


@dataclass(frozen=True)
class SetupLog:
    """Static Rerun log emitted after recording initialization."""

    entity_path: str
    archetype: Any


class RerunVizualizer:
    """Rerun vizualizer."""

    def __init__(self, app_name: str, *, spawn: bool = True) -> None:
        """Initialize the rerun vizualizer."""
        self.app_name = app_name
        self.spawn = spawn
        self.logger = spawn_logger(app=app_name)
        self.modules: list[IVizModule] = []
        self.blueprint_parts: list[rrb.BlueprintPart] = []
        self.setup_logs: list[SetupLog] = []

    def info(self) -> dict[str, Any]:
        """Get the info of the rerun vizualizer."""
        return {
            "app_name": self.app_name,
            "spawn": self.spawn,
            "modules_count": len(self.modules),
            "modules": self.modules,
        }

    def add_bluepint_part(self, blueprint_part: rrb.BlueprintPart) -> None:
        """Add a blueprint part to the rerun vizualizer."""
        self.blueprint_parts.append(blueprint_part)

    def add_module(self, module: IVizModule) -> None:
        """Add a module to the rerun vizualizer."""
        self.modules.append(module)

    def add_setup_log(self, setup_log: SetupLog) -> None:
        """Add a static setup log to the rerun vizualizer."""
        self.setup_logs.append(setup_log)

    @coroutine
    def pipeline_generator(self) -> Generator[None, PipelineContext]:
        """Rerun generator for dataflow pipelines.."""
        self.logger.info("Connecting to rerun")
        blueprint = rrb.Blueprint(*reversed(self.blueprint_parts))
        rr.init(self.app_name, spawn=self.spawn, default_blueprint=blueprint)
        for setup_log in self.setup_logs:
            rr.log(setup_log.entity_path, setup_log.archetype, static=True)
        for module in self.modules:
            module.setup()

        try:
            while True:
                ctx: PipelineContext = yield
                rr.set_time("sim_time", timestamp=ctx.get_scalar("timestamp", float) / 1e9)
                rr.log("timestamp", rr.TextLog(f"{ctx.get_scalar('timestamp', float):.0f}"))
                for module in self.modules:
                    module.process(ctx)
        finally:
            self.logger.info("Disconnecting from rerun")
            rr.disconnect()
