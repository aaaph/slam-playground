from __future__ import annotations

from typing import Any

from visualizer.rerun.schemas import DEFAULT_BRANCH, ModuleType, RerunConfigSchema, ViewSchema, ViewType


def build_rerun_stream_index(config: RerunConfigSchema) -> list[dict[str, Any]]:
    """Build an agent-readable index of configured Rerun streams."""
    streams: list[dict[str, Any]] = []
    default_branch = config.default_branch or DEFAULT_BRANCH
    for view in config.views:
        _append_view_streams(
            streams,
            view=view,
            inherited_branch=default_branch,
            view_path=[],
        )
    return streams


def _append_view_streams(
    streams: list[dict[str, Any]],
    *,
    view: ViewSchema,
    inherited_branch: str,
    view_path: list[str],
) -> None:
    branch = view.branch or inherited_branch
    current_view_path = [*view_path, view.name]
    if view.type == ViewType.CONTAINER:
        for child_view in view.views:
            _append_view_streams(
                streams,
                view=child_view,
                inherited_branch=branch,
                view_path=current_view_path,
            )
        return

    for entity in view.streams:
        entity_path = _resolve_entity_path(view.origin, entity.entity)
        stream = {
            "branch": entity.branch or branch,
            "view": view.name,
            "view_path": current_view_path,
            "view_type": view.type.value,
            "view_origin": view.origin,
            "module": entity.module.value,
            "property_name": entity.id,
            "entity_path": entity_path,
            "options": entity.options,
            "logged_entities": _logged_entities(entity.module, entity_path, entity.options),
        }
        streams.append(stream)


def _resolve_entity_path(view_origin: str | None, entity_path: str) -> str:
    if entity_path != ".":
        return entity_path
    if view_origin is None:
        msg = "entity='.' requires the view to define an origin"
        raise ValueError(msg)
    return view_origin


def _logged_entities(module: ModuleType, entity_path: str, options: dict[str, Any]) -> list[dict[str, Any]]:
    if module == ModuleType.PLOT_COLUMN:
        return _plot_column_logged_entities(entity_path, options)
    if module in {ModuleType.PLOT_SCALAR, ModuleType.PLOT_3D_VECTOR}:
        return [
            {
                "entity_path": entity_path,
                "component": "Scalars:scalars",
                "timeline": "sim_time",
            }
        ]
    if module == ModuleType.IMAGE:
        return [{"entity_path": entity_path, "component": "Image", "timeline": "sim_time"}]
    return [{"entity_path": entity_path}]


def _plot_column_logged_entities(entity_path: str, options: dict[str, Any]) -> list[dict[str, Any]]:
    timeline = options.get("timeline", "sim_time")
    raw_mapping = options.get("mapping", [])
    if not isinstance(raw_mapping, list):
        return []

    result: list[dict[str, Any]] = []
    for item in raw_mapping:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        if not isinstance(label, str):
            continue
        result.append(
            {
                "entity_path": f"{entity_path}/{label}",
                "component": "Scalars:scalars",
                "timeline": timeline,
                "source_column": item.get("index"),
                "label": label,
            }
        )
    return result
