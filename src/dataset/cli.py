import math
import shutil
import webbrowser
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
from yaml import safe_dump

from dataset.fetch import (
    DatasetCacheManager,
    DatasetFetchError,
    DatasetFetchPlan,
    DownloadHeadResult,
    fetch_download_headers,
)
from dataset.manifest import DatasetManifest
from dataset.registry import DatasetRegistry

app = typer.Typer()
cache_app = typer.Typer()
app.add_typer(cache_app, name="cache")

DatasetDirOption = Annotated[
    Path | None,
    typer.Option("--dataset-dir", help="Dataset manifest registry directory."),
]
OutputFormatOption = Annotated[
    Literal["table", "yaml"],
    typer.Option("--format", help="Output format."),
]
DatasetArgument = Annotated[str, typer.Argument(help="Dataset name or unique dataset type selector.")]
OptionalDatasetArgument = Annotated[
    str | None,
    typer.Argument(help="Optional dataset name or unique dataset type selector."),
]
ForceOption = Annotated[bool, typer.Option("--force", help="Refresh existing raw files or cache.")]
YesOption = Annotated[bool, typer.Option("--yes", "-y", help="Skip the preflight confirmation prompt.")]
BYTES_PER_KIB = 1024
DOWNLOAD_PROGRESS_BAR_WIDTH = 32
PERCENT_DENOMINATOR = 100


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _display_issue(path: Path, manifest: DatasetManifest, registry: DatasetRegistry) -> str:
    root = registry.resolve_path(manifest.root)
    if path == root:
        return "root"
    try:
        return str(path.relative_to(root))
    except ValueError:
        return _display_path(path, registry.repo_root)


def _dataset_summary(manifest: DatasetManifest, registry: DatasetRegistry) -> dict[str, Any]:
    local_status = registry.local_status(manifest)
    return {
        "name": manifest.name,
        "type": manifest.type,
        "root": str(manifest.root),
        "rig": str(manifest.rig),
        "cache": str(manifest.cache) if manifest.cache is not None else None,
        "local": {
            "exists": local_status.exists,
            "verified": local_status.verified,
            "issues": [_display_path(path, registry.repo_root) for path in local_status.issues],
            "cache": {
                "path": _display_path(local_status.cache_path, registry.repo_root),
                "exists": local_status.cache_exists,
                "verified": local_status.cache_verified,
                "issues": [_display_path(path, registry.repo_root) for path in local_status.cache_issues],
            },
        },
    }


def _dataset_table_row(manifest: DatasetManifest, registry: DatasetRegistry) -> dict[str, str]:
    local_status = registry.local_status(manifest)
    issues = ", ".join(_display_issue(path, manifest, registry) for path in local_status.issues)
    return {
        "NAME": manifest.name,
        "TYPE": manifest.type,
        "EXISTS": str(local_status.exists).lower(),
        "VERIFIED": str(local_status.verified).lower(),
        "CACHE": str(local_status.cache_verified).lower(),
        "ISSUES": issues or "-",
        "ROOT": str(manifest.root),
    }


def _render_table(rows: list[dict[str, str]]) -> str:
    headers = ["NAME", "TYPE", "EXISTS", "VERIFIED", "CACHE", "ISSUES", "ROOT"]
    widths = {header: max(len(header), *(len(row[header]) for row in rows)) for header in headers}
    lines = [
        "  ".join(header.ljust(widths[header]) for header in headers),
        *["  ".join(row[header].ljust(widths[header]) for header in headers) for row in rows],
    ]
    return "\n".join(lines)


@app.callback()
def dataset_cli() -> None:
    """Dataset registry CLI."""


@app.command("list")
def list_datasets(dataset_dir: DatasetDirOption = None, output_format: OutputFormatOption = "table") -> None:
    """List supported dataset manifests."""
    registry = DatasetRegistry(dataset_dir=dataset_dir)
    manifests = registry.list()
    if output_format == "yaml":
        summaries = [_dataset_summary(manifest, registry) for manifest in manifests]
        typer.echo(safe_dump(summaries, sort_keys=False))
        return

    typer.echo(_render_table([_dataset_table_row(manifest, registry) for manifest in manifests]))


@app.command("open")
def open_dataset(dataset: DatasetArgument, dataset_dir: DatasetDirOption = None) -> None:
    """Open the official dataset page in the default browser."""
    registry = DatasetRegistry(dataset_dir=dataset_dir)
    manifest = registry.find(dataset)
    source = manifest.source
    url = source.page_url if source is not None and source.page_url is not None else None
    url = url or (source.download_url if source is not None else None)
    if url is None:
        typer.echo(
            f"Dataset '{manifest.name}' has no source.page_url or source.download_url configured.",
            err=True,
        )
        raise typer.Exit(1)

    if not webbrowser.open(url):
        typer.echo(f"Could not open browser for {url}", err=True)
        raise typer.Exit(1)
    typer.echo(url)


@app.command("fetch")
def fetch_dataset(
    dataset: DatasetArgument,
    dataset_dir: DatasetDirOption = None,
    force: ForceOption = False,  # noqa: FBT002
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Only fetch raw stream files.")] = False,  # noqa: FBT002
    yes: YesOption = False,  # noqa: FBT002
) -> None:
    """Fetch raw files and materialize the local dataset cache."""
    registry = DatasetRegistry(dataset_dir=dataset_dir)
    manifest = registry.find(dataset)
    progress = _DownloadProgressPrinter()
    manager = DatasetCacheManager(registry=registry, progress_callback=progress)
    try:
        plan = manager.plan_dataset(
            manifest,
            force_raw=force,
            force_cache=force,
            materialize_cache=not no_cache,
        )
    except DatasetFetchError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    typer.echo(_render_fetch_plan(plan))
    if plan.actions and not yes and not typer.confirm("Continue?", default=True):
        typer.echo("Aborted.")
        raise typer.Exit(0)

    try:
        result = manager.ensure_dataset(
            manifest,
            force_raw=force,
            force_cache=force,
            materialize_cache=not no_cache,
        )
    except DatasetFetchError as exc:
        progress.finish()
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    progress.finish()

    raw_status = "fetched" if result.raw_fetched else "ready"
    cache_status = "skipped" if no_cache else ("built" if result.cache_built else "ready")
    typer.echo(f"{result.dataset}: raw={raw_status} cache={cache_status}")


@app.command("fetch-head")
def fetch_head_dataset(dataset: DatasetArgument, dataset_dir: DatasetDirOption = None) -> None:
    """Print HTTP HEAD headers for the dataset download URL."""
    registry = DatasetRegistry(dataset_dir=dataset_dir)
    manifest = registry.find(dataset)
    source = manifest.source
    if source is None or source.download_url is None:
        typer.echo(f"Dataset '{manifest.name}' has no source.download_url configured.", err=True)
        raise typer.Exit(1)

    try:
        results = fetch_download_headers(source.download_url)
    except DatasetFetchError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    typer.echo(_render_head_results(results))


@cache_app.command("clear")
def clear_cache(dataset: OptionalDatasetArgument = None, dataset_dir: DatasetDirOption = None) -> None:
    """Remove materialized dataset caches without touching raw dataset files."""
    registry = DatasetRegistry(dataset_dir=dataset_dir)
    manifests = [registry.find(dataset)] if dataset is not None else registry.list()
    for manifest in manifests:
        cache_root = _cache_root(manifest, registry)
        _remove_cache_root(cache_root, manifest, registry)


def _cache_root(manifest: DatasetManifest, registry: DatasetRegistry) -> Path:
    if manifest.cache is not None:
        return registry.resolve_path(manifest.cache)
    return registry.resolve_path(manifest.root) / "cache"


def _remove_cache_root(cache_root: Path, manifest: DatasetManifest, registry: DatasetRegistry) -> None:
    dataset_root = registry.resolve_path(manifest.root)
    if cache_root in {registry.repo_root, dataset_root}:
        typer.echo(f"{manifest.name}: refusing to remove unsafe cache path {cache_root}", err=True)
        raise typer.Exit(1)

    display_path = _display_path(cache_root, registry.repo_root)
    if not cache_root.exists():
        typer.echo(f"{manifest.name}: cache already clear ({display_path})")
        return

    if cache_root.is_symlink() or cache_root.is_file():
        cache_root.unlink()
    else:
        shutil.rmtree(cache_root)
    typer.echo(f"{manifest.name}: cleared {display_path}")


def _render_head_results(results: list[DownloadHeadResult]) -> str:
    rendered: list[str] = []
    for result in results:
        if rendered:
            rendered.append("")
        rendered.append(f"URL: {result.url}")
        status = f"{result.status} {result.reason}".rstrip()
        rendered.append(f"Status: {status}")
        rendered.extend(f"{name}: {value}" for name, value in result.headers.items())
    return "\n".join(rendered)


def _render_fetch_plan(plan: DatasetFetchPlan) -> str:
    lines = [f"Dataset: {plan.dataset}"]
    download_bytes = plan.download_bytes
    download_size = "unknown" if download_bytes is None else _format_bytes(download_bytes)
    lines.append(f"Download: {download_size}")
    if _plan_downloads(plan):
        if plan.dataset_page_url is not None:
            lines.append(f"Dataset link page: {plan.dataset_page_url}")
        if plan.open_command is not None:
            lines.append(f"To open dataset link page - cli command: {plan.open_command}")

    if not plan.actions:
        lines.append("Actions: none")
        return "\n".join(lines)

    lines.append("Actions:")
    for index, action in enumerate(plan.actions, start=1):
        action_line = f"  {index}. {action.name}"
        if action.download_bytes is None:
            action_line += " (download size unknown)"
        elif action.download_bytes:
            action_line += f" ({_format_bytes(action.download_bytes)} download)"
        lines.append(action_line)
        if action.detail is not None:
            lines.append(f"     {action.detail}")
    return "\n".join(lines)


def _plan_downloads(plan: DatasetFetchPlan) -> bool:
    return any(action.download_bytes is None or action.download_bytes > 0 for action in plan.actions)


class _DownloadProgressPrinter:
    """Small terminal progress reporter for archive downloads."""

    def __init__(self) -> None:
        self._printed_inline = False
        self._active_label: str | None = None
        self._last_render: tuple[int, int, int] | None = None
        self._last_line_length = 0
        self._next_unknown_update = 128 * 1024 * 1024
        self._simple_started: set[str] = set()
        self._simple_done: set[str] = set()

    def __call__(self, label: str, completed_bytes: int, total_bytes: int | None) -> None:
        if total_bytes == 1:
            self._print_simple_step(label, completed_bytes)
            return

        if total_bytes is None or total_bytes <= 0:
            self._print_unknown_total(label, completed_bytes)
            return

        if self._active_label != label:
            self.finish()
            self._active_label = label

        percent = min(PERCENT_DENOMINATOR, math.floor(completed_bytes * PERCENT_DENOMINATOR / total_bytes))
        render_key = (completed_bytes, total_bytes, percent)
        if render_key == self._last_render:
            return

        self._last_render = render_key
        self._printed_inline = True
        line = (
            f"{label}: [{self._progress_bar(percent)}] "
            f"{percent:3d}% "
            f"({_format_bytes(completed_bytes)} / {_format_bytes(total_bytes)})"
        )
        padding = " " * max(0, self._last_line_length - len(line))
        self._last_line_length = len(line)
        typer.echo(f"\r{line}{padding}", nl=False)

    def finish(self) -> None:
        if self._printed_inline:
            typer.echo()
            self._printed_inline = False
            self._last_line_length = 0
            self._last_render = None

    def _print_unknown_total(self, label: str, completed_bytes: int) -> None:
        if self._active_label != label:
            self.finish()
            self._active_label = label
            self._next_unknown_update = 128 * 1024 * 1024

        if completed_bytes < self._next_unknown_update:
            return
        self._next_unknown_update += 128 * 1024 * 1024
        typer.echo(f"{label}: {_format_bytes(completed_bytes)}")

    def _print_simple_step(self, label: str, completed_steps: int) -> None:
        self.finish()
        if completed_steps <= 0:
            if label not in self._simple_started:
                self._simple_started.add(label)
                typer.echo(f"{label}...")
            return

        if label not in self._simple_done:
            self._simple_done.add(label)
            typer.echo(f"{label}: done")

    @staticmethod
    def _progress_bar(percent: int) -> str:
        filled = math.floor(percent * DOWNLOAD_PROGRESS_BAR_WIDTH / PERCENT_DENOMINATOR)
        filled = min(DOWNLOAD_PROGRESS_BAR_WIDTH, max(0, filled))
        return "#" * filled + "-" * (DOWNLOAD_PROGRESS_BAR_WIDTH - filled)


def _format_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    unit = units[0]
    for unit in units:
        if abs(size) < BYTES_PER_KIB or unit == units[-1]:
            break
        size /= BYTES_PER_KIB
    return f"{size:.1f} {unit}"


def main() -> None:
    """CLI entrypoint for the dataset package."""
    app()
