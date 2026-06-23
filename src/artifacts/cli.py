from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import typer
from yaml import safe_dump

from artifacts.fetch import (
    ArtifactFetchError,
    ArtifactFetchManager,
    ArtifactFetchPlan,
)
from artifacts.registry import ArtifactRegistry

if TYPE_CHECKING:
    from artifacts.manifest import ArtifactManifest

app = typer.Typer()

BYTES_PER_KIB = 1024
DOWNLOAD_PROGRESS_BAR_WIDTH = 32
PERCENT_DENOMINATOR = 100

ArtifactDirOption = Annotated[
    Path | None,
    typer.Option("--artifact-dir", help="Artifact manifest registry directory."),
]
OutputFormatOption = Annotated[
    Literal["table", "yaml"],
    typer.Option("--format", help="Output format."),
]
ArtifactArgument = Annotated[str, typer.Argument(help="Artifact name or unique artifact type selector.")]
OptionalFileArgument = Annotated[
    str | None,
    typer.Argument(help="Optional artifact file id. Prints the artifact root when omitted."),
]
OptionalArtifactArgument = Annotated[
    str | None,
    typer.Argument(help="Optional artifact name or unique artifact type selector."),
]
VerifyHashesOption = Annotated[
    bool,
    typer.Option("--verify-hashes", help="Validate SHA-256 checksums declared by manifests."),
]
ForceOption = Annotated[bool, typer.Option("--force", help="Refresh existing local artifact files.")]
YesOption = Annotated[bool, typer.Option("--yes", "-y", help="Skip the preflight confirmation prompt.")]


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _display_issue(path: Path, manifest: ArtifactManifest, registry: ArtifactRegistry) -> str:
    root = registry.resolve_path(manifest.root)
    if path == root:
        return "root"
    try:
        return str(path.relative_to(root))
    except ValueError:
        return _display_path(path, registry.repo_root)


def _artifact_summary(
    manifest: ArtifactManifest,
    registry: ArtifactRegistry,
    *,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    local_status = registry.local_status(manifest, verify_hashes=verify_hashes)
    source = manifest.source.model_dump(mode="json", exclude_none=True) if manifest.source is not None else None
    return {
        "name": manifest.name,
        "type": manifest.type,
        "version": manifest.version,
        "description": manifest.description,
        "tags": manifest.tags,
        "root": str(manifest.root),
        "source": source,
        "metadata": manifest.metadata,
        "files": {
            file_id: {
                "path": str(file_config.path),
                "description": file_config.description,
                "optional": file_config.optional,
                "size_bytes": file_config.size_bytes,
                "sha256": file_config.sha256,
                "local": {
                    "path": _display_path(local_status.files[file_id].path, registry.repo_root),
                    "exists": local_status.files[file_id].exists,
                    "verified": local_status.files[file_id].verified,
                    "actual_size_bytes": local_status.files[file_id].actual_size_bytes,
                    "actual_sha256": local_status.files[file_id].actual_sha256,
                    "issue": local_status.files[file_id].issue,
                },
            }
            for file_id, file_config in manifest.files.items()
        },
        "local": {
            "exists": local_status.exists,
            "verified": local_status.verified,
            "issues": [_display_path(path, registry.repo_root) for path in local_status.issues],
        },
    }


def _artifact_table_row(
    manifest: ArtifactManifest,
    registry: ArtifactRegistry,
    *,
    verify_hashes: bool = False,
) -> dict[str, str]:
    local_status = registry.local_status(manifest, verify_hashes=verify_hashes)
    issues = ", ".join(_display_issue(path, manifest, registry) for path in local_status.issues)
    return {
        "NAME": manifest.name,
        "TYPE": manifest.type,
        "EXISTS": str(local_status.exists).lower(),
        "VERIFIED": str(local_status.verified).lower(),
        "ISSUES": issues or "-",
        "ROOT": str(manifest.root),
    }


def _render_table(rows: list[dict[str, str]]) -> str:
    headers = ["NAME", "TYPE", "EXISTS", "VERIFIED", "ISSUES", "ROOT"]
    if not rows:
        return "No artifact manifests found."
    widths = {header: max(len(header), *(len(row[header]) for row in rows)) for header in headers}
    lines = [
        "  ".join(header.ljust(widths[header]) for header in headers),
        *["  ".join(row[header].ljust(widths[header]) for header in headers) for row in rows],
    ]
    return "\n".join(lines)


@app.callback()
def artifact_cli() -> None:
    """Artifact registry CLI."""


@app.command("list")
def list_artifacts(
    artifact_dir: ArtifactDirOption = None,
    output_format: OutputFormatOption = "table",
    verify_hashes: VerifyHashesOption = False,  # noqa: FBT002
) -> None:
    """List supported artifact manifests."""
    registry = ArtifactRegistry(artifact_dir=artifact_dir)
    manifests = registry.list()
    if output_format == "yaml":
        summaries = [_artifact_summary(manifest, registry, verify_hashes=verify_hashes) for manifest in manifests]
        typer.echo(safe_dump(summaries, sort_keys=False))
        return

    typer.echo(
        _render_table(
            [_artifact_table_row(manifest, registry, verify_hashes=verify_hashes) for manifest in manifests],
        ),
    )


@app.command("show")
def show_artifact(
    artifact: ArtifactArgument,
    artifact_dir: ArtifactDirOption = None,
    output_format: OutputFormatOption = "yaml",
    verify_hashes: VerifyHashesOption = False,  # noqa: FBT002
) -> None:
    """Show one artifact manifest with local availability status."""
    registry = ArtifactRegistry(artifact_dir=artifact_dir)
    try:
        manifest = registry.find(artifact)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    if output_format == "table":
        typer.echo(_render_table([_artifact_table_row(manifest, registry, verify_hashes=verify_hashes)]))
        return
    typer.echo(safe_dump(_artifact_summary(manifest, registry, verify_hashes=verify_hashes), sort_keys=False))


@app.command("fetch")
def fetch_artifact(
    artifact: ArtifactArgument,
    artifact_dir: ArtifactDirOption = None,
    force: ForceOption = False,  # noqa: FBT002
    yes: YesOption = False,  # noqa: FBT002
) -> None:
    """Fetch and prepare an artifact payload through its configured strategy."""
    registry = ArtifactRegistry(artifact_dir=artifact_dir)
    try:
        manifest = registry.find(artifact)
        progress = _DownloadProgressPrinter()
        manager = ArtifactFetchManager(registry=registry, progress_callback=progress)
        plan = manager.plan_artifact(manifest, force=force)
    except (FileNotFoundError, ValueError, ArtifactFetchError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    typer.echo(_render_fetch_plan(plan))
    if plan.actions and not yes and not typer.confirm("Continue?", default=True):
        typer.echo("Aborted.")
        raise typer.Exit(0)

    try:
        result = manager.ensure_artifact(manifest, force=force)
    except ArtifactFetchError as exc:
        progress.finish()
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    progress.finish()

    fetch_status = "fetched" if result.fetched else "ready"
    extract_status = "extracted" if result.extracted else "ready"
    convert_status = "converted" if result.converted else "ready"
    typer.echo(
        f"{result.artifact}: source={fetch_status} text={extract_status} "
        f"binary={convert_status} ready={str(result.ready).lower()}"
    )


@app.command("path")
def artifact_path(
    artifact: ArtifactArgument,
    file_id: OptionalFileArgument = None,
    artifact_dir: ArtifactDirOption = None,
) -> None:
    """Print the resolved artifact root path or one resolved artifact file path."""
    registry = ArtifactRegistry(artifact_dir=artifact_dir)
    try:
        manifest = registry.find(artifact)
        path = (
            registry.resolve_path(manifest.root)
            if file_id is None
            else registry.resolve_file_path(manifest, file_id)
        )
    except KeyError as exc:
        typer.echo(f"Artifact '{artifact}' has no file id '{file_id}'.", err=True)
        raise typer.Exit(1) from exc
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(path)


@app.command("verify")
def verify_artifacts(
    artifact: OptionalArtifactArgument = None,
    artifact_dir: ArtifactDirOption = None,
    output_format: OutputFormatOption = "table",
    verify_hashes: VerifyHashesOption = True,  # noqa: FBT002
) -> None:
    """Verify artifact payload availability, including checksums by default."""
    registry = ArtifactRegistry(artifact_dir=artifact_dir)
    try:
        manifests = [registry.find(artifact)] if artifact is not None else registry.list()
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    if output_format == "yaml":
        summaries = [_artifact_summary(manifest, registry, verify_hashes=verify_hashes) for manifest in manifests]
        typer.echo(safe_dump(summaries, sort_keys=False))
        return
    typer.echo(
        _render_table(
            [_artifact_table_row(manifest, registry, verify_hashes=verify_hashes) for manifest in manifests],
        ),
    )


@app.command("open")
def open_artifact(artifact: ArtifactArgument, artifact_dir: ArtifactDirOption = None) -> None:
    """Open the artifact source page or download URL in the default browser."""
    registry = ArtifactRegistry(artifact_dir=artifact_dir)
    try:
        manifest = registry.find(artifact)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    source = manifest.source
    url = source.page_url if source is not None and source.page_url is not None else None
    url = url or (source.download_url if source is not None else None)
    if url is None:
        typer.echo(
            f"Artifact '{manifest.name}' has no source.page_url or source.download_url configured.",
            err=True,
        )
        raise typer.Exit(1)

    if not webbrowser.open(url):
        typer.echo(f"Could not open browser for {url}", err=True)
        raise typer.Exit(1)
    typer.echo(url)


def _render_fetch_plan(plan: ArtifactFetchPlan) -> str:
    lines = [f"Artifact: {plan.artifact}"]
    download_bytes = plan.download_bytes
    download_size = "unknown" if download_bytes is None else _format_bytes(download_bytes)
    lines.append(f"Download: {download_size}")
    if _plan_downloads(plan):
        if plan.source_page_url is not None:
            lines.append(f"Artifact source page: {plan.source_page_url}")
        if plan.open_command is not None:
            lines.append(f"To open artifact source page - cli command: {plan.open_command}")

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


def _plan_downloads(plan: ArtifactFetchPlan) -> bool:
    return any(action.download_bytes is None or action.download_bytes > 0 for action in plan.actions)


class _DownloadProgressPrinter:
    """Small terminal progress reporter for artifact downloads."""

    def __init__(self) -> None:
        self._printed_inline = False
        self._active_label: str | None = None
        self._last_render: tuple[int, int, int] | None = None
        self._last_line_length = 0

    def __call__(self, label: str, completed_bytes: int, total_bytes: int | None) -> None:
        if total_bytes is None or total_bytes <= 0:
            return
        if self._active_label != label:
            self.finish()
            self._active_label = label

        percent = min(PERCENT_DENOMINATOR, completed_bytes * PERCENT_DENOMINATOR // total_bytes)
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

    @staticmethod
    def _progress_bar(percent: int) -> str:
        filled = min(
            DOWNLOAD_PROGRESS_BAR_WIDTH, max(0, percent * DOWNLOAD_PROGRESS_BAR_WIDTH // PERCENT_DENOMINATOR)
        )
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
    """CLI entrypoint for the artifact package."""
    app()
