from pathlib import Path
from typing import Annotated, Any, Literal

import typer
from yaml import safe_dump

from dataset.manifest import DatasetManifest, DatasetManifestLoader

app = typer.Typer()

DatasetDirOption = Annotated[
    Path | None,
    typer.Option("--dataset-dir", help="Dataset manifest registry directory."),
]
OutputFormatOption = Annotated[
    Literal["table", "yaml"],
    typer.Option("--format", help="Output format."),
]


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _display_issue(path: Path, manifest: DatasetManifest, loader: DatasetManifestLoader) -> str:
    root = loader.resolve_path(manifest.root)
    if path == root:
        return "root"
    try:
        return str(path.relative_to(root))
    except ValueError:
        return _display_path(path, loader.repo_root)


def _dataset_summary(manifest: DatasetManifest, loader: DatasetManifestLoader) -> dict[str, Any]:
    local_status = loader.local_status(manifest)
    return {
        "name": manifest.name,
        "type": manifest.type,
        "root": str(manifest.root),
        "rig": str(manifest.rig),
        "cache": str(manifest.cache) if manifest.cache is not None else None,
        "local": {
            "exists": local_status.exists,
            "verified": local_status.verified,
            "issues": [_display_path(path, loader.repo_root) for path in local_status.issues],
        },
    }


def _dataset_table_row(manifest: DatasetManifest, loader: DatasetManifestLoader) -> dict[str, str]:
    local_status = loader.local_status(manifest)
    issues = ", ".join(_display_issue(path, manifest, loader) for path in local_status.issues)
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
    loader = DatasetManifestLoader(dataset_dir=dataset_dir)
    manifests = loader.list_datasets()
    if output_format == "yaml":
        summaries = [_dataset_summary(manifest, loader) for manifest in manifests]
        typer.echo(safe_dump(summaries, sort_keys=False))
        return

    typer.echo(_render_table([_dataset_table_row(manifest, loader) for manifest in manifests]))


def main() -> None:
    """CLI entrypoint for the dataset package."""
    app()
