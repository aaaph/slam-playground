from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from typer.testing import CliRunner
from yaml import safe_dump, safe_load

from artifacts.cli import app

if TYPE_CHECKING:
    from pathlib import Path


class TestArtifactCli:
    """Artifact CLI tests."""

    def test_list_outputs_artifact_manifest_table(self, tmp_path: Path) -> None:
        """List artifact manifests through the artifact CLI."""
        registry = tmp_path / "registry"
        root = tmp_path / "vocabulary"
        _write_manifest(registry, root)
        _write_file(root / "ORBvoc.dbow3")

        result = CliRunner().invoke(app, ["list", "--artifact-dir", str(registry)])

        assert result.exit_code == 0
        lines = result.output.splitlines()
        assert lines[0].split() == ["NAME", "TYPE", "EXISTS", "VERIFIED", "ISSUES", "ROOT"]
        assert lines[1].split(maxsplit=5) == [
            "orb_vocabulary",
            "vocabulary",
            "true",
            "true",
            "-",
            str(root),
        ]

    def test_list_outputs_artifact_manifest_summaries_as_yaml(self, tmp_path: Path) -> None:
        """List artifact manifests through the artifact CLI as structured YAML."""
        registry = tmp_path / "registry"
        root = tmp_path / "vocabulary"
        _write_manifest(registry, root)
        _write_file(root / "ORBvoc.dbow3")

        result = CliRunner().invoke(app, ["list", "--artifact-dir", str(registry), "--format", "yaml"])

        assert result.exit_code == 0
        output = safe_load(result.output)
        assert output[0]["name"] == "orb_vocabulary"
        assert output[0]["type"] == "vocabulary"
        assert output[0]["root"] == str(root)
        assert output[0]["files"]["dbow3"]["path"] == "ORBvoc.dbow3"
        assert output[0]["files"]["dbow3"]["local"]["exists"] is True
        assert output[0]["local"]["verified"] is True

    def test_show_outputs_one_artifact_summary(self, tmp_path: Path) -> None:
        """Show should resolve one artifact by selector."""
        registry = tmp_path / "registry"
        root = tmp_path / "vocabulary"
        _write_manifest(registry, root)

        result = CliRunner().invoke(app, ["show", "orb_vocabulary", "--artifact-dir", str(registry)])

        assert result.exit_code == 0
        output = safe_load(result.output)
        assert output["name"] == "orb_vocabulary"
        assert output["local"]["issues"] == [str(root), str(root / "ORBvoc.dbow3")]

    def test_path_outputs_resolved_root_or_file_path(self, tmp_path: Path) -> None:
        """Path should print absolute root and file paths for scripting."""
        registry = tmp_path / "registry"
        root = tmp_path / "vocabulary"
        _write_manifest(registry, root)

        root_result = CliRunner().invoke(app, ["path", "orb_vocabulary", "--artifact-dir", str(registry)])
        file_result = CliRunner().invoke(app, ["path", "orb_vocabulary", "dbow3", "--artifact-dir", str(registry)])

        assert root_result.exit_code == 0
        assert root_result.output.strip() == str(root)
        assert file_result.exit_code == 0
        assert file_result.output.strip() == str(root / "ORBvoc.dbow3")

    def test_fetch_reports_no_actions_when_artifact_is_verified(self, tmp_path: Path) -> None:
        """Fetch should be a no-op when the configured payload already verifies."""
        registry = tmp_path / "registry"
        root = tmp_path / "vocabulary"
        _write_manifest(
            registry,
            root,
            files={
                "dbow3": {
                    "path": "ORBvoc.dbow3",
                    "size_bytes": len("payload"),
                    "sha256": _sha256(b"payload"),
                }
            },
            source={"download_url": "https://example.test/ORBvoc.txt.tar.gz"},
            metadata={
                "fetch_strategy": "orb_slam3_vocabulary",
                "archive_member": "ORBvoc.txt",
                "output_file_id": "dbow3",
            },
        )
        _write_file(root / "ORBvoc.dbow3")

        result = CliRunner().invoke(app, ["fetch", "orb_vocabulary", "--artifact-dir", str(registry)])

        assert result.exit_code == 0
        assert "Actions: none" in result.output
        assert "orb_vocabulary: source=ready text=ready binary=ready ready=true" in result.output

    def test_verify_uses_hash_checks_by_default(self, tmp_path: Path) -> None:
        """Verify should check declared hashes unless disabled."""
        registry = tmp_path / "registry"
        root = tmp_path / "vocabulary"
        _write_manifest(
            registry,
            root,
            files={"dbow3": {"path": "ORBvoc.dbow3", "sha256": "0" * 64}},
        )
        _write_file(root / "ORBvoc.dbow3")

        result = CliRunner().invoke(app, ["verify", "orb_vocabulary", "--artifact-dir", str(registry)])

        assert result.exit_code == 0
        assert "false" in result.output
        assert "ORBvoc.dbow3" in result.output

    def test_open_artifact_uses_source_page_url(self, tmp_path: Path, monkeypatch) -> None:
        """Open should resolve the manifest and pass the source page URL to the browser."""
        registry = tmp_path / "registry"
        root = tmp_path / "vocabulary"
        _write_manifest(registry, root)
        opened: list[str] = []
        monkeypatch.setattr("artifacts.cli.webbrowser.open", lambda url: opened.append(url) or True)

        result = CliRunner().invoke(app, ["open", "orb_vocabulary", "--artifact-dir", str(registry)])

        assert result.exit_code == 0
        assert opened == ["https://example.test/orb"]
        assert "https://example.test/orb" in result.output


def _write_manifest(
    registry: Path,
    root: Path,
    *,
    files: dict[str, dict[str, object]] | None = None,
    source: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    registry.mkdir()
    manifest = {
        "name": "orb_vocabulary",
        "type": "vocabulary",
        "root": str(root),
        "source": source or {"page_url": "https://example.test/orb"},
        "files": files or {"dbow3": {"path": "ORBvoc.dbow3"}},
        "metadata": metadata or {},
    }
    (registry / "orb_vocabulary.yaml").write_text(safe_dump(manifest), encoding="utf-8")


def _write_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("payload", encoding="utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
