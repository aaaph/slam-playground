from __future__ import annotations

import importlib
import shutil
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from artifacts.registry import ArtifactRegistry, _sha256_file

if TYPE_CHECKING:
    from artifacts.manifest import ArtifactManifest, ArtifactSourceConfig

DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DOWNLOADS_DIR = Path("artifacts/.downloads")
ORB_SLAM3_VOCABULARY_STRATEGY = "orb_slam3_vocabulary"
DEFAULT_ORB_SLAM3_ARCHIVE_MEMBER = "ORBvoc.txt"
DEFAULT_ORB_SLAM3_OUTPUT_FILE_ID = "dbow3"
SAVE_BINARY_COMPRESSED = True

type ArtifactProgressCallback = Callable[[str, int, int | None], None]


class ArtifactFetchError(RuntimeError):
    """Raised when an artifact cannot be fetched or prepared."""


@dataclass(frozen=True)
class ArtifactFetchResult:
    """Outcome of preparing an artifact for local use."""

    artifact: str
    fetched: bool
    extracted: bool
    converted: bool
    ready: bool


@dataclass(frozen=True)
class ArtifactFetchPlanAction:
    """One operation that will be executed while preparing an artifact."""

    name: str
    detail: str | None = None
    download_bytes: int | None = 0


@dataclass(frozen=True)
class ArtifactFetchPlan:
    """Preflight description of artifact preparation operations."""

    artifact: str
    actions: tuple[ArtifactFetchPlanAction, ...]
    source_page_url: str | None = None
    open_command: str | None = None

    @property
    def download_bytes(self) -> int | None:
        """Total bytes that will be downloaded, or None when at least one download is unknown."""
        total = 0
        for action in self.actions:
            if action.download_bytes is None:
                return None
            total += action.download_bytes
        return total


class ArtifactFetchStrategy(Protocol):
    """Strategy that can populate one artifact family."""

    def plan(self, manifest: ArtifactManifest, *, force: bool = False) -> list[ArtifactFetchPlanAction]:
        """Return operations needed to make an artifact available locally."""

    def ensure(self, manifest: ArtifactManifest, *, force: bool = False) -> ArtifactFetchResult:
        """Make an artifact available locally."""


class ArtifactFetchManager:
    """Dispatch artifact fetches to concrete artifact strategies."""

    def __init__(
        self,
        *,
        registry: ArtifactRegistry | None = None,
        downloads_dir: Path | None = None,
        progress_callback: ArtifactProgressCallback | None = None,
        strategies: dict[str, ArtifactFetchStrategy] | None = None,
    ) -> None:
        """Create an artifact fetch manager."""
        self.registry = registry or ArtifactRegistry()
        self.downloads_dir = self.registry.resolve_path(downloads_dir or DOWNLOADS_DIR)
        self.progress_callback = progress_callback
        self.strategies = strategies or {
            ORB_SLAM3_VOCABULARY_STRATEGY: OrbSlam3VocabularyFetchStrategy(
                registry=self.registry,
                downloads_dir=self.downloads_dir,
                progress_callback=progress_callback,
            ),
        }

    def plan_artifact(self, manifest: ArtifactManifest, *, force: bool = False) -> ArtifactFetchPlan:
        """Build a preflight plan for one artifact."""
        strategy = self._strategy_for(manifest)
        source = manifest.source
        actions = tuple(strategy.plan(manifest, force=force))
        return ArtifactFetchPlan(
            artifact=manifest.name,
            source_page_url=source.page_url if source is not None else None,
            open_command=f"just artifact open {manifest.name}" if source is not None else None,
            actions=actions,
        )

    def ensure_artifact(self, manifest: ArtifactManifest, *, force: bool = False) -> ArtifactFetchResult:
        """Fetch and prepare one artifact."""
        strategy = self._strategy_for(manifest)
        return strategy.ensure(manifest, force=force)

    def _strategy_for(self, manifest: ArtifactManifest) -> ArtifactFetchStrategy:
        strategy_name = manifest.metadata.get("fetch_strategy")
        if not isinstance(strategy_name, str) or not strategy_name:
            msg = f"Artifact '{manifest.name}' has no metadata.fetch_strategy configured"
            raise ArtifactFetchError(msg)
        strategy = self.strategies.get(strategy_name)
        if strategy is None:
            supported = ", ".join(sorted(self.strategies)) or "<none>"
            msg = (
                f"Artifact '{manifest.name}' uses unsupported fetch strategy "
                f"'{strategy_name}'. Supported: {supported}"
            )
            raise ArtifactFetchError(msg)
        return strategy


class OrbSlam3VocabularyFetchStrategy:
    """Fetch ORB-SLAM3's ORB vocabulary and convert it into a DBoW3 binary vocabulary."""

    def __init__(
        self,
        *,
        registry: ArtifactRegistry | None = None,
        downloads_dir: Path | None = None,
        progress_callback: ArtifactProgressCallback | None = None,
    ) -> None:
        """Create an ORB-SLAM3 vocabulary fetch strategy."""
        self.registry = registry or ArtifactRegistry()
        self.downloads_dir = self.registry.resolve_path(downloads_dir or DOWNLOADS_DIR)
        self.progress_callback = progress_callback

    def plan(self, manifest: ArtifactManifest, *, force: bool = False) -> list[ArtifactFetchPlanAction]:
        """Return operations needed to fetch and convert the ORB vocabulary."""
        source = _require_source(manifest)
        output_file_id = _output_file_id(manifest)
        output_status = self.registry.local_status(manifest, verify_hashes=True).files[output_file_id]
        if output_status.verified and not force:
            return []

        archive_path = self._archive_path(manifest)
        text_path = self._text_path(manifest)
        actions: list[ArtifactFetchPlanAction] = []

        if force or not self._source_archive_ready(manifest, archive_path):
            actions.append(
                ArtifactFetchPlanAction(
                    name="Download source vocabulary archive",
                    detail=f"{source.download_url} -> {archive_path}",
                    download_bytes=source.size_bytes,
                )
            )
        if force or not text_path.exists():
            actions.append(
                ArtifactFetchPlanAction(
                    name="Extract ORB vocabulary text",
                    detail=f"{_archive_member(manifest)} -> {text_path}",
                )
            )
        if force or not output_status.verified:
            actions.append(
                ArtifactFetchPlanAction(
                    name="Convert ORB vocabulary to DBoW3 binary",
                    detail=f"{text_path} -> {output_status.path}",
                )
            )

        return actions

    def ensure(self, manifest: ArtifactManifest, *, force: bool = False) -> ArtifactFetchResult:
        """Fetch ORB-SLAM3's compressed text vocabulary and convert it to DBoW3 binary format."""
        status = self.registry.local_status(manifest, verify_hashes=True)
        output_file_id = _output_file_id(manifest)
        if status.verified and not force:
            return ArtifactFetchResult(
                artifact=manifest.name,
                fetched=False,
                extracted=False,
                converted=False,
                ready=True,
            )

        archive_path, fetched = self._ensure_archive(manifest, force=force)
        text_path, extracted = self._ensure_text_vocabulary(manifest, archive_path, force=force)
        output_path = self.registry.resolve_file_path(manifest, output_file_id)
        converted = force or not status.files[output_file_id].verified
        if converted:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _convert_txt_to_dbow3(text_path, output_path)

        final_status = self.registry.local_status(manifest, verify_hashes=True)
        if not final_status.verified:
            issues = ", ".join(str(path) for path in final_status.issues)
            msg = f"Artifact '{manifest.name}' is incomplete after fetch: {issues}"
            raise ArtifactFetchError(msg)

        return ArtifactFetchResult(
            artifact=manifest.name,
            fetched=fetched,
            extracted=extracted,
            converted=converted,
            ready=True,
        )

    def _ensure_archive(self, manifest: ArtifactManifest, *, force: bool) -> tuple[Path, bool]:
        archive_path = self._archive_path(manifest)
        if not force and self._source_archive_ready(manifest, archive_path):
            return archive_path, False

        source = _require_source(manifest)
        if source.download_url is None:
            msg = f"Artifact '{manifest.name}' is missing source.download_url"
            raise ArtifactFetchError(msg)

        scheme = urlparse(source.download_url).scheme
        if scheme not in {"http", "https"}:
            msg = f"Unsupported download URL scheme for '{manifest.name}': {scheme or '<none>'}"
            raise ArtifactFetchError(msg)

        archive_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = archive_path.with_name(f"{archive_path.name}.tmp")
        request = Request(  # noqa: S310 - schemes are restricted immediately above.
            source.download_url,
            headers={"User-Agent": "vins-rnd-artifact-fetch/1.0"},
        )
        try:
            with urlopen(request) as response, tmp_path.open("wb") as file:  # noqa: S310
                total_bytes = source.size_bytes or _content_length(response)
                self._copy_response(response, file, total_bytes)
        except HTTPError as exc:
            msg = f"Could not download artifact '{manifest.name}': HTTP {exc.code} {exc.reason}"
            raise ArtifactFetchError(msg) from exc
        except URLError as exc:
            msg = f"Could not download artifact '{manifest.name}': {exc.reason}"
            raise ArtifactFetchError(msg) from exc

        if source.sha256 is not None:
            _verify_sha256(tmp_path, source.sha256, f"source archive for '{manifest.name}'")
        tmp_path.replace(archive_path)
        return archive_path, True

    def _ensure_text_vocabulary(
        self,
        manifest: ArtifactManifest,
        archive_path: Path,
        *,
        force: bool,
    ) -> tuple[Path, bool]:
        text_path = self._text_path(manifest)
        if text_path.exists() and not force:
            return text_path, False

        text_path.parent.mkdir(parents=True, exist_ok=True)
        member_name = _archive_member(manifest)
        try:
            with tarfile.open(archive_path, "r:gz") as archive:
                member = archive.getmember(member_name)
                member_file = archive.extractfile(member)
                if member_file is None:
                    msg = f"Archive member '{member_name}' is not a regular file"
                    raise ArtifactFetchError(msg)
                with member_file, text_path.open("wb") as file:
                    shutil.copyfileobj(member_file, file)
        except KeyError as exc:
            msg = f"Archive '{archive_path}' does not contain '{member_name}'"
            raise ArtifactFetchError(msg) from exc
        except tarfile.TarError as exc:
            msg = f"Could not extract artifact '{manifest.name}' archive '{archive_path}': {exc}"
            raise ArtifactFetchError(msg) from exc
        return text_path, True

    def _copy_response(self, response, file, total_bytes: int | None) -> None:  # noqa: ANN001
        downloaded_bytes = 0
        if self.progress_callback is not None:
            self.progress_callback("Downloading source artifact", downloaded_bytes, total_bytes)
        while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
            file.write(chunk)
            downloaded_bytes += len(chunk)
            if self.progress_callback is not None:
                self.progress_callback("Downloading source artifact", downloaded_bytes, total_bytes)

    def _source_archive_ready(self, manifest: ArtifactManifest, archive_path: Path) -> bool:
        if not archive_path.exists():
            return False
        source = _require_source(manifest)
        return source.sha256 is None or _sha256_file(archive_path) == source.sha256

    def _archive_path(self, manifest: ArtifactManifest) -> Path:
        source = _require_source(manifest)
        filename = source.filename or f"{manifest.name}.tar.gz"
        return self.downloads_dir / manifest.name / filename

    def _text_path(self, manifest: ArtifactManifest) -> Path:
        member_name = _archive_member(manifest)
        text_filename = Path(member_name).name
        return self.downloads_dir / manifest.name / text_filename


def _require_source(manifest: ArtifactManifest) -> ArtifactSourceConfig:
    source = manifest.source
    if source is None:
        msg = f"Artifact '{manifest.name}' is missing source metadata"
        raise ArtifactFetchError(msg)
    return source


def _archive_member(manifest: ArtifactManifest) -> str:
    value = manifest.metadata.get("archive_member", DEFAULT_ORB_SLAM3_ARCHIVE_MEMBER)
    if not isinstance(value, str) or not value:
        msg = f"Artifact '{manifest.name}' metadata.archive_member must be a non-empty string"
        raise ArtifactFetchError(msg)
    return value


def _output_file_id(manifest: ArtifactManifest) -> str:
    value = manifest.metadata.get("output_file_id", DEFAULT_ORB_SLAM3_OUTPUT_FILE_ID)
    if not isinstance(value, str) or value not in manifest.files:
        msg = f"Artifact '{manifest.name}' metadata.output_file_id must reference a declared file"
        raise ArtifactFetchError(msg)
    return value


def _content_length(response) -> int | None:  # noqa: ANN001
    value = response.headers.get("Content-Length")
    return int(value) if value is not None else None


def _verify_sha256(path: Path, expected_sha256: str, label: str) -> None:
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        msg = f"SHA-256 mismatch for {label}: expected {expected_sha256}, got {actual_sha256}"
        raise ArtifactFetchError(msg)


def _convert_txt_to_dbow3(text_path: Path, output_path: Path) -> None:
    try:
        pydbow3 = importlib.import_module("pydbow3")
    except ImportError as exc:
        msg = "pydbow3 is required to convert ORBvoc.txt into ORBvoc.dbow3. Run `just install-pydbow3`."
        raise ArtifactFetchError(msg) from exc

    vocabulary = pydbow3.Vocabulary()
    vocabulary.load(str(text_path))
    vocabulary.save(str(output_path), SAVE_BINARY_COMPRESSED)
