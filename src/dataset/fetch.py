from __future__ import annotations

import hashlib
import shutil
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, TYPE_CHECKING, BinaryIO, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zipfile import ZipFile

from dataset.registry import DatasetRegistry

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dataset.builder import DatasetBuilder
    from dataset.manifest import DatasetManifest


DOWNLOADS_DIR = Path("datasets/.downloads")
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
HTTP_SUCCESS_MIN = 200
HTTP_SUCCESS_MAX = 400

type DownloadProgressCallback = Callable[[int, int | None], None]
type DatasetProgressCallback = Callable[[str, int, int | None], None]


class _ResponseHeaders(Protocol):
    def get(self, name: str, default: str | None = None) -> str | None:
        """Return a response header value."""


class _ResponseHeaderItems(Protocol):
    def items(self) -> Iterable[tuple[object, object]]:
        """Return all response header pairs."""


class _DownloadResponse(Protocol):
    headers: _ResponseHeaders

    def read(self, size: int) -> bytes:
        """Read up to size bytes from the response body."""


class DatasetFetchError(RuntimeError):
    """Raised when a dataset cannot be fetched or prepared."""


@dataclass(frozen=True)
class DatasetFetchResult:
    """Outcome of preparing a dataset for local use."""

    dataset: str
    root: Path
    cache_path: Path
    raw_fetched: bool
    raw_ready: bool
    cache_built: bool
    cache_ready: bool


@dataclass(frozen=True)
class DatasetFetchPlanAction:
    """One operation that will be executed while preparing a dataset."""

    name: str
    detail: str | None = None
    download_bytes: int | None = 0


@dataclass(frozen=True)
class DatasetFetchPlan:
    """Preflight description of dataset preparation operations."""

    dataset: str
    actions: tuple[DatasetFetchPlanAction, ...]
    dataset_page_url: str | None = None
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


@dataclass(frozen=True)
class DownloadHeadResult:
    """HTTP HEAD metadata for a dataset download URL."""

    url: str
    status: int
    reason: str
    headers: dict[str, str]


class DatasetFetchStrategy(Protocol):
    """Strategy that can populate raw files for one dataset family."""

    def ensure_raw(self, manifest: DatasetManifest, *, force: bool = False) -> bool:
        """Ensure raw stream files exist. Returns true when files were fetched."""


class EurocFetchStrategy:
    """Fetch EuRoC raw files from the official archive declared by the manifest."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        registry: DatasetRegistry | None = None,
        downloads_dir: Path | None = None,
        progress_callback: DatasetProgressCallback | None = None,
    ) -> None:
        """Create a EuRoC fetch strategy."""
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.registry = registry or DatasetRegistry(repo_root=self.repo_root)
        self.downloads_dir = self._resolve_path(downloads_dir or DOWNLOADS_DIR)
        self.progress_callback = progress_callback

    def ensure_raw(self, manifest: DatasetManifest, *, force: bool = False) -> bool:
        """Download and extract raw EuRoC files when local stream files are missing."""
        status = self.registry.local_status(manifest)
        if status.verified and not force:
            return False

        source = manifest.source
        if source is None or source.download_url is None:
            msg = f"Dataset '{manifest.name}' is missing source.download_url; run `just ds open {manifest.name}`."
            raise DatasetFetchError(msg)

        archive_path = self._download_archive(manifest)
        self._extract_archive(manifest, archive_path)

        final_status = self.registry.local_status(manifest)
        if not final_status.verified:
            missing = ", ".join(str(path) for path in final_status.issues)
            msg = f"Dataset '{manifest.name}' is still incomplete after fetch: {missing}"
            raise DatasetFetchError(msg)
        return True

    def plan_raw_actions(self, manifest: DatasetManifest, *, force: bool = False) -> list[DatasetFetchPlanAction]:
        """Describe raw-file preparation steps that would be executed."""
        status = self.registry.local_status(manifest)
        if status.verified and not force:
            return []

        source = manifest.source
        if source is None or source.download_url is None:
            msg = f"Dataset '{manifest.name}' is missing source.download_url; run `just ds open {manifest.name}`."
            raise DatasetFetchError(msg)

        archive_path = self.downloads_dir / (source.filename or f"{manifest.name}.zip")
        actions: list[DatasetFetchPlanAction] = []
        if not archive_path.exists():
            actions.append(
                DatasetFetchPlanAction(
                    name="Download source archive",
                    detail=f"{source.download_url} -> {archive_path}",
                    download_bytes=source.size_bytes,
                )
            )

        archive = source.archive
        if archive is not None and archive.nested is not None:
            nested_path = self.downloads_dir / archive.nested.name
            if not nested_path.exists():
                actions.append(
                    DatasetFetchPlanAction(
                        name="Extract nested archive",
                        detail=f"{archive.nested} -> {nested_path}",
                    )
                )

        archive_root = archive.root if archive is not None and archive.root is not None else "<auto>"
        actions.append(
            DatasetFetchPlanAction(
                name="Extract dataset files",
                detail=f"{archive_root} -> {manifest.root}",
            )
        )
        return actions

    def _download_archive(self, manifest: DatasetManifest) -> Path:
        source = manifest.source
        if source is None or source.download_url is None:
            msg = f"Dataset '{manifest.name}' is missing source.download_url"
            raise DatasetFetchError(msg)

        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        archive_path = self.downloads_dir / (source.filename or f"{manifest.name}.zip")
        if archive_path.exists():
            self._verify_sha256(archive_path, manifest)
            return archive_path

        scheme = urlparse(source.download_url).scheme
        if scheme not in {"https", "http", "file"}:
            msg = f"Unsupported download URL scheme for '{manifest.name}': {scheme or '<none>'}"
            raise DatasetFetchError(msg)

        errors: list[str] = []
        tmp_path = archive_path.with_suffix(f"{archive_path.suffix}.part")
        for download_url in self._candidate_download_urls(source.download_url):
            request = Request(  # noqa: S310
                download_url,
                headers={
                    "Accept": "application/octet-stream,*/*",
                    "User-Agent": "vins-rnd-dataset-fetch/1.0",
                },
            )
            try:
                with urlopen(request) as response, tmp_path.open("wb") as file:  # noqa: S310
                    total_bytes = source.size_bytes or self._content_length(response)
                    if total_bytes is None and self.progress_callback is not None:
                        total_bytes = probe_download_size(download_url)
                    self._copy_response(response, file, total_bytes=total_bytes)
            except HTTPError as exc:
                tmp_path.unlink(missing_ok=True)
                errors.append(f"{download_url} -> HTTP {exc.code} {exc.reason}")
                continue
            except URLError as exc:
                tmp_path.unlink(missing_ok=True)
                errors.append(f"{download_url} -> {exc.reason}")
                continue
            break
        else:
            details = "; ".join(errors)
            msg = f"Could not download dataset '{manifest.name}': {details}"
            raise DatasetFetchError(msg)

        tmp_path.replace(archive_path)
        self._verify_sha256(archive_path, manifest)
        return archive_path

    def _copy_response(
        self,
        response: _DownloadResponse,
        file: BinaryIO,
        *,
        total_bytes: int | None = None,
    ) -> None:
        total_bytes = total_bytes or self._content_length(response)
        downloaded_bytes = 0
        if self.progress_callback is not None and total_bytes is not None:
            self.progress_callback("Downloading source archive", downloaded_bytes, total_bytes)
        while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
            file.write(chunk)
            downloaded_bytes += len(chunk)
            if self.progress_callback is not None:
                self.progress_callback("Downloading source archive", downloaded_bytes, total_bytes)

    @staticmethod
    def _content_length(response: _DownloadResponse) -> int | None:
        raw_length = response.headers.get("Content-Length")
        if raw_length is None:
            return None
        try:
            return int(raw_length)
        except ValueError:
            return None

    @staticmethod
    def _candidate_download_urls(download_url: str) -> list[str]:
        return candidate_download_urls(download_url)

    def _verify_sha256(self, archive_path: Path, manifest: DatasetManifest) -> None:
        source = manifest.source
        if source is None or source.sha256 is None:
            return

        digest = hashlib.sha256()
        with archive_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)

        actual = digest.hexdigest()
        if actual != source.sha256:
            msg = f"Checksum mismatch for {archive_path}: expected {source.sha256}, got {actual}"
            raise DatasetFetchError(msg)

    def _extract_archive(self, manifest: DatasetManifest, archive_path: Path) -> None:
        source = manifest.source
        archive = source.archive if source is not None else None
        if archive is not None and archive.format != "zip":
            msg = f"Unsupported archive format for '{manifest.name}': {archive.format}"
            raise DatasetFetchError(msg)

        root = self._resolve_path(manifest.root)
        root.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(archive_path) as zip_file:
            try:
                self._extract_manifest_from_zip(zip_file, manifest, root)
            except DatasetFetchError:
                nested_member = self._find_nested_archive_member(zip_file, manifest)
                if nested_member is None:
                    raise

                nested_path = self.downloads_dir / PurePosixPath(nested_member).name
                if not nested_path.exists():
                    self.downloads_dir.mkdir(parents=True, exist_ok=True)
                    nested_info = zip_file.getinfo(nested_member)
                    with zip_file.open(nested_member) as source_file, nested_path.open("wb") as target_file:
                        self._copy_stream(
                            source_file,
                            target_file,
                            label="Extracting nested archive",
                            total_bytes=nested_info.file_size,
                        )
            else:
                return

        with ZipFile(nested_path) as nested_zip_file:
            self._extract_manifest_from_zip(nested_zip_file, manifest, root)

    def _extract_manifest_from_zip(self, zip_file: ZipFile, manifest: DatasetManifest, root: Path) -> None:
        archive_root = self._resolve_archive_root(zip_file, manifest)
        with tempfile.TemporaryDirectory(prefix=f".{manifest.name}.", dir=str(root.parent)) as tmp_dir:
            tmp_root = Path(tmp_dir)
            self._extract_zip_root(zip_file, archive_root, tmp_root)
            root.mkdir(parents=True, exist_ok=True)
            shutil.copytree(tmp_root, root, dirs_exist_ok=True)

    @staticmethod
    def _find_nested_archive_member(zip_file: ZipFile, manifest: DatasetManifest) -> str | None:
        source = manifest.source
        archive = source.archive if source is not None else None
        nested = (
            archive.nested.as_posix().strip("/") if archive is not None and archive.nested is not None else None
        )
        if nested is None:
            return None

        names = [info.filename for info in zip_file.infolist() if not info.is_dir()]
        return next((name for name in names if name == nested or name.endswith(f"/{nested}")), None)

    def _resolve_archive_root(self, zip_file: ZipFile, manifest: DatasetManifest) -> str:
        names = [info.filename for info in zip_file.infolist() if not info.is_dir()]
        source = manifest.source
        archive_root = source.archive.root if source is not None and source.archive is not None else None
        configured_root = archive_root.as_posix().strip("/") if archive_root is not None else None

        if configured_root is not None:
            resolved_root = self._match_configured_root(names, configured_root)
            if resolved_root is not None:
                return resolved_root

        detected_root = self._detect_archive_root(names, manifest)
        if detected_root is None:
            msg = f"Could not find raw stream layout for '{manifest.name}' inside {zip_file.filename}"
            raise DatasetFetchError(msg)
        return detected_root

    @staticmethod
    def _match_configured_root(names: list[str], configured_root: str) -> str | None:
        if any(name == configured_root or name.startswith(f"{configured_root}/") for name in names):
            return configured_root

        suffix = f"/{configured_root}/"
        for name in names:
            if suffix in name:
                return name.split(suffix, maxsplit=1)[0] + f"/{configured_root}"
        return None

    @staticmethod
    def _detect_archive_root(names: list[str], manifest: DatasetManifest) -> str | None:
        stream_paths = [Path(path).as_posix() for path in manifest.streams.model_dump(exclude_none=True).values()]
        candidates: set[str] | None = None
        for stream_path in stream_paths:
            suffix = f"/{stream_path}"
            stream_candidates = {name.removesuffix(suffix) for name in names if name.endswith(suffix)} | (
                {""} if stream_path in names else set()
            )
            candidates = stream_candidates if candidates is None else candidates & stream_candidates

        if not candidates:
            return None
        return min(candidates, key=lambda candidate: (len(candidate), candidate))

    def _extract_zip_root(self, zip_file: ZipFile, archive_root: str, target_root: Path) -> None:
        prefix = f"{archive_root}/" if archive_root else ""
        members = [info for info in zip_file.infolist() if not info.is_dir() and info.filename.startswith(prefix)]
        total_bytes = sum(info.file_size for info in members)
        copied_bytes = 0
        label = "Extracting dataset files"
        if self.progress_callback is not None:
            self.progress_callback(label, copied_bytes, total_bytes)

        for info in members:
            relative_name = info.filename.removeprefix(prefix)
            if not relative_name:
                continue

            relative_path = PurePosixPath(relative_name)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                msg = f"Unsafe archive member path: {info.filename}"
                raise DatasetFetchError(msg)

            target_path = target_root / Path(*relative_path.parts)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zip_file.open(info) as source, target_path.open("wb") as target:
                copied_bytes = self._copy_stream(
                    source,
                    target,
                    label=label,
                    total_bytes=total_bytes,
                    completed_bytes=copied_bytes,
                )

    def _copy_stream(
        self,
        source: IO[bytes],
        target: IO[bytes],
        *,
        label: str,
        total_bytes: int | None,
        completed_bytes: int = 0,
    ) -> int:
        if self.progress_callback is not None and total_bytes is not None and completed_bytes == 0:
            self.progress_callback(label, completed_bytes, total_bytes)

        while chunk := source.read(DOWNLOAD_CHUNK_BYTES):
            target.write(chunk)
            completed_bytes += len(chunk)
            if self.progress_callback is not None:
                self.progress_callback(label, completed_bytes, total_bytes)
        return completed_bytes

    def _resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else self.repo_root / path


def fetch_download_headers(download_url: str) -> list[DownloadHeadResult]:
    """Fetch HEAD metadata for a download URL and known fallback candidates."""
    scheme = urlparse(download_url).scheme
    if scheme not in {"https", "http"}:
        msg = f"Unsupported HEAD URL scheme: {scheme or '<none>'}"
        raise DatasetFetchError(msg)

    return [_fetch_url_headers(url) for url in candidate_download_urls(download_url)]


def probe_download_size(download_url: str) -> int | None:
    """Best-effort remote download size lookup without fetching the archive body."""
    scheme = urlparse(download_url).scheme
    if scheme not in {"https", "http"}:
        return None

    try:
        head = _fetch_url_headers(download_url)
    except DatasetFetchError:
        head = None

    if head is not None and HTTP_SUCCESS_MIN <= head.status < HTTP_SUCCESS_MAX:
        content_length = _parse_int_header(head.headers.get("Content-Length"))
        if content_length is not None:
            return content_length

    return _probe_range_download_size(download_url)


def candidate_download_urls(download_url: str) -> list[str]:
    """Return the primary download URL plus known fallback URL candidates."""
    parsed = urlparse(download_url)
    parts = parsed.path.strip("/").split("/")
    match parts:
        case ["bitstreams", bitstream_id, "download"]:
            rest_url = f"{parsed.scheme}://{parsed.netloc}/server/api/core/bitstreams/{bitstream_id}/content"
            return [download_url, rest_url]
        case _:
            return [download_url]


def _fetch_url_headers(download_url: str) -> DownloadHeadResult:
    request = Request(  # noqa: S310
        download_url,
        method="HEAD",
        headers={
            "Accept": "application/octet-stream,*/*",
            "User-Agent": "vins-rnd-dataset-fetch/1.0",
        },
    )
    try:
        with urlopen(request) as response:  # noqa: S310
            status = _response_status(response)
            return DownloadHeadResult(
                url=download_url,
                status=status,
                reason=_response_reason(response, status),
                headers=_headers_dict(response.headers),
            )
    except HTTPError as exc:
        return DownloadHeadResult(
            url=download_url,
            status=exc.code,
            reason=str(exc.reason),
            headers=_headers_dict(exc.headers),
        )
    except URLError as exc:
        msg = f"Could not fetch headers for {download_url}: {exc.reason}"
        raise DatasetFetchError(msg) from exc


def _probe_range_download_size(download_url: str) -> int | None:
    request = Request(  # noqa: S310
        download_url,
        headers={
            "Accept": "application/octet-stream,*/*",
            "Range": "bytes=0-0",
            "User-Agent": "vins-rnd-dataset-fetch/1.0",
        },
    )
    try:
        with urlopen(request) as response:  # noqa: S310
            return _parse_content_range_total(response.headers.get("Content-Range")) or _parse_int_header(
                response.headers.get("Content-Length")
            )
    except HTTPError as exc:
        return _parse_content_range_total(exc.headers.get("Content-Range"))
    except URLError:
        return None


def _response_status(response: object) -> int:
    status_value = getattr(response, "status", None)
    if status_value is None:
        getcode = getattr(response, "getcode", None)
        if not callable(getcode):
            return 0
        status_value = getcode()
    return int(status_value)


def _response_reason(response: object, status: int) -> str:
    reason = getattr(response, "reason", "")
    if isinstance(reason, str) and reason and not reason.isdecimal():
        return reason
    return _http_status_phrase(status)


def _http_status_phrase(status: int) -> str:
    match status:
        case 200:
            return "OK"
        case 206:
            return "Partial Content"
        case 404:
            return "Not Found"
        case 500:
            return "Internal Server Error"
        case _:
            return ""


def _parse_int_header(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _parse_content_range_total(value: str | None) -> int | None:
    if value is None:
        return None
    _range, _separator, total = value.rpartition("/")
    if not total.isdecimal():
        return None
    return int(total)


def _headers_dict(headers: _ResponseHeaderItems) -> dict[str, str]:
    return {str(name): str(value) for name, value in headers.items()}


class DatasetCacheManager:
    """Ensure raw dataset files and materialized cache exist before use."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        repo_root: Path | None = None,
        dataset_dir: Path | None = None,
        registry: DatasetRegistry | None = None,
        downloads_dir: Path | None = None,
        progress_callback: DatasetProgressCallback | None = None,
        strategies: Mapping[str, DatasetFetchStrategy] | None = None,
        builders: Mapping[str, DatasetBuilder] | None = None,
    ) -> None:
        """Create a dataset cache manager."""
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.registry = registry or DatasetRegistry(repo_root=self.repo_root, dataset_dir=dataset_dir)
        self.downloads_dir = downloads_dir
        self.progress_callback = progress_callback
        self.strategies = dict(strategies or {})
        self.builders = dict(builders or {})

    def ensure_dataset(
        self,
        manifest: DatasetManifest,
        *,
        fetch_raw: bool = True,
        force_raw: bool = False,
        force_cache: bool = False,
        materialize_cache: bool = True,
    ) -> DatasetFetchResult:
        """Ensure raw files and, by default, the materialized cache are available."""
        status = self.registry.local_status(manifest)
        raw_fetched = False
        if force_raw or not status.verified:
            if not fetch_raw:
                missing = ", ".join(str(path) for path in status.issues)
                msg = f"Dataset '{manifest.name}' raw files are incomplete: {missing}"
                raise DatasetFetchError(msg)
            raw_fetched = self._strategy_for(manifest).ensure_raw(manifest, force=force_raw)
            status = self.registry.local_status(manifest)

        if not status.verified:
            missing = ", ".join(str(path) for path in status.issues)
            msg = f"Dataset '{manifest.name}' raw files are incomplete: {missing}"
            raise DatasetFetchError(msg)

        cache_built = False
        if materialize_cache and (force_cache or not status.cache_verified):
            if self.progress_callback is not None:
                self.progress_callback("Building dataset cache", 0, 1)
            self._builder_for(manifest).build(manifest)
            if self.progress_callback is not None:
                self.progress_callback("Building dataset cache", 1, 1)
            cache_built = True
            status = self.registry.local_status(manifest)
            if not status.cache_verified:
                missing = ", ".join(str(path) for path in status.cache_issues)
                msg = f"Dataset '{manifest.name}' cache is incomplete: {missing}"
                raise DatasetFetchError(msg)

        return DatasetFetchResult(
            dataset=manifest.name,
            root=self.registry.resolve_path(manifest.root),
            cache_path=status.cache_path,
            raw_fetched=raw_fetched,
            raw_ready=status.verified,
            cache_built=cache_built,
            cache_ready=status.cache_verified,
        )

    def plan_dataset(
        self,
        manifest: DatasetManifest,
        *,
        fetch_raw: bool = True,
        force_raw: bool = False,
        force_cache: bool = False,
        materialize_cache: bool = True,
    ) -> DatasetFetchPlan:
        """Describe the dataset preparation steps without executing them."""
        status = self.registry.local_status(manifest)
        actions: list[DatasetFetchPlanAction] = []
        if force_raw or not status.verified:
            if not fetch_raw:
                missing = ", ".join(str(path) for path in status.issues)
                msg = f"Dataset '{manifest.name}' raw files are incomplete: {missing}"
                raise DatasetFetchError(msg)
            actions.extend(self._plan_raw_actions(manifest, force=force_raw))

        if materialize_cache and (force_cache or not status.cache_verified):
            actions.append(
                DatasetFetchPlanAction(
                    name="Build dataset cache",
                    detail=str(status.cache_path),
                )
            )

        source = manifest.source
        page_url = source.page_url if source is not None else None
        return DatasetFetchPlan(
            dataset=manifest.name,
            actions=tuple(actions),
            dataset_page_url=page_url,
            open_command=f"just dataset open {manifest.name}",
        )

    def _plan_raw_actions(self, manifest: DatasetManifest, *, force: bool = False) -> list[DatasetFetchPlanAction]:
        strategy = self._strategy_for(manifest)
        if isinstance(strategy, EurocFetchStrategy):
            return strategy.plan_raw_actions(manifest, force=force)
        return [DatasetFetchPlanAction(name="Prepare raw dataset files", detail=manifest.type)]

    def _strategy_for(self, manifest: DatasetManifest) -> DatasetFetchStrategy:
        strategy = self.strategies.get(manifest.type)
        if strategy is not None:
            return strategy
        if manifest.type == "euroc":
            return EurocFetchStrategy(
                repo_root=self.repo_root,
                registry=self.registry,
                downloads_dir=self.downloads_dir,
                progress_callback=self.progress_callback,
            )

        msg = f"Unsupported dataset fetch strategy '{manifest.type}'"
        raise DatasetFetchError(msg)

    def _builder_for(self, manifest: DatasetManifest) -> DatasetBuilder:
        builder = self.builders.get(manifest.type)
        if builder is not None:
            return builder
        if manifest.type == "euroc":
            from dataset.euroc import EurocDatasetBuilder  # noqa: PLC0415

            return EurocDatasetBuilder(repo_root=self.repo_root)

        msg = f"Unsupported dataset cache builder '{manifest.type}'"
        raise DatasetFetchError(msg)
