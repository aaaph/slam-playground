from __future__ import annotations

from pathlib import Path  # noqa: TC003 - pydantic needs Path in the runtime model namespace.
from string import hexdigits
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator, model_validator

SHA256_HEX_LENGTH = 64


class ArtifactSourceConfig(BaseModel):
    """Download and documentation metadata for an artifact manifest."""

    page_url: str | None = None
    download_url: str | None = None
    filename: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    license: str | None = None
    notes: str | None = None

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str | None) -> str | None:
        """Normalize optional SHA-256 checksums."""
        return _validate_sha256(value)


class ArtifactFileConfig(BaseModel):
    """One file that belongs to an artifact."""

    path: Path
    description: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    optional: bool = False

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str | None) -> str | None:
        """Normalize optional SHA-256 checksums."""
        return _validate_sha256(value)


class ArtifactManifest(BaseModel):
    """Manifest for a local research artifact such as a vocabulary, model, or weight file."""

    name: str
    type: str
    root: Path
    version: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    files: dict[str, ArtifactFileConfig] = Field(default_factory=dict)
    source: ArtifactSourceConfig | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_files(self) -> Self:
        """Require manifests to declare at least one concrete payload file."""
        if not self.files:
            msg = "Artifact manifest must declare at least one file"
            raise ValueError(msg)
        return self


def _validate_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    checksum = value.lower()
    if len(checksum) != SHA256_HEX_LENGTH or any(char not in hexdigits for char in checksum):
        msg = "SHA-256 checksum must be 64 hexadecimal characters"
        raise ValueError(msg)
    return checksum
