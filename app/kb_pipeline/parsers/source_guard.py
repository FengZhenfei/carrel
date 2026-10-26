"""Recognize source placeholders before calling any document parser."""
from pathlib import Path
import re

from .errors import NonRetryableParseError


class MissingSourceContent(NonRetryableParseError):
    """The current source contains a reference instead of the document's bytes."""

_LFS = re.compile(
    rb"\Aversion https://git-lfs.github.com/spec/v1\r?\n"
    rb"(?:ext-[^\r\n]+\r?\n)*oid sha256:([0-9a-f]{64})\r?\n"
    rb"size ([0-9]+)\r?\n?\Z"
)


def git_lfs_pointer(path: Path) -> dict | None:
    with path.open("rb") as handle:
        data = handle.read(1025)
    match = _LFS.fullmatch(data) if len(data) <= 1024 else None
    return {"oid": match[1].decode("ascii"), "size": int(match[2])} if match else None


def require_source_content(path: Path) -> None:
    pointer = git_lfs_pointer(path)
    if pointer:
        raise MissingSourceContent(
            f"Git LFS original file missing: {path.name} is a pointer, not document content "
            f"(expected {pointer['size']} bytes, sha256:{pointer['oid']}). "
            "Replace it with the original file to parse it."
        )
