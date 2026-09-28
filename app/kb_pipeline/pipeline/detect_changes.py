from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .. import db
from ..models import SourceFile
from ..parsers.common import parser_profile_for_path


ChangeType = Literal["new", "content_changed", "parser_changed", "metadata_changed", "needs_parse", "unchanged"]


@dataclass(frozen=True)
class Change:
    change_type: ChangeType
    job_type: str | None
    reason: str


def detect_change(old_row, file: SourceFile) -> Change:
    if old_row is None:
        return Change("new", "parse", "new file")
    if str(old_row["status"]) == "deleted":
        old_content_version = str(old_row["content_version"] or "")
        new_content_version = file.content_version
        old_indexed_version = str(old_row["indexed_version"] or "")
        if old_content_version == new_content_version and old_indexed_version == new_content_version and int(old_row["size"]) == file.size:
            return Change("metadata_changed", "metadata_update", "file restored after delete")
        return Change("new", "parse", "file restored after delete")

    old_content_version = str(old_row["content_version"] or "")
    new_content_version = file.content_version
    if old_content_version != new_content_version or int(old_row["size"]) != file.size:
        return Change("content_changed", "parse", "checksum/size changed")
    old_indexed_version = str(old_row["indexed_version"] or "")
    if old_indexed_version != new_content_version:
        return Change("needs_parse", "parse", "current content version is not indexed")

    old_profile = ""
    if "indexed_parser_profile" in old_row.keys():
        old_profile = str(old_row["indexed_parser_profile"] or "")
    if not old_profile:
        # Old data did not record the profile used at the time. Deriving it from the current rules necessarily
        # equals the new profile, so these rows are never re-parsed because of a parser rule upgrade -- keep
        # that behaviour (avoid rerunning every historical file after one rule tweak), but call the situation
        # out so it can be handled explicitly with "re-parse the whole knowledge base" when needed.
        old_profile = parser_profile_for_path(Path(str(old_row["filename"])))
    new_profile = parser_profile_for(file)
    if old_profile != new_profile:
        return Change("parser_changed", "parse", f"parser profile changed: {old_profile} -> {new_profile}")

    old_metadata = str(old_row["metadata_fingerprint"] or "")
    new_metadata = db.metadata_fingerprint(file)
    if old_metadata != new_metadata:
        parts: list[str] = []
        if str(old_row["rel_path"]) != file.rel_path or str(old_row["filename"]) != file.filename:
            parts.append("path/name changed")
        return Change("metadata_changed", "metadata_update", ", ".join(parts) or "metadata changed")

    return Change("unchanged", None, "unchanged")


def parser_profile_for(file: SourceFile) -> str:
    return parser_profile_for_path(Path(file.filename))
