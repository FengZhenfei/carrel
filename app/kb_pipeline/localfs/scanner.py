from __future__ import annotations

import hashlib
import mimetypes
import time
from pathlib import Path

from ..models import KBSource, SourceFile
from ..utils import normalize_rel_path, sha256_file
from ..vision.images import IMAGE_SUFFIXES


SUPPORTED_EXTS = {
    ".txt", ".md", ".markdown", ".pdf", ".pptx", ".docx", ".xlsx", ".xls", ".csv",
    # Code suffixes aligned with parsers.code_symbols.LANGUAGE_BY_SUFFIX (.mjs/.cjs/.lua/.cxx/.hh added in
    # health check B8)
    ".py", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".c", ".cc", ".cpp", ".cxx",
    ".h", ".hh", ".hpp", ".cs", ".php", ".rb", ".lua", ".swift", ".kt", ".kts", ".scala", ".sql",
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".yaml", ".yml", ".json", ".toml",
    ".ini", ".cfg", ".conf", ".xml", ".html", ".css", ".scss", ".vue", ".svelte",
    # standalone pictures: VLM description + visual vector (parsers.image_file)
    *IMAGE_SUFFIXES,
}
SUPPORTED_FILENAMES = {"Dockerfile", "Makefile", "Gemfile", "Rakefile", "Jenkinsfile"}


def list_source_files(
    source: KBSource,
    *,
    limit: int | None = None,
    min_age_seconds: int = 0,
    hash_content: bool = True,
    known_files: dict[str, tuple[int, int, str]] | None = None,
    too_recent_keys: set[int] | None = None,
) -> list[SourceFile]:
    """Walk one KB's mirror directory.

    `too_recent_keys`, when given, collects the file_key of every file skipped
    for being younger than `min_age_seconds`. Those files are present on disk
    and merely deferred to a later scan, so the caller must still count them as
    "seen" -- otherwise delete detection reads a freshly edited file as gone and
    tears down its points before the next scan restores them.

    `known_files` maps rel_path -> (size, mtime, checksum) as already recorded
    in the state DB. It is keyed by path rather than file_key because move
    detection deliberately keeps a moved file's original key, which would then
    no longer match the key derived from its new path. When a file still matches its recorded size and mtime, the
    stored checksum is reused instead of re-reading the whole file. That is the
    same quick check rsync uses to decide whether to copy at all, so a change
    invisible to it would never have reached the mirror to begin with. Anything
    that genuinely needs a fresh hash -- new, moved or changed files -- misses
    the check and gets hashed.
    """
    if source.physical_base is None:
        return []
    root = source.physical_base.expanduser()
    if not root.exists():
        return []
    boundary = root.resolve()

    now = time.time()
    result: list[SourceFile] = []
    reused = hashed = outside = 0
    for path in sorted(root.rglob("*")):
        if limit is not None and len(result) >= limit:
            break
        if not path.is_file() or should_skip(path):
            continue
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTS and path.name not in SUPPORTED_FILENAMES:
            continue
        if not inside_boundary(path, boundary):
            outside += 1          # a symlink (or a file under one) that leaves the enrolled directory
            continue
        # The mirror is rsync's working directory; a file can vanish between
        # rglob and stat/hash. One racing delete must not abort the whole
        # scan run -- the next scan settles it (list_recent_source_files
        # already tolerates this the same way).
        try:
            stat = path.stat()
        except (FileNotFoundError, PermissionError):
            continue
        rel_path = normalize_rel_path(str(path.relative_to(root)))
        file_key = stable_int(f"file:{source.kb_id}:{rel_path}")
        if min_age_seconds > 0 and now - stat.st_mtime < min_age_seconds:
            # Still settling: skip parsing this round, but report it as present.
            if too_recent_keys is not None:
                too_recent_keys.add(file_key)
            continue

        size = stat.st_size
        mtime = int(stat.st_mtime)

        checksum = None
        if hash_content:
            recorded = (known_files or {}).get(rel_path)
            if recorded and recorded[0] == size and recorded[1] == mtime and recorded[2]:
                checksum = recorded[2]
                reused += 1
            else:
                try:
                    checksum = sha256_file(path)
                except (FileNotFoundError, PermissionError):
                    continue
                hashed += 1
        content_version = checksum or f"mtime:{int(stat.st_mtime_ns)}:size:{stat.st_size}"
        full_source_path = normalize_rel_path(f"{source.source_root}/{rel_path}")
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        parent = normalize_rel_path(str(Path(rel_path).parent))
        if parent == ".":
            parent = ""

        result.append(
            SourceFile(
                kb_id=source.kb_id,
                collection=source.collection,
                source_root=source.source_root,
                source_type=source.source_type,
                file_key=file_key,
                source_path=full_source_path,
                rel_path=rel_path,
                filename=path.name,
                dir=parent,
                physical_path=str(path),
                mime_type=mime_type,
                size=size,
                mtime=mtime,
                checksum=checksum,
            )
        )
    if hash_content and known_files is not None:
        print(f"[scan] source={source.kb_id} hashed={hashed} reused={reused}", flush=True)
    if outside:
        print(f"[scan] source={source.kb_id} skipped_outside_links={outside} (symlinks pointing outside the enrolled directory)", flush=True)
    return result


def list_recent_source_files(
    source: KBSource,
    *,
    min_age_seconds: int,
    limit: int | None = None,
) -> list[Path]:
    if min_age_seconds <= 0 or source.physical_base is None:
        return []
    root = source.physical_base.expanduser()
    if not root.exists():
        return []

    now = time.time()
    boundary = root.resolve()
    result: list[Path] = []
    for path in sorted(root.rglob("*")):
        if limit is not None and len(result) >= limit:
            break
        if not path.is_file() or should_skip(path):
            continue
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTS and path.name not in SUPPORTED_FILENAMES:
            continue
        if not inside_boundary(path, boundary):
            continue
        try:
            age = now - path.stat().st_mtime
        except FileNotFoundError:
            continue
        if age < min_age_seconds:
            result.append(path)
    return result


def inside_boundary(path: Path, boundary: Path) -> bool:
    """True when the file's real location is inside the (resolved) enrolled directory.
    A symlink that points elsewhere -- or a file under a symlinked sub-directory that
    does -- would otherwise let a synced tree pull arbitrary readable files on this
    host into the index and send them to the configured models."""
    try:
        real = path.resolve(strict=True)
    except OSError:
        return False
    return real == boundary or boundary in real.parents


def should_skip(path: Path) -> bool:
    parts = path.parts
    name = path.name
    if name.startswith(".") or name.startswith("._") or name.startswith("~$"):
        return True
    if any(part.startswith(".") for part in parts[:-1]):
        return True
    return bool(set(parts) & {".TemporaryItems", ".Trashes", ".fseventsd", ".Spotlight-V100", "__MACOSX"})


def stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:15], 16)
