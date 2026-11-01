"""Static, project-scoped import resolution without executing source files."""
from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable


@dataclass(frozen=True)
class ImportResolution:
    status: str
    candidates: tuple[str, ...] = ()

    @property
    def target(self) -> str | None:
        return self.candidates[0] if self.status == "internal" else None


class ImportResolver:
    # Project/package metadata, including self-contained agent instruction packages.
    MARKERS = ("pyproject.toml", "setup.py", "setup.cfg", "package.json", "Cargo.toml", "go.mod",
               "pom.xml", "build.gradle", "composer.json", "Gemfile", "SKILL.md", ".git")
    EXTENSIONS = ("", ".py", "/__init__.py", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".d.ts",
                  "/index.js", "/index.ts", "/index.tsx", ".go", ".java", ".rs", "/mod.rs", ".c", ".h",
                  ".cc", ".cpp", ".hpp", ".cs", ".php", ".rb", ".swift", ".kt", ".scala", ".sh", ".lua")

    def __init__(self, files: set[str], file_for: Callable):
        self.files = files
        self.file_for = file_for
        self._roots: dict[str, str] = {}

    def root(self, rel: str) -> str:
        if rel in self._roots:
            return self._roots[rel]
        parents = PurePosixPath(rel).parents
        root = rel.split("/", 1)[0] if "/" in rel else ""
        for parent in parents:
            directory = "" if str(parent) == "." else str(parent)
            for marker in self.MARKERS:
                candidate = f"{directory}/{marker}" if directory else marker
                path = self.file_for(candidate)
                try:
                    exists = candidate in self.files or bool(path and path.exists())
                except OSError:
                    exists = False
                if exists:
                    self._roots[rel] = directory
                    return directory
        self._roots[rel] = root
        return root

    def resolve(self, importer: str, module: str, level: int = 0, *, name: str | None = None) -> ImportResolution:
        module = str(module or "").strip().strip("\"'")
        root, directory = self.root(importer), posixpath.dirname(importer)
        python = importer.lower().endswith(".py")
        relative = bool(level or module.startswith(("./", "../")))
        if not module and not level:
            return ImportResolution("unresolved")
        if module.startswith(("/", "<")):
            return ImportResolution("unresolved")
        path = module.replace(".", "/") if python else re.sub(r"::|\\", "/", module)
        if not python and "/" not in path and not path.endswith(tuple(e for e in self.EXTENSIONS if e.startswith("."))):
            path = path.replace(".", "/")
        if level:
            for _ in range(level - 1):
                directory = posixpath.dirname(directory)
            roots = [directory]
        elif relative:
            roots = [directory]
        else:
            roots = list(dict.fromkeys([directory, root, posixpath.join(root, "src")]))
        if name:
            path = posixpath.join(path, name)
        stems = [posixpath.normpath(posixpath.join(base, path)) for base in roots]
        extensions = (".py", "/__init__.py") if python else self.EXTENSIONS
        hits = set()
        for stem in stems:
            for ext in extensions:
                candidate = stem.rstrip("/") + ext
                if candidate in self.files and self.root(candidate) == root:
                    hits.add(candidate)
        # Non-Python qualified namespaces may have language-specific source roots.
        # Match the full declared namespace, never a successively shortened basename.
        if not python and not relative and not hits and "/" in path:
            for candidate in self.files:
                if self.root(candidate) == root and any(candidate.endswith("/" + path + ext) for ext in extensions):
                    hits.add(candidate)
        if len(hits) == 1:
            return ImportResolution("internal", tuple(hits))
        if hits:
            return ImportResolution("ambiguous", tuple(sorted(hits)))
        return ImportResolution("unresolved" if relative else "external")
