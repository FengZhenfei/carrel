"""Rule-based extraction: code, structured markdown and configuration files bypass the model and are routed by
file type to deterministic graph builders.

They produce records of the same shape as model extraction (entities / relations / facts), so the merging,
scoping, vectors, Neo4j and recall downstream are all shared.
- code (.py): module / class / function / method / constant, with defines / has_method / calls / imports /
  inherits / depends_on edges. Symbol identity = the qualified name within the file scope (class / function /
  method are part, scoped per document); cross-file calls and imports point at the target file via scope_doc.
- structured_md (SKILL.md / AGENTS.md / CLAUDE.md / README.md, or any md with frontmatter): a document entity
  (skill / instructions / readme), frontmatter fields go into the fact layer, files referenced in the body get
  references edges.
- config (json / yaml / toml / requirements.txt): key paths go into the fact layer; requirements packages become
  package entities with depends_on edges.
Everything else is llm (the existing model extraction).
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .extract import ExtractionResult
from .facts import fact_id, normalize_fact
from .units import Unit

VERSION = "det-v5"

TYPES = ("module", "class", "function", "method", "constant", "package", "repository", "skill", "instructions", "readme",
         "config_file", "document")
UPPER = {
    "module": "entity", "class": "part", "function": "part", "method": "part", "constant": "property",
    "package": "standard", "repository": "entity", "skill": "entity", "instructions": "document", "readme": "document",
    "config_file": "document", "document": "document",
}
PREDICATES = [
    {"name": "defines", "description": "a module defines a class, function or constant", "source_parents": ["entity"], "target_parents": ["part", "property"]},
    {"name": "has_method", "description": "a class has a method", "source_parents": ["part"], "target_parents": ["part"]},
    {"name": "calls", "description": "a function, method or module-level code calls a function, method or class", "source_parents": ["part", "entity"], "target_parents": ["part"]},
    {"name": "imports", "description": "a module imports another module or package", "source_parents": ["entity"], "target_parents": ["entity", "standard"]},
    {"name": "inherits", "description": "a class inherits from a base class", "source_parents": ["part"], "target_parents": ["part"]},
    {"name": "depends_on", "description": "a repository depends on a package", "source_parents": ["entity"], "target_parents": ["standard"]},
    {"name": "references", "description": "a document references a file, module or document", "source_parents": ["entity", "document"], "target_parents": ["entity", "document", "part"]},
]
ALLOWED_ENDS = {p["name"]: (set(p["source_parents"]), set(p["target_parents"])) for p in PREDICATES}

CONFIG_SUFFIXES = {".json", ".yaml", ".yml", ".toml", ".json5", ".ini", ".cfg", ".conf"}
# Code with a syntax tree goes to code; other source-like text (styles, templates, SQL, fish...) goes to code_plain:
# only a file entity is built, no calls are spent
CODE_PLAIN_SUFFIXES = {".css", ".scss", ".vue", ".svelte", ".sql", ".fish", ".xml", ".html", ".htm"}
_CODE_EXTS = {".py"}
try:
    from ..parsers.code_symbols import LANGUAGE_BY_SUFFIX as _TS_LANGS

    _CODE_EXTS |= set(_TS_LANGS)
except Exception:      # pragma: no cover
    _TS_LANGS = {}
_SOURCE_EXTS = {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".go", ".java", ".rs", ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".php",
                ".rb", ".swift", ".kt", ".kts", ".scala", ".sh", ".bash", ".zsh", ".lua", ".ps1", ".vue", ".svelte", ".css", ".scss", ".sql", ".fish"}
_NODE_BUILTINS = {"fs", "path", "os", "http", "https", "url", "util", "crypto", "events", "stream", "child_process", "assert", "buffer",
                  "net", "readline", "zlib", "process", "tty", "dns", "cluster", "worker_threads", "timers", "module", "querystring"}
_JVM_STD_PREFIXES = ("java.", "javax.", "jakarta.", "kotlin.", "kotlinx.", "scala.", "android.", "androidx.")
_RUST_STD = {"std", "core", "alloc", "crate", "self", "super"}
_SWIFT_STD = {"Foundation", "UIKit", "SwiftUI", "Combine", "AppKit", "CoreData", "Dispatch"}
_DOTNET_PREFIXES = ("System", "Microsoft.")
STRUCTURED_MD_NAMES = {"skill.md", "agents.md", "claude.md", "readme.md", "soul.md", "tools.md", "user.md"}
_REF_RE = re.compile(r"(?<![\w/])((?:[\w.-]+/)*[\w.-]+\.(?:py|md|json|yaml|yml|toml|txt|sh|js|ts))(?![\w/])")
_STDLIB = set(getattr(sys, "stdlib_module_names", ()))


def route_for(rel_path: str, *, first_line: str | None = None) -> str:
    """File -> graph builder: code / config / structured_md / llm."""
    name = rel_path.rsplit("/", 1)[-1]
    lower = name.lower()
    suffix = "." + lower.rsplit(".", 1)[-1] if "." in lower else ""
    if suffix in _CODE_EXTS:
        return "code"
    if suffix in CONFIG_SUFFIXES or lower == "requirements.txt":
        return "config"
    if suffix in CODE_PLAIN_SUFFIXES:
        return "code_plain"
    if suffix in (".md", ".markdown"):
        if lower in STRUCTURED_MD_NAMES or (first_line or "").strip() == "---":
            return "structured_md"
    return "llm"


def _entity(name: str, etype: str, description: str = "", *, aliases: list[str] | None = None, scope_doc: str | None = None,
            mentions: int = 1) -> dict[str, Any]:
    rec: dict[str, Any] = {"name": name, "type": etype, "descriptions": [description] if description else [], "mentions": mentions,
                           "types": {etype: 1}, "exact": True}
    if aliases:
        rec["aliases"] = [a for a in aliases if a and a != name]
    if scope_doc:
        rec["scope_doc"] = scope_doc
    return rec


def _relation(source: str, target: str, predicate: str, description: str = "", strength: float = 1.0) -> dict[str, Any]:
    return {"source": source, "target": target, "predicate": predicate, "predicate_raw": predicate,
            "descriptions": [description] if description else [], "strength": float(strength)}


def _dotted(rel_path: str, repo: str) -> str:
    inner = rel_path[len(repo) + 1:] if repo and rel_path.startswith(repo + "/") else rel_path
    stem = inner.rsplit(".", 1)[0] if "." in inner.rsplit("/", 1)[-1] else inner
    parts = stem.split("/")
    if parts and parts[-1] in ("__init__", "index", "mod", "main"):
        parts = parts[:-1] or parts
    return ".".join(p for p in parts if p)


def _repo_of(rel_path: str) -> str:
    return rel_path.split("/", 1)[0] if "/" in rel_path else ""


def module_aliases(rel_path: str) -> list[str]:
    """Aliases of a module entity: the dotted module name, the path without the repository prefix, the file name --
    all three spellings in a question hit exactly."""
    repo = _repo_of(rel_path)
    inner = rel_path[len(repo) + 1:] if repo and rel_path.startswith(repo + "/") else rel_path
    out = [_dotted(rel_path, repo), inner, rel_path.rsplit("/", 1)[-1]]
    return [a for i, a in enumerate(out) if a and a != rel_path and a not in out[:i]]


class DeterministicExtractor:
    """Produces rule-based extraction results per unit. units are all units of this KB (needed to map rel_path to
    doc_id and to know every file); file_for(rel_path) gives the original file's path in the mirror directory."""

    def __init__(self, units: list[Unit], *, file_for: Callable[[str], Path | None], kb_id: str = "") -> None:
        self.kb_id = kb_id
        self.file_for = file_for
        self.doc_of: dict[str, str] = {}
        self.rel_of_doc: dict[str, str] = {}
        for u in units:
            if u.rel_path:
                self.doc_of[u.rel_path] = u.doc_id
                self.rel_of_doc[u.doc_id] = u.rel_path
        self.known_files = set(self.doc_of)
        # each document's first unit (order is KB-wide, not within the document): document-level entities and
        # facts hang off this unit only
        self.first_order: dict[str, int] = {}
        for u in units:
            if u.doc_id not in self.first_order or u.order < self.first_order[u.doc_id]:
                self.first_order[u.doc_id] = u.order
        self._routes: dict[str, str] = {}
        self._code: dict[str, dict[str, Any]] = {}       # rel_path -> analysis result
        self._text_cache: dict[str, str] = {}
        self.route_counts: Counter = Counter()

    # ── Routing ──
    def route(self, unit: Unit) -> str:
        rel = unit.rel_path
        if rel not in self._routes:
            first = None
            if rel.lower().endswith((".md", ".markdown")):
                text = self._text(rel)
                first = text.splitlines()[0] if text else ""
            self._routes[rel] = route_for(rel, first_line=first)
        return self._routes[rel]

    def wants(self, unit: Unit) -> bool:
        return self.route(unit) != "llm"

    def is_first_unit(self, unit: Unit) -> bool:
        return self.first_order.get(unit.doc_id, unit.order) == unit.order

    def _text(self, rel: str) -> str:
        if rel not in self._text_cache:
            path = self.file_for(rel)
            try:
                self._text_cache[rel] = path.read_text(encoding="utf-8", errors="ignore") if path and path.exists() else ""
            except OSError:
                self._text_cache[rel] = ""
        return self._text_cache[rel]

    # ── Code ──
    def _analysis(self, rel: str) -> dict[str, Any]:
        """The symbol table of one .py file: {qualname: meta}, module metadata, import table. Parsed on demand,
        shared KB-wide."""
        if rel in self._code:
            return self._code[rel]
        from ..parsers.code_python import python_symbol_blocks
        from ..parsers.code_symbols import language_for, symbol_blocks

        out: dict[str, Any] = {"symbols": {}, "module": None, "ok": False, "language": "python" if rel.lower().endswith(".py") else (language_for(rel) or "")}
        text = self._text(rel)
        blocks = []
        if text.strip():
            if rel.lower().endswith(".py"):
                try:
                    blocks = python_symbol_blocks(text)
                    out["ok"] = True
                except SyntaxError:
                    blocks = []
            elif out["language"]:
                try:
                    blocks = symbol_blocks(text, out["language"]) or []
                    out["ok"] = bool(blocks)
                except Exception:
                    blocks = []
            for b in blocks:
                sym = b.metadata.get("symbol") or {}
                if sym.get("kind") == "module":
                    out["module"] = sym
                elif sym.get("qualname"):
                    out["symbols"][sym["qualname"]] = sym
                if b.metadata.get("module"):
                    out["module"] = b.metadata["module"]
        self._code[rel] = out
        return out

    _EXT_CANDIDATES = ("", ".py", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".go", ".java", ".rs", ".c", ".h", ".cc", ".cpp", ".hpp",
                       ".cs", ".php", ".rb", ".swift", ".kt", ".scala", ".sh", ".lua", "/index.js", "/index.ts", "/index.tsx", "/__init__.py",
                       "/mod.rs", ".d.ts")

    def _suffix_match(self, stem: str) -> str | None:
        """Find the unique KB file that ends with stem (a/b/c) plus one of the extensions; if none, drop the first
        segment and retry (package paths that carry a repository prefix)."""
        parts = [p for p in stem.split("/") if p and p not in (".", "..")]
        while parts:
            tail = "/".join(parts)
            hits: list[str] = []
            for ext in self._EXT_CANDIDATES:
                cand = tail + ext
                hits += [f for f in self.known_files if f == cand or f.endswith("/" + cand)]
            hits = list(dict.fromkeys(hits))
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                # several candidates: prefer the same repository and the shortest path
                return sorted(hits, key=lambda f: (len(f.split("/")), f))[0]
            if len(parts) <= 1:
                break
            parts = parts[1:]
        return None

    def _resolve_module(self, importer_rel: str, module: str, level: int) -> str | None:
        """Module name in an import statement -> the file's rel_path in the KB (not found means an external package).
        Path-style names (./x, ../y, lib/util.h, lib/util.sh) resolve as relative paths; dotted / double-colon /
        backslash / go package paths match by suffix."""
        repo = _repo_of(importer_rel)
        module = str(module or "").strip().strip("\"'")
        if not module or module.startswith("<"):
            return None
        base_dir = importer_rel.rsplit("/", 1)[0] if "/" in importer_rel else ""
        if module.startswith("./") or module.startswith("../") or module.endswith((".js", ".ts", ".h", ".hpp", ".sh", ".lua", ".php", ".rb")) or "/" in module:
            rel_base = base_dir
            m = module
            while m.startswith("./") or m.startswith("../"):
                if m.startswith("../"):
                    rel_base = rel_base.rsplit("/", 1)[0] if "/" in rel_base else ""
                    m = m[3:]
                else:
                    m = m[2:]
            for cand_stem in ([f"{rel_base}/{m}" if rel_base else m] if (module.startswith(".") or level) else [f"{repo}/{m}" if repo else m, f"{rel_base}/{m}" if rel_base else m, m]):
                for ext in self._EXT_CANDIDATES:
                    cand = cand_stem.rstrip("/") + ext
                    if cand in self.known_files:
                        return cand
            if "." in module.rsplit("/", 1)[-1] or module.startswith("."):
                return None if module.startswith(".") else self._suffix_match(module)
            return self._suffix_match(module)
        if "::" in module or "\\" in module:
            parts = [p for p in re.split(r"::|\\\\", module) if p and p not in _RUST_STD]
            return self._suffix_match("/".join(parts)) if parts else None
        if level:
            base = importer_rel.rsplit("/", 1)[0] if "/" in importer_rel else ""
            for _ in range(level - 1):
                base = base.rsplit("/", 1)[0] if "/" in base else ""
            parts = [base] if base else []
            if module:
                parts.append(module.replace(".", "/"))
            stem = "/".join(p for p in parts if p)
        else:
            stem = ((repo + "/") if repo else "") + module.replace(".", "/")
        for cand in (stem + ".py", stem + "/__init__.py"):
            if cand in self.known_files:
                return cand
        # absolute imports without a repository prefix (files at the KB root), and dotted paths of java / kotlin /
        # scala / lua
        if repo:
            for cand in (module.replace(".", "/") + ".py", module.replace(".", "/") + "/__init__.py"):
                if cand in self.known_files:
                    return cand
        return self._suffix_match(module.replace(".", "/")) if module and not level else None

    def _import_map(self, rel: str) -> dict[str, tuple[str | None, str | None, str | None]]:
        """Names in this file -> (in-KB module rel_path | None, symbol name | None, external package name | None)."""
        out: dict[str, tuple[str | None, str | None, str | None]] = {}
        mod = self._analysis(rel).get("module") or {}
        for imp in mod.get("imports") or []:
            module, name, asname, level = imp.get("module") or "", imp.get("name"), imp.get("asname"), int(imp.get("level") or 0)
            if name:
                local = asname or name
            else:
                seg = module.strip("<>\"'")
                seg = seg.rsplit("/", 1)[-1] if "/" in seg else re.split(r"::|\.", seg)[-1] if seg else seg
                local = asname or seg.rsplit(".", 1)[0] if "." in seg and "/" in module else (asname or seg)
                if not local:
                    local = module.split(".")[0]
            target_rel = self._resolve_module(rel, module, level)
            if name and (target_rel is None or name not in self._analysis(target_rel)["symbols"]):
                # from a.b import c: when c is not a symbol defined in module a.b, try the submodule a/b/c.py first,
                # for relative imports too (`from .. import db`, `from .graph import build` are the usual in-package
                # forms; both used to lose their calls / imports edges).
                sub = self._resolve_module(rel, f"{module}.{name}" if module else name, level)
                if sub and sub != target_rel:
                    out[local] = (sub, None, None)
                    continue
            if target_rel is not None:
                out[local] = (target_rel, name, None)
            else:
                pkg = self._external_package(rel, module or name or "", level)
                if pkg:
                    out[local] = (None, name, pkg)
        return out

    def _external_package(self, rel: str, module: str, level: int) -> str | None:
        """An import not found in the KB -> external package name; each language's standard library does not
        count."""
        module = str(module or "").strip().strip("\"'")
        if not module or level or module.startswith((".", "/", "<")):
            return None
        lang = self._analysis(rel).get("language") or ""
        if lang == "python":
            top = module.split(".")[0]
            return top if top and top not in _STDLIB else None
        if lang in ("javascript", "typescript", "tsx"):
            top = module.split("/")[0] if not module.startswith("@") else "/".join(module.split("/")[:2])
            return None if top.replace("node:", "") in _NODE_BUILTINS or module.startswith("node:") else top
        if lang == "go":
            first = module.split("/")[0]
            return module if "." in first else None            # a standard-library path's first segment has no dot (fmt, net/http)
        if lang in ("java", "kotlin", "scala"):
            return None if module.startswith(_JVM_STD_PREFIXES) else ".".join(module.split(".")[:3])
        if lang == "rust":
            top = module.split("::")[0]
            return top if top and top not in _RUST_STD else None
        if lang in ("c", "cpp"):
            return None                                          # a header that cannot be found is let go, not treated as a package
        if lang == "csharp":
            return None if module.startswith(_DOTNET_PREFIXES) else module.split(".")[0]
        if lang == "php":
            return module.split(".")[0] if module else None
        if lang == "ruby":
            return module.split("/")[0]
        if lang == "swift":
            return None if module in _SWIFT_STD else module
        if lang == "lua":
            return module.split(".")[0]
        return None

    def _code_records(self, unit: Unit) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rel = unit.rel_path
        analysis = self._analysis(rel)
        symbols: dict[str, dict[str, Any]] = analysis["symbols"]
        module_meta = analysis.get("module") or {}
        repo = _repo_of(rel)
        dotted = _dotted(rel, repo)
        module_title = rel
        ents: list[dict[str, Any]] = []
        rels: list[dict[str, Any]] = []
        names_in_unit: list[str] = []
        has_module_block = False
        for bid in unit.block_ids:
            if not (bid.startswith("py-") or bid.startswith("sym-")):
                continue
            if bid in ("py-0000-module", "sym-0000-module"):
                has_module_block = True
                continue
            qual = bid.split("-", 2)[2] if bid.count("-") >= 2 else ""
            if qual in symbols:
                names_in_unit.append(qual)
            else:
                safe_lookup = {re.sub(r"[^A-Za-z0-9_.]", "_", q)[:80]: q for q in symbols}
                if qual in safe_lookup:
                    names_in_unit.append(safe_lookup[qual])
        if not names_in_unit and not has_module_block:
            return ents, rels
        imports = self._import_map(rel)

        def module_entity(target_rel: str) -> dict[str, Any]:
            return _entity(target_rel, "module", aliases=module_aliases(target_rel), mentions=1)

        def symbol_entity(qual: str, meta: dict[str, Any], scope_doc: str | None = None) -> dict[str, Any]:
            kind = meta.get("kind") or "function"
            desc = meta.get("signature") or ""
            if meta.get("docstring"):
                desc = f"{desc}: {meta['docstring']}" if desc else meta["docstring"]
            if kind == "class" and meta.get("methods"):
                desc += f" (methods: {', '.join(meta['methods'][:12])})"
            bare = qual.rsplit(".", 1)[-1]
            return _entity(qual, kind, desc, aliases=[bare] if bare != qual else None, scope_doc=scope_doc)

        # the module itself: the unit holding the module block is responsible for imports, constants and defines edges
        ents.append(_entity(module_title, "module", (module_meta.get("docstring") or f"Python module {dotted}"), aliases=module_aliases(rel)))
        if has_module_block:
            for local, (target_rel, name, pkg) in imports.items():
                if target_rel:
                    ents.append(module_entity(target_rel))
                    rels.append(_relation(module_title, target_rel, "imports", f"{rel} imports {name or ''} from {target_rel}".strip()))
                elif pkg:
                    ents.append(_entity(pkg, "package", f"external package {pkg}"))
                    rels.append(_relation(module_title, pkg, "imports", f"{rel} imports {pkg}"))
            for const in module_meta.get("constants") or []:
                ents.append(_entity(const, "constant", f"module-level constant in {rel}"))
                rels.append(_relation(module_title, const, "defines", f"{rel} defines {const}"))
            for qual, meta in symbols.items():
                if "." in qual:
                    continue
                ents.append(symbol_entity(qual, meta))
                rels.append(_relation(module_title, qual, "defines", f"{rel} defines {meta.get('kind')} {qual}"))
            for callee in module_meta.get("calls") or []:
                self._call_edge(rel, module_title, callee, symbols, imports, None, ents, rels)
        # symbols in the unit: method ownership, inheritance, calls
        for qual in names_in_unit:
            meta = symbols[qual]
            ents.append(symbol_entity(qual, meta))
            if meta.get("kind") == "method":
                cls = meta.get("class") or qual.split(".")[0]
                if cls in symbols:
                    ents.append(symbol_entity(cls, symbols[cls]))
                    rels.append(_relation(cls, qual, "has_method", f"{cls} has method {qual.rsplit('.', 1)[-1]}"))
            if meta.get("kind") == "class":
                for base in meta.get("bases") or []:
                    base_name = base.split("[")[0]
                    self._target_edge(rel, qual, base_name, "inherits", symbols, imports, ents, rels)
            for callee in meta.get("calls") or []:
                self._call_edge(rel, qual, callee, symbols, imports, meta.get("class"), ents, rels)
        return ents, rels

    def _target_edge(self, rel: str, source: str, name: str, predicate: str, symbols: dict[str, Any],
                     imports: dict[str, tuple[str | None, str | None, str | None]], ents: list, rels: list) -> bool:
        """Name -> target entity (a symbol of this file / an imported in-KB symbol / an external package), with one
        edge."""
        if name in symbols:
            ents.append(self._sym(symbols[name], name))
            rels.append(_relation(source, name, predicate, f"{source} {predicate} {name}"))
            return True
        if name in imports:
            target_rel, sym_name, pkg = imports[name]
            if target_rel and sym_name:
                tsyms = self._analysis(target_rel)["symbols"]
                if sym_name in tsyms:
                    ents.append(self._sym(tsyms[sym_name], sym_name, scope_doc=self.doc_of.get(target_rel)))
                    rels.append(_relation(source, sym_name, predicate, f"{source} {predicate} {sym_name} ({target_rel})"))
                    return True
            elif pkg:
                ents.append(_entity(pkg, "package", f"external package {pkg}"))
                rels.append(_relation(source, pkg, predicate if predicate != "calls" else "calls", f"{source} {predicate} {name} from {pkg}"))
                return True
        # whole-file imports (C #include, bash source, go packages): look for a symbol of that name in the imported file
        for target_rel, sym_name, _pkg in imports.values():
            if target_rel and not sym_name and target_rel != rel:
                tsyms = self._analysis(target_rel)["symbols"]
                if name in tsyms:
                    ents.append(self._sym(tsyms[name], name, scope_doc=self.doc_of.get(target_rel)))
                    rels.append(_relation(source, name, predicate, f"{source} {predicate} {name} ({target_rel})"))
                    return True
        return False

    def _sym(self, meta: dict[str, Any], qual: str, scope_doc: str | None = None) -> dict[str, Any]:
        kind = meta.get("kind") or "function"
        desc = meta.get("signature") or ""
        if meta.get("docstring"):
            desc = f"{desc}: {meta['docstring']}" if desc else meta["docstring"]
        bare = qual.rsplit(".", 1)[-1]
        return _entity(qual, kind, desc, aliases=[bare] if bare != qual else None, scope_doc=scope_doc)

    def _call_edge(self, rel: str, source: str, callee: str, symbols: dict[str, Any],
                   imports: dict[str, tuple[str | None, str | None, str | None]], cls: str | None, ents: list, rels: list) -> None:
        if callee.startswith("self.") and cls:
            method = f"{cls}.{callee[5:]}"
            if method in symbols:
                ents.append(self._sym(symbols[method], method))
                rels.append(_relation(source, method, "calls", f"{source} calls {method}"))
                return
            # not in this class: look one level up the base classes (the base may live in another file)
            for base in (symbols.get(cls) or {}).get("bases") or []:
                base = base.split("[")[0]
                base_rel, base_syms = rel, symbols
                if base not in symbols and base in imports and imports[base][0]:
                    base_rel = imports[base][0]
                    base_syms = self._analysis(base_rel)["symbols"]
                target = f"{base}.{callee[5:]}"
                if target in base_syms:
                    scope = self.doc_of.get(base_rel) if base_rel != rel else None
                    ents.append(self._sym(base_syms[target], target, scope_doc=scope))
                    rels.append(_relation(source, target, "calls", f"{source} calls inherited {target}"))
                    return
            return
        if "." in callee:
            head, tail = callee.split(".", 1)
            if "." in tail:
                return                                            # multi-level paths (a.b.c()) cannot be resolved statically
            if head in imports and imports[head][0]:
                target_rel, sym_name, _pkg = imports[head]
                tsyms = self._analysis(target_rel)["symbols"]
                # module alias.function (db.claim) -> that module's symbol; imported class.method (Store.open) -> that method
                target = f"{sym_name}.{tail}" if sym_name else tail
                if target in tsyms:
                    ents.append(self._sym(tsyms[target], target, scope_doc=self.doc_of.get(target_rel)))
                    rels.append(_relation(source, target, "calls", f"{source} calls {target} ({target_rel})"))
            elif f"{head}.{tail}" in symbols:
                target = f"{head}.{tail}"                          # class.method defined in this file
                ents.append(self._sym(symbols[target], target))
                rels.append(_relation(source, target, "calls", f"{source} calls {target}"))
            return
        self._target_edge(rel, source, callee, "calls", symbols, imports, ents, rels)

    # ── Structured markdown ──
    def _md_records(self, unit: Unit) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        from ..parsers.common import parse_frontmatter

        rel = unit.rel_path
        text = self._text(rel)
        fields, body = parse_frontmatter(text)
        name_l = rel.rsplit("/", 1)[-1].lower()
        etype = "skill" if name_l == "skill.md" else ("readme" if name_l.startswith("readme") else ("instructions" if name_l in STRUCTURED_MD_NAMES else "document"))
        title = str(fields.get("name") or fields.get("title") or "").strip()
        if not title:
            m = re.search(r"^#\s+(.+)$", body, re.M)
            title = (m.group(1).strip() if m else rel.rsplit("/", 1)[-1])
        description = str(fields.get("description") or "").strip()
        if not description:
            para = next((p.strip() for p in re.split(r"\n\s*\n", body) if p.strip() and not p.strip().startswith("#")), "")
            description = para[:300]
        aliases = [str(fields.get("display_name") or "").strip(), rel.rsplit("/", 1)[-1]]
        ents: list[dict[str, Any]] = []
        rels: list[dict[str, Any]] = []
        facts: list[dict[str, Any]] = []
        is_first = "md-fm" in unit.block_ids or self.is_first_unit(unit)
        if is_first:
            ents.append(_entity(title, etype, description, aliases=aliases))
            for k, v in fields.items():
                if k in ("name", "description", "title") or not str(v).strip():
                    continue
                fact = normalize_fact({"subject": title, "property": k, "value": str(v)[:120], "note": rel})
                if fact:
                    fact["id"] = fact_id(unit.unit_id, fact)
                    fact["unit_id"] = unit.unit_id
                    facts.append(fact)
        # files referenced in the body (only those that really exist in the KB)
        seen: set[str] = set()
        for ref in _REF_RE.findall(unit.text):
            target = self._match_file(ref, rel)
            if not target or target == rel or target in seen:
                continue
            seen.add(target)
            if not is_first:
                ents.append(_entity(title, etype, description, aliases=aliases))
                is_first = True
            troute = route_for(target)
            ttype = "module" if troute == "code" else ("config_file" if troute == "config" else "document")
            t_alias = module_aliases(target) if ttype == "module" else [target.rsplit("/", 1)[-1]]
            ents.append(_entity(target, ttype, f"{ttype} {target}", aliases=t_alias))
            rels.append(_relation(title, target, "references", f"{rel} references {target}"))
        return ents, rels, facts

    def _match_file(self, ref: str, from_rel: str) -> str | None:
        ref = ref.strip("./")
        if ref in self.known_files:
            return ref
        base = from_rel.rsplit("/", 1)[0] if "/" in from_rel else ""
        cand = f"{base}/{ref}" if base else ref
        if cand in self.known_files:
            return cand
        matches = [f for f in self.known_files if f.endswith("/" + ref)]
        return matches[0] if len(matches) == 1 else None

    # ── Configuration ──
    def _config_records(self, unit: Unit) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        rel = unit.rel_path
        text = self._text(rel)
        lower = rel.lower()
        ents: list[dict[str, Any]] = []
        rels: list[dict[str, Any]] = []
        facts: list[dict[str, Any]] = []
        if not self.is_first_unit(unit):
            return ents, rels, facts       # a config file's facts hang off the first unit only, no repeats
        if lower.endswith("requirements.txt"):
            repo = _repo_of(rel) or "repository"
            ents.append(_entity(repo, "repository", f"repository {repo}"))
            for line in text.splitlines():
                line = line.split("#", 1)[0].strip()
                if not line or line.startswith("-"):
                    continue
                m = re.match(r"^([A-Za-z0-9_.\-\[\]]+)\s*(.*)$", line)
                if not m:
                    continue
                pkg, spec = m.group(1).split("[")[0], m.group(2).strip()
                ents.append(_entity(pkg, "package", f"Python dependency {pkg} {spec}".strip()))
                rels.append(_relation(repo, pkg, "depends_on", f"{rel}: {line}"))
                fact = normalize_fact({"subject": repo, "property": "dependency", "symbol": pkg, "value": spec or "any", "note": rel})
                if fact:
                    fact["id"] = fact_id(unit.unit_id, fact); fact["unit_id"] = unit.unit_id
                    facts.append(fact)
            return ents, rels, facts
        data = self._load_config(rel, text)
        ents.append(_entity(rel, "config_file", f"configuration file {rel}"))
        if data is None:
            return ents, rels, facts
        for key_path, value in self._flatten(data)[:200]:
            fact = normalize_fact({"subject": rel, "property": key_path, "value": str(value)[:120], "note": ""})
            if fact:
                fact["id"] = fact_id(unit.unit_id, fact); fact["unit_id"] = unit.unit_id
                facts.append(fact)
        return ents, rels, facts

    @staticmethod
    def _load_config(rel: str, text: str) -> Any:
        lower = rel.lower()
        try:
            if lower.endswith((".json", ".json5")):
                return json.loads(text)
            if lower.endswith(".toml"):
                import tomllib

                return tomllib.loads(text)
            if lower.endswith((".yaml", ".yml")):
                try:
                    import yaml  # type: ignore
                except ImportError:
                    return None
                return yaml.safe_load(text)
        except Exception:
            return None
        return None

    @staticmethod
    def _flatten(data: Any, prefix: str = "") -> list[tuple[str, Any]]:
        out: list[tuple[str, Any]] = []
        if isinstance(data, dict):
            for k, v in data.items():
                out.extend(DeterministicExtractor._flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        elif isinstance(data, list):
            if all(not isinstance(x, (dict, list)) for x in data):
                out.append((prefix or "(root)", ", ".join(str(x) for x in data)[:200]))
            else:
                for i, v in enumerate(data):
                    out.extend(DeterministicExtractor._flatten(v, f"{prefix}[{i}]"))
        elif data is not None and prefix:
            out.append((prefix, data))
        return out

    # ── Public API ──
    def extract(self, unit: Unit) -> ExtractionResult:
        route = self.route(unit)
        self.route_counts[route] += 1
        if route == "code":
            ents, rels = self._code_records(unit)
        elif route == "code_plain":
            ents, rels = ([_entity(unit.rel_path, "module", f"source file {unit.rel_path}", aliases=module_aliases(unit.rel_path))] if self.is_first_unit(unit) else []), []
        elif route == "structured_md":
            ents, rels, _ = self._md_records(unit)
        elif route == "config":
            ents, rels, _ = self._config_records(unit)
        else:
            ents, rels = [], []
        return ExtractionResult(unit_id=unit.unit_id, entities=_dedupe(ents), relations=_dedupe_rel(rels), calls=0,
                                stats={"builder": route, "records": len(ents) + len(rels), "unit_kind": "body", "malformed": 0})

    def facts(self, unit: Unit) -> list[dict[str, Any]]:
        route = self.route(unit)
        if route == "structured_md":
            return self._md_records(unit)[2]
        if route == "config":
            return self._config_records(unit)[2]
        return []


def _dedupe(ents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for e in ents:
        key = (e["name"], e["type"], e.get("scope_doc") or "")
        cur = out.get(key)
        if cur is None:
            out[key] = dict(e)
        else:
            for d in e.get("descriptions") or []:
                if d not in cur["descriptions"]:
                    cur["descriptions"].append(d)
            for a in e.get("aliases") or []:
                cur.setdefault("aliases", [])
                if a not in cur["aliases"]:
                    cur["aliases"].append(a)
    return list(out.values())


def _dedupe_rel(rels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in rels:
        key = (r["source"], r["target"], r["predicate"])
        if key not in out:
            out[key] = dict(r)
        else:
            out[key]["strength"] = float(out[key]["strength"]) + float(r.get("strength") or 1.0)
    return list(out.values())
