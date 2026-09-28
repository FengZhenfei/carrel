"""Symbol-level chunking of Python source: module level (imports / constants / top-level statements),
functions, class heads and methods each become one block.

Every line belongs to exactly one block (lossless), and the block's metadata.symbol records the kind,
qualified name, line numbers, signature, called names, base classes and import table; rule-based
extraction in the graph build (graph.deterministic) can build definition / call / import /
inheritance edges from this metadata alone without parsing again. On a syntax error it falls back to
one block for the whole file.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from ..models import ParsedBlock

PROFILE = "py-symbols-v1"
_SAFE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.")


def _safe_id(text: str) -> str:
    return "".join(ch if ch in _SAFE else "_" for ch in text)[:80]


def _docstring(node: ast.AST) -> str:
    try:
        doc = ast.get_docstring(node) or ""
    except TypeError:
        doc = ""
    return doc.strip().split("\n\n", 1)[0].strip()[:300]


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = []
    a = node.args
    positional = [*a.posonlyargs, *a.args]
    defaults = [None] * (len(positional) - len(a.defaults)) + list(a.defaults)
    for arg, default in zip(positional, defaults):
        text = arg.arg
        if arg.annotation is not None:
            text += ": " + ast.unparse(arg.annotation)
        if default is not None:
            text += " = " + ast.unparse(default)
        args.append(text)
    if a.vararg:
        args.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        args.append("*")
    for arg, default in zip(a.kwonlyargs, a.kw_defaults):
        text = arg.arg + (" = " + ast.unparse(default) if default is not None else "")
        args.append(text)
    if a.kwarg:
        args.append("**" + a.kwarg.arg)
    ret = " -> " + ast.unparse(node.returns) if node.returns is not None else ""
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    return f"{prefix}{node.name}({', '.join(args)}){ret}"


def _call_names(node: ast.AST) -> list[str]:
    """Names called in a function body: foo() -> foo; self.bar() -> self.bar; pkg.func() -> pkg.func."""
    names: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Name):
                names.append(func.id)
            elif isinstance(func, ast.Attribute):
                parts = []
                cur: ast.AST = func
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                    names.append(".".join(reversed(parts)))
    seen: list[str] = []
    for n in names:
        if n not in seen:
            seen.append(n)
    return seen


def _imports(tree: ast.Module) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append({"module": alias.name, "name": None, "asname": alias.asname, "level": 0})
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                out.append({"module": node.module or "", "name": alias.name, "asname": alias.asname, "level": int(node.level or 0)})
    return out


def _start_line(node: ast.AST) -> int:
    decos = getattr(node, "decorator_list", None) or []
    return min([node.lineno, *[d.lineno for d in decos]])


def python_symbol_blocks(source: str, *, parser_profile: str = PROFILE, doc_type: str = "py") -> list[ParsedBlock]:
    lines = source.splitlines()
    tree = ast.parse(source)
    covered = [False] * (len(lines) + 1)
    symbols: list[tuple[int, int, dict[str, Any], list[str]]] = []   # (start, end, meta, section_path)

    def take(start: int, end: int) -> str:
        for i in range(start, end + 1):
            covered[i] = True
        return "\n".join(lines[start - 1:end])

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start, end = _start_line(node), int(node.end_lineno or node.lineno)
            symbols.append((start, end, {
                "kind": "function", "name": node.name, "qualname": node.name, "lineno": start, "end_lineno": end,
                "signature": _signature(node), "docstring": _docstring(node), "calls": _call_names(node),
                "decorators": [ast.unparse(d) for d in node.decorator_list],
            }, []))
        elif isinstance(node, ast.ClassDef):
            cstart, cend = _start_line(node), int(node.end_lineno or node.lineno)
            methods = [m for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
            bases = [ast.unparse(b) for b in node.bases]
            fields = [t.id for stmt in node.body if isinstance(stmt, (ast.Assign, ast.AnnAssign))
                      for t in (stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]) if isinstance(t, ast.Name)]
            head_end = (_start_line(methods[0]) - 1) if methods else cend
            symbols.append((cstart, head_end, {
                "kind": "class", "name": node.name, "qualname": node.name, "lineno": cstart, "end_lineno": cend,
                "signature": f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}",
                "docstring": _docstring(node), "bases": bases, "fields": fields[:40],
                "methods": [m.name for m in methods], "calls": [],
                "decorators": [ast.unparse(d) for d in node.decorator_list],
            }, []))
            for i, m in enumerate(methods):
                mstart = _start_line(m)
                # Class-level statements between methods go to the preceding method, and the last
                # method runs to the end of the class: every line belongs to exactly one place
                mend = (_start_line(methods[i + 1]) - 1) if i + 1 < len(methods) else cend
                symbols.append((mstart, mend, {
                    "kind": "method", "name": m.name, "qualname": f"{node.name}.{m.name}", "lineno": mstart, "end_lineno": int(m.end_lineno or m.lineno),
                    "signature": _signature(m), "docstring": _docstring(m), "calls": _call_names(m), "class": node.name,
                    "decorators": [ast.unparse(d) for d in m.decorator_list],
                }, [node.name]))

    blocks: list[ParsedBlock] = []
    for idx, (start, end, meta, section_path) in enumerate(sorted(symbols, key=lambda s: s[0]), start=1):
        text = take(start, end)
        if not text.strip():
            continue
        blocks.append(ParsedBlock(
            parser="python-ast", parser_profile=parser_profile, doc_type=doc_type, block_type="code",
            text=text, title=meta["qualname"], block_id=f"py-{idx:04d}-{_safe_id(meta['qualname'])}",
            metadata={"section_path": list(section_path), "symbol": meta},
        ))
    rest = [lines[i - 1] for i in range(1, len(lines) + 1) if not covered[i]]
    rest_text = "\n".join(rest).strip("\n")
    constants = [t.id for stmt in tree.body if isinstance(stmt, (ast.Assign, ast.AnnAssign))
                 for t in (stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target])
                 if isinstance(t, ast.Name) and t.id.isupper()]
    module_meta = {"kind": "module", "name": "", "qualname": "", "docstring": _docstring(tree), "imports": _imports(tree),
                   "constants": constants[:60], "calls": _call_names(ast.Module(body=[s for s in tree.body if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))], type_ignores=[]))}
    if rest_text.strip():
        blocks.insert(0, ParsedBlock(
            parser="python-ast", parser_profile=parser_profile, doc_type=doc_type, block_type="code",
            text=rest_text, title="(module)", block_id="py-0000-module",
            metadata={"section_path": [], "symbol": module_meta},
        ))
    elif blocks:
        blocks[0].metadata["module"] = module_meta
    return blocks


def parse_python(path: Path, doc_type: str = "py", parser_profile: str = PROFILE) -> list[ParsedBlock]:
    from .common import read_text_smart

    source = read_text_smart(path)
    if not source.strip():
        return []
    try:
        return python_symbol_blocks(source, parser_profile=parser_profile, doc_type=doc_type)
    except SyntaxError:
        return [ParsedBlock(parser="native", parser_profile=parser_profile, doc_type=doc_type, block_type="text",
                            text=source, block_id="text-0001", metadata={"symbol": {"kind": "module", "syntax_error": True}})]
