"""Symbol-level chunking of multi-language source code (tree-sitter): one block each for module level /
function / class head / method, with metadata.symbol in the same shape as code_python (kind / name /
qualname / lineno / end_lineno / signature / docstring / calls / bases / methods / class; imports /
constants / calls for the module block). Rule-based extraction in the graph build relies on this
metadata alone.

Each language is just a mapping table of syntax-tree node types (LANGUAGES); chunking, qualified
names, doc comments, callee names and lossless line assignment are all shared. When tree-sitter is not
installed or the syntax tree errors out, the caller falls back to one block for the whole file.
"""
from __future__ import annotations

import re
from collections import deque
from typing import Any, Callable

from ..models import ParsedBlock

PROFILE = "code-symbols-v2"
MAX_CALLS = 40          # how many callee names are recorded per symbol

# Extension -> tree-sitter language name
LANGUAGE_BY_SUFFIX = {
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "tsx",
    ".go": "go", ".java": "java", ".rs": "rust",
    ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".cs": "csharp", ".php": "php", ".rb": "ruby", ".swift": "swift",
    ".kt": "kotlin", ".kts": "kotlin", ".scala": "scala",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash", ".lua": "lua", ".ps1": "powershell",
}

# Node types per language: function / method / class (including struct, interface, trait, enum,
# protocol ...) / namespace container / import / call
_COMMON_FUNCTION = {"function_declaration", "function_definition", "function_item", "method_declaration",
                    "method_definition", "constructor_declaration", "init_declaration", "function_statement",
                    "class_method_definition", "method", "singleton_method", "function_signature_item",
                    "protocol_function_declaration"}
_COMMON_CLASS = {"class_declaration", "abstract_class_declaration", "class_specifier", "struct_specifier",
                 "struct_item", "enum_item", "trait_item", "interface_declaration", "enum_declaration",
                 "record_declaration", "struct_declaration", "class_definition", "object_definition",
                 "trait_definition", "class", "protocol_declaration", "class_statement", "union_specifier",
                 "type_alias_declaration"}
_NAMESPACE = {"namespace_definition", "namespace_declaration", "module", "file_scoped_namespace_declaration"}
_WRAPPERS = {"export_statement", "template_declaration", "decorated_definition", "declaration"}
_TRANSPARENT = {"statement_list"}          # powershell: program -> statement_list -> statements
_COMMENTS = {"comment", "line_comment", "block_comment", "multiline_comment", "doc_comment"}
# Decorators that sit in the syntax tree as siblings in front of the definition: @Get() on TypeScript class
# members, Rust's #[test]. Other languages keep them inside the definition node
_DECORATORS = {"decorator", "attribute_item"}
_CALL_TYPES = {"call_expression", "method_invocation", "invocation_expression", "function_call_expression",
               "member_call_expression", "scoped_call_expression", "new_expression", "object_creation_expression",
               "call", "command", "nullsafe_member_call_expression"}
_IMPORT_TYPES = {"import_statement", "import_declaration", "use_declaration", "preproc_include", "using_directive",
                 "namespace_use_declaration", "import_header", "import_list"}
_NAME_FIELDS = ("name", "declarator")
_NAME_TYPES = {"identifier", "type_identifier", "field_identifier", "simple_identifier", "word", "name",
               "constant", "function_name", "namespace_identifier", "property_identifier", "qualified_identifier",
               "scoped_identifier", "dot_index_expression", "method_index_expression", "class_name", "variable_name", "simple_name"}
_ACCESS_WORDS = {"public", "private", "protected", "virtual", "override", "final", "sealed", "abstract", "internal", "open"}
_STRING_TYPES = {"string", "interpreted_string_literal", "string_literal", "encapsed_string", "system_lib_string",
                 "raw_string_literal", "string_fragment", "string_content", "simple_string"}


def language_for(path_name: str) -> str | None:
    lower = path_name.lower()
    suffix = "." + lower.rsplit(".", 1)[-1] if "." in lower else ""
    return LANGUAGE_BY_SUFFIX.get(suffix)


def _get_parser(language: str):
    try:
        import tree_sitter_language_pack as pack  # type: ignore
    except ImportError:
        return None
    try:
        return pack.get_parser(language)
    except Exception:
        return None


def _text(node) -> str:
    return node.text.decode("utf-8", errors="ignore") if node is not None else ""


def _first_child_of_types(node, types: set[str]):
    for ch in node.named_children:
        if ch.type in types:
            return ch
    return None


def _name_node(node):
    for field in _NAME_FIELDS:
        n = node.child_by_field_name(field)
        while n is not None and n.type in ("function_declarator", "pointer_declarator", "reference_declarator",
                                           "parenthesized_declarator", "init_declarator", "variable_declarator"):
            inner = n.child_by_field_name("declarator") or n.child_by_field_name("name")
            if inner is None:
                break
            n = inner
        if n is not None and (n.type in _NAME_TYPES or n.type.endswith("identifier")):
            return n
    # Grammars without a name field (kotlin class / function, powershell function): take the first
    # identifier child node
    for ch in node.named_children:
        if ch.type in _NAME_TYPES or ch.type.endswith("identifier"):
            return ch
    return None


def _qual(text: str) -> str:
    return text.replace("::", ".").replace("->", ".").replace("\\", ".").replace(":", ".").strip()


def _clean_comment(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        for marker in ("/**", "/*!", "/*", "*/", "///", "//!", "//", "#", "--", "*"):
            if s.startswith(marker):
                s = s[len(marker):].strip()
                break
        if s.endswith("*/"):
            s = s[:-2].strip()
        if s:
            lines.append(s)
    return " ".join(lines).split(". ")[0][:300]


class _Siblings(list):
    """The sibling nodes of one container, with a node -> position index. Looking up its own position linearly
    for every symbol takes quadratic time on a file with many top-level symbols."""

    def __init__(self, nodes) -> None:
        super().__init__(nodes)
        self.position = {n.id: i for i, n in enumerate(self)}


def _leading(node, siblings: _Siblings) -> int:
    """Where a definition starts among its siblings: decorators directly before it count as part of it (the
    same rule as code_python._start_line). Returns -1 when the node is not among these siblings."""
    idx = siblings.position.get(node.id, -1)
    while idx > 0 and siblings[idx - 1].type in _DECORATORS:
        idx -= 1
    return idx


def _start_row(node, siblings: _Siblings) -> int:
    idx = _leading(node, siblings)
    return siblings[idx].start_point[0] if idx >= 0 else node.start_point[0]


def _doc_before(node, siblings: _Siblings) -> str:
    """The comment directly before a definition (together with its decorators); may be several consecutive
    lines."""
    idx = _leading(node, siblings)
    if idx < 0:
        return ""
    parts: list[str] = []
    line = siblings[idx].start_point[0]
    for s in reversed(siblings[:idx]):
        if s.type in _COMMENTS and s.end_point[0] >= line - 1:
            parts.insert(0, _text(s))
            line = s.start_point[0]
        else:
            break
    return _clean_comment("\n".join(parts)) if parts else ""


def _signature(node) -> str:
    body = node.child_by_field_name("body")
    end = body.start_byte if body is not None else node.end_byte
    raw = node.text[: end - node.start_byte].decode("utf-8", errors="ignore") if body is not None else _text(node).split("{", 1)[0]
    return re.sub(r"\s+", " ", raw).strip().rstrip("{:=").strip()[:200]


def _callee(node) -> str | None:
    t = node.type
    if t == "scoped_call_expression":     # php Util::helper()
        scope, name = node.child_by_field_name("scope"), node.child_by_field_name("name")
        return _qual(f"{_text(scope)}.{_text(name)}") if name is not None else None
    if t in ("call_expression", "invocation_expression", "function_call_expression"):
        f = node.child_by_field_name("function") or node.child_by_field_name("name")
        return _qual(_text(f)) if f is not None else None
    if t in ("member_call_expression", "nullsafe_member_call_expression"):
        obj, name = node.child_by_field_name("object"), node.child_by_field_name("name")
        return _qual(f"{_text(obj)}.{_text(name)}") if name is not None else None
    if t == "method_invocation":
        obj, name = node.child_by_field_name("object"), node.child_by_field_name("name")
        return _qual((f"{_text(obj)}." if obj is not None else "") + _text(name)) if name is not None else None
    if t in ("new_expression", "object_creation_expression"):
        c = node.child_by_field_name("constructor") or node.child_by_field_name("type")
        return _qual(_text(c)) if c is not None else None
    if t == "call":      # ruby
        recv, m = node.child_by_field_name("receiver"), node.child_by_field_name("method")
        return _qual((f"{_text(recv)}." if recv is not None else "") + _text(m)) if m is not None else None
    if t == "command":   # bash
        n = node.child_by_field_name("name")
        return _text(n) if n is not None else None
    return None


def _calls_in(node) -> list[str]:
    """The names called inside the node, the first MAX_CALLS in breadth-first order. Stops once full: in a
    bundled file a single statement is the whole file."""
    out: list[str] = []
    seen: set[str] = set()
    queue = deque([node] if node.type in _CALL_TYPES else node.named_children)
    while queue and len(out) < MAX_CALLS:
        n = queue.popleft()
        if n.type in _CALL_TYPES:
            c = _callee(n)
            if c:
                c = re.sub(r"^\$?this(\.|->)", "self.", c)
                c = c.replace("self::", "self.").replace("static.", "self.")
                if c not in seen and len(c) <= 80:
                    seen.add(c)
                    out.append(c)
        queue.extend(n.named_children)
    return out


def _string_in(node) -> str:
    """Content of the first string literal inside the node (an import path)."""
    stack = deque([node])
    while stack:
        n = stack.popleft()
        if n.type in _STRING_TYPES:
            s = _text(n).strip().strip("\"'`<>")
            if n.type == "system_lib_string":
                return "<" + s + ">"
            if s:
                return s
        stack.extend(n.named_children)
    return ""


def _imports_of(node, language: str) -> list[dict[str, Any]]:
    t = node.type
    out: list[dict[str, Any]] = []
    if t in ("import_statement", "import_declaration") and language in ("javascript", "typescript", "tsx", "go", "swift", "scala"):
        if language == "go":
            for spec in [n for n in node.named_children if n.type == "import_spec"] + [n for c in node.named_children for n in c.named_children if n.type == "import_spec"]:
                path = _string_in(spec)
                if path:
                    out.append({"module": path, "name": None, "asname": None, "level": 0})
            return out
        if language in ("swift",):
            return [{"module": _qual(_text(node).replace("import", "", 1)), "name": None, "asname": None, "level": 0}]
        if language == "scala":
            parts = [_text(c) for c in node.named_children]
            return [{"module": ".".join(parts[:-1]) if len(parts) > 1 else ".".join(parts), "name": parts[-1] if len(parts) > 1 else None, "asname": None, "level": 0, "full": ".".join(parts)}]
        src = _string_in(node)
        if not src:
            return []
        level = 1 if src.startswith(".") else 0
        clause = _first_child_of_types(node, {"import_clause"})
        recs: list[dict[str, Any]] = []
        if clause is not None:
            for c in clause.named_children:
                if c.type == "identifier":                      # default import
                    recs.append({"module": src, "name": _text(c), "asname": None, "level": level})
                elif c.type == "namespace_import":              # * as x
                    ident = _first_child_of_types(c, {"identifier"})
                    recs.append({"module": src, "name": None, "asname": _text(ident) if ident is not None else None, "level": level})
                elif c.type == "named_imports":
                    for spec in c.named_children:
                        if spec.type == "import_specifier":
                            n, a = spec.child_by_field_name("name"), spec.child_by_field_name("alias")
                            recs.append({"module": src, "name": _text(n), "asname": _text(a) if a is not None else None, "level": level})
        return recs or [{"module": src, "name": None, "asname": None, "level": level}]
    if t == "import_declaration" and language == "java":
        full = _text(node).replace("import", "").replace("static", "").strip(" ;")
        pkg, _, last = full.rpartition(".")
        return [{"module": pkg or full, "name": None if last in ("*", "") or not pkg else last, "asname": None, "level": 0, "full": full}]
    if t == "use_declaration":      # rust
        arg = node.child_by_field_name("argument")
        text = _text(arg).replace(" ", "")
        if "{" in text:
            prefix, rest = text.split("{", 1)
            names = [x for x in rest.rstrip("}").split(",") if x]
            return [{"module": prefix.rstrip(":"), "name": n.split("::")[-1], "asname": None, "level": 0} for n in names]
        parts = text.split("::")
        return [{"module": "::".join(parts[:-1]) if len(parts) > 1 else parts[0], "name": parts[-1] if len(parts) > 1 else None, "asname": None, "level": 0}]
    if t == "preproc_include":
        path = _string_in(node)
        return [{"module": path, "name": None, "asname": None, "level": 0 if path.startswith("<") else 1}] if path else []
    if t == "using_directive":
        return [{"module": _text(node).replace("using", "").strip(" ;"), "name": None, "asname": None, "level": 0}]
    if t == "namespace_use_declaration":
        return [{"module": _qual(_text(c)), "name": None, "asname": None, "level": 0} for c in node.named_children if c.type == "namespace_use_clause"]
    if t == "import_header":        # kotlin
        full = _text(node).replace("import", "").strip()
        pkg, _, last = full.rpartition(".")
        return [{"module": pkg or full, "name": None if last in ("*", "") or not pkg else last, "asname": None, "level": 0, "full": full}]
    if t == "import_list":
        return [i for c in node.named_children for i in _imports_of(c, language)]
    return out


def _is_upper(name: str) -> bool:
    letters = [c for c in name if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def _unwrap(node):
    """Wrappers such as export / template / decorator: take the real declaration inside."""
    while node.type in _WRAPPERS:
        inner = node.child_by_field_name("declaration") or node.child_by_field_name("definition") or _first_child_of_types(node, _COMMON_FUNCTION | _COMMON_CLASS | {"lexical_declaration", "variable_declaration"})
        if inner is None:
            break
        node = inner
    return node


def _arrow_function(node):
    """JS/TS: const f = (...) => {...} also counts as a function. Returns (name node, arrow function
    node) or None."""
    if node.type not in ("lexical_declaration", "variable_declaration"):
        return None
    for d in node.named_children:
        if d.type == "variable_declarator":
            name, value = d.child_by_field_name("name"), d.child_by_field_name("value")
            if value is not None and value.type in ("arrow_function", "function_expression", "function"):
                return name, value
    return None


def _class_name_and_methods(node, language: str):
    """Class name, list of method nodes, list of base class names, and the host name for methods (a
    rust impl hangs off its type)."""
    name = None
    bases: list[str] = []
    if node.type == "impl_item":
        typ = node.child_by_field_name("type")
        name = _qual(_text(typ)) if typ is not None else None
        trait = node.child_by_field_name("trait")
        if trait is not None:
            bases.append(_qual(_text(trait)))
    else:
        n = _name_node(node)
        name = _qual(_text(n)) if n is not None else None
    for field in ("superclass", "interfaces", "class_heritage", "base_clause", "class_interface_clause", "extend"):
        h = node.child_by_field_name(field)
        if h is not None:
            bases += [_qual(_text(c)) for c in h.named_children if (c.type in _NAME_TYPES or c.type.endswith("identifier") or c.type in ("type_list", "user_type")) and _text(c).lower() not in _ACCESS_WORDS]
    for ch in node.named_children:
        if ch.type in ("class_heritage", "base_class_clause", "base_list", "superclass", "super_interfaces", "inheritance_specifier",
                       "delegation_specifier", "extends_clause", "base_clause", "class_interface_clause"):
            for c in ch.named_children:
                txt = _qual(_text(c)).split("(")[0].split("<")[0]
                if txt and txt not in bases and c.type not in _COMMENTS and txt.lower() not in _ACCESS_WORDS and c.type != "access_specifier":
                    bases.append(txt)
        if ch.type == "superclass" and not ch.named_children:
            bases.append(_qual(_text(ch)).lstrip("< ").strip())
    body = node.child_by_field_name("body") or _first_child_of_types(node, {"class_body", "declaration_list", "field_declaration_list", "template_body", "body_statement", "interface_body", "protocol_body", "enum_body", "enum_class_body"})
    methods = []
    if body is not None:
        for m in body.named_children:
            m2 = _unwrap(m)
            if m2.type in _COMMON_FUNCTION:
                methods.append(m2)
            elif m2.type == "field_declaration" and m2.child_by_field_name("declarator") is not None and m2.child_by_field_name("declarator").type == "function_declarator":
                continue     # a method only declared inside a C++ class: defined elsewhere
    if not bases:
        bases = [b for b in bases]
    return name, methods, [b for b in bases if b and b.lower() not in ("object",)], body


def symbol_blocks(source: str, language: str, *, parser_profile: str = PROFILE, doc_type: str = "code") -> list[ParsedBlock] | None:
    parser = _get_parser(language)
    if parser is None:
        return None
    # Lines must be split the same way the syntax tree numbers them: tree-sitter only knows \n, so \r\n and \r
    # are normalized first and the text is split on \n alone. str.splitlines also breaks at form feeds, \x85
    # and U+2028, and a single one in the file shifts the text taken for every symbol after it
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    data = source.encode("utf-8", errors="ignore")
    tree = parser.parse(data)
    root = tree.root_node
    lines = source.split("\n")
    covered = [False] * (len(lines) + 2)
    symbols: list[tuple[int, int, dict[str, Any], list[str]]] = []
    imports: list[dict[str, Any]] = []
    constants: list[str] = []
    top_calls: list[str] = []
    file_doc = ""

    def add_symbol(start: int, end: int, meta: dict[str, Any], section: list[str]) -> None:
        symbols.append((start, end, meta, section))

    def take(start: int, end: int) -> str:
        for i in range(start, min(end, len(lines)) + 1):
            covered[i] = True
        return "\n".join(lines[start - 1:end])

    def handle_class(node, prefix: str, siblings: _Siblings, doc_node=None) -> None:
        name, methods, bases, body = _class_name_and_methods(node, language)
        doc_node = doc_node or node
        if not name:
            return
        qual = f"{prefix}{name}"
        cstart, cend = _start_row(doc_node, siblings) + 1, doc_node.end_point[0] + 1
        body_children = _Siblings(body.named_children if body is not None else [])
        flavor = node.type.replace("_declaration", "").replace("_item", "").replace("_specifier", "").replace("_definition", "").replace("_statement", "")
        method_names = []
        ranges = []
        for m in methods:
            mn = _name_node(m)
            mname = _qual(_text(mn)) if mn is not None else None
            if not mname:
                continue
            mstart = _start_row(m, body_children) + 1
            if mstart <= node.start_point[0] + 1:
                method_names.append(mname)          # on the same line as the class head (one-line class): record the name only, no separate block
                continue
            ranges.append((mstart, m.end_point[0] + 1, m, mname))
        ranges.sort(key=lambda r: r[:2])
        head_end = (ranges[0][0] - 1) if ranges else cend
        if node.type != "impl_item":
            add_symbol(cstart, max(cstart, head_end), {
                "kind": "class", "flavor": flavor, "name": name, "qualname": qual, "lineno": cstart, "end_lineno": cend,
                "signature": _signature(node), "docstring": _doc_before(doc_node, siblings), "bases": bases,
                "methods": method_names + [r[3] for r in ranges], "fields": [], "calls": [], "decorators": [],
            }, list(filter(None, prefix.rstrip(".").split("."))))
        else:
            # impl block: the implemented trait counts as a base class, recorded on the class head of
            # the same name (same key when merging)
            add_symbol(cstart, max(cstart, head_end), {
                "kind": "class", "flavor": "impl", "name": name, "qualname": qual, "lineno": cstart, "end_lineno": cend,
                "signature": _signature(node), "docstring": _doc_before(doc_node, siblings), "bases": bases,
                "methods": method_names + [r[3] for r in ranges], "fields": [], "calls": [], "decorators": [],
            }, list(filter(None, prefix.rstrip(".").split("."))))
        for i, (mstart, mend, m, mname) in enumerate(ranges):
            mend2 = (ranges[i + 1][0] - 1) if i + 1 < len(ranges) else cend
            add_symbol(mstart, mend2, {
                "kind": "method", "name": mname, "qualname": f"{qual}.{mname}", "lineno": mstart, "end_lineno": mend,
                "signature": _signature(m), "docstring": _doc_before(m, body_children), "calls": _calls_in(m),
                "class": qual, "decorators": [],
            }, list(filter(None, qual.split("."))))

    def handle_function(node, name: str, prefix: str, siblings: _Siblings, kind: str = "function", cls: str | None = None, doc_node=None) -> None:
        doc_node = doc_node or node
        start, end = _start_row(doc_node, siblings) + 1, doc_node.end_point[0] + 1
        meta = {"kind": kind, "name": name.rsplit(".", 1)[-1], "qualname": f"{prefix}{name}", "lineno": start, "end_lineno": end,
                "signature": _signature(node), "docstring": _doc_before(doc_node, siblings), "calls": _calls_in(node), "decorators": []}
        if cls:
            meta["class"] = cls
        add_symbol(start, end, meta, list(filter(None, (cls or prefix.rstrip(".")).split("."))))

    def note_calls(node) -> None:
        if len(top_calls) < MAX_CALLS:
            top_calls.extend(c for c in _calls_in(node) if c not in top_calls)

    def visit(container, prefix: str) -> None:
        children = _Siblings(container.named_children)
        for raw in children:
            node = _unwrap(raw)
            t = node.type
            if language == "bash" and t == "command":
                nn = node.child_by_field_name("name")
                args = [c for c in node.named_children if c.type in ("word", "string", "raw_string", "concatenation")]
                if nn is not None and _text(nn) in ("source", ".") and args:
                    imports.append({"module": _text(args[0]).strip("\"'"), "name": None, "asname": None, "level": 1})
                    continue
            if language == "ruby" and t == "call":
                m = node.child_by_field_name("method")
                if m is not None and _text(m) in ("require", "require_relative", "load"):
                    mod_name = _string_in(node)
                    if mod_name:
                        imports.append({"module": mod_name, "name": None, "asname": None, "level": 1 if _text(m) == "require_relative" else 0})
                        continue
            if language == "lua" and t in ("variable_declaration", "assignment_statement"):
                txt = _text(node)
                if "require" in txt:
                    mod_name = _string_in(node)
                    nn = _name_node(node)
                    if mod_name:
                        imports.append({"module": mod_name, "name": None, "asname": _text(nn) if nn is not None else None, "level": 0})
                        continue
            if t in _IMPORT_TYPES or (t == "expression_statement" and language == "php"):
                got = _imports_of(node, language)
                if not got and language == "php":
                    inner = node.named_children[0] if node.named_children else None
                    if inner is not None and inner.type in ("require_once_expression", "require_expression", "include_expression", "include_once_expression"):
                        got = [{"module": _string_in(inner), "name": None, "asname": None, "level": 1}]
                imports.extend(got)
                continue
            if t in _TRANSPARENT:
                visit(node, prefix)
                continue
            if t in _NAMESPACE:
                nn = _name_node(node)
                ns = _qual(_text(nn)) if nn is not None else ""
                body = node.child_by_field_name("body") or _first_child_of_types(node, {"declaration_list", "body_statement"})
                if body is not None:
                    visit(body, f"{prefix}{ns}." if ns else prefix)
                continue
            if t == "type_declaration" and language == "go":
                group = _Siblings(node.named_children)
                specs = [spec for spec in group if spec.type == "type_spec"]
                for spec in specs:
                    nn = spec.child_by_field_name("name")
                    typ = spec.child_by_field_name("type")
                    if nn is not None and typ is not None and typ.type in ("struct_type", "interface_type"):
                        # In a grouped declaration type ( A ...; B ... ) each type takes only its own lines: taking
                        # the whole group put the same text into the block of every type
                        own, doc = (spec, _doc_before(spec, group)) if len(specs) > 1 else (node, "")
                        doc = doc or _doc_before(raw, children)
                        start, end = own.start_point[0] + 1, own.end_point[0] + 1
                        add_symbol(start, end, {"kind": "class", "flavor": typ.type.replace("_type", ""), "name": _text(nn), "qualname": f"{prefix}{_text(nn)}",
                                                "lineno": start, "end_lineno": end, "signature": _signature(own), "docstring": doc,
                                                "bases": [], "methods": [], "fields": [], "calls": [], "decorators": []}, [])
                continue
            if t == "method_declaration" and language == "go":
                recv = node.child_by_field_name("receiver")
                nn = node.child_by_field_name("name")
                rtype = ""
                if recv is not None:
                    for pd in recv.named_children:
                        typ = pd.child_by_field_name("type")
                        if typ is not None:
                            rtype = _text(typ).lstrip("*")
                if nn is not None:
                    handle_function(node, f"{rtype}.{_text(nn)}" if rtype else _text(nn), prefix, children, kind="method" if rtype else "function", cls=f"{prefix}{rtype}" if rtype else None, doc_node=raw)
                continue
            if t == "impl_item" or t in _COMMON_CLASS:
                handle_class(node, prefix, children, doc_node=raw)
                continue
            if t in _COMMON_FUNCTION:
                nn = _name_node(node)
                name = _qual(_text(nn)) if nn is not None else ""
                if not name:
                    continue
                if "." in name:      # lua function Engine:start / C++ Engine::check defined outside the class
                    cls, _, mname = name.rpartition(".")
                    handle_function(node, name, prefix, children, kind="method", cls=f"{prefix}{cls}", doc_node=raw)
                else:
                    handle_function(node, name, prefix, children, doc_node=raw)
                continue
            arrow = _arrow_function(node)
            if arrow is not None:
                nn, fn = arrow
                handle_function(node, _text(nn), prefix, children, doc_node=raw)
                continue
            if t in ("lexical_declaration", "variable_declaration", "const_declaration", "const_item", "static_item", "preproc_def",
                     "variable_assignment", "assignment", "property_declaration", "const_declaration", "val_definition", "expression_statement"):
                nn = _name_node(node) if t not in ("const_declaration", "lexical_declaration", "variable_declaration", "expression_statement") else None
                names = [_text(nn)] if nn is not None else []
                if not names:
                    for d in node.named_children:
                        dn = d.child_by_field_name("name") or d.child_by_field_name("left")
                        if dn is not None:
                            names.append(_text(dn))
                    if t == "expression_statement" and node.named_children and node.named_children[0].type == "assignment_expression":
                        left = node.named_children[0].child_by_field_name("left")
                        if left is not None:
                            names.append(_text(left))
                for nm in names:
                    if nm and _is_upper(nm.split(".")[-1]) and nm not in constants:
                        constants.append(nm.split("$")[-1])
                note_calls(node)
                continue
            note_calls(node)

    top = list(root.named_children)
    if top and top[0].type in _COMMENTS:
        file_doc = _clean_comment(_text(top[0]))
    visit(root, "")

    blocks: list[ParsedBlock] = []
    symbols.sort(key=lambda s: (s[0], s[1]))
    for idx, (start, end, meta, section) in enumerate(symbols, start=1):
        text = take(start, end)
        if not text.strip():
            continue
        safe = re.sub(r"[^A-Za-z0-9_.]", "_", meta["qualname"])[:80]
        blocks.append(ParsedBlock(parser="tree-sitter", parser_profile=parser_profile, doc_type=doc_type, block_type="code",
                                  text=text, title=meta["qualname"], block_id=f"sym-{idx:04d}-{safe}",
                                  metadata={"section_path": list(section), "symbol": meta}))
    rest = [lines[i - 1] for i in range(1, len(lines) + 1) if not covered[i]]
    rest_text = "\n".join(rest).strip("\n")
    module_meta = {"kind": "module", "name": "", "qualname": "", "docstring": file_doc, "imports": imports,
                   "constants": constants[:60], "calls": top_calls[:MAX_CALLS], "language": language, "has_error": bool(root.has_error)}
    if rest_text.strip():
        blocks.insert(0, ParsedBlock(parser="tree-sitter", parser_profile=parser_profile, doc_type=doc_type, block_type="code",
                                     text=rest_text, title="(module)", block_id="sym-0000-module",
                                     metadata={"section_path": [], "symbol": module_meta}))
    elif blocks:
        blocks[0].metadata["module"] = module_meta
    return blocks


def parse_code(path, doc_type: str, parser_profile: str = PROFILE) -> list[ParsedBlock] | None:
    """Chunk by symbol; returns None for an unsupported language / missing tree-sitter / empty file, and
    the caller falls back to plain text."""
    from .common import read_text_smart

    language = language_for(path.name)
    if language is None:
        return None
    source = read_text_smart(path)
    if not source.strip():
        return []
    try:
        return symbol_blocks(source, language, parser_profile=parser_profile, doc_type=doc_type)
    except Exception:
        return None
