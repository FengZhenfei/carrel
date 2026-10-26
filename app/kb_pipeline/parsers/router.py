from __future__ import annotations

from pathlib import Path

from ..models import ParsedBlock
from ..utils import infer_doc_type
from .common import markdown_to_blocks, parser_profile_for_path, read_text_smart
from .errors import NonRetryableParseError
from .native import parse_text
from .formats import EXTRA_TEXT_SUFFIXES, WORKFLOW_FILENAMES, extensionless_script
from .source_guard import require_source_content

TEXT_SUFFIXES = {
    *EXTRA_TEXT_SUFFIXES,
    ".md", ".markdown", ".txt", ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go",
    ".rs", ".c", ".cc", ".cpp", ".h", ".hpp", ".cs", ".php", ".rb", ".swift", ".kt",
    ".kts", ".scala", ".sql", ".sh", ".bash", ".zsh", ".fish", ".ps1", ".yaml",
    ".yml", ".json", ".toml", ".ini", ".cfg", ".conf", ".xml", ".css",
    ".scss", ".vue", ".svelte",
}
TEXT_FILENAMES = {"Dockerfile", "Makefile", "Gemfile", "Rakefile", "Jenkinsfile", *WORKFLOW_FILENAMES}


def parse_native(path: Path, mime_type: str = "") -> list[ParsedBlock]:
    require_source_content(path)
    doc_type = infer_doc_type(path.name, mime_type)
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".rst", ".tex", ".bib", ".feature", ".xsd"}:
        from .text_formats import parse_text_format
        return parse_text_format(path)
    if suffix in {".md", ".markdown", ".qmd", ".rmd"}:
        # Split on headings so chunks align with sections and carry
        # section_path; the chunker still subdivides long sections. Falls back
        # to one text block when a file has no headings at all.
        blocks = markdown_to_blocks(
            read_text_smart(path),
            parser="native",
            parser_profile=parser_profile_for_path(path),
            doc_type=doc_type,
        )
        if blocks:
            return blocks
        # A file containing only markup comments has no visible document body.
        from .common import strip_markdown_comments
        if not strip_markdown_comments(read_text_smart(path)).strip():
            return []
        return parse_text(path, doc_type, parser_profile_for_path(path))
    if suffix == ".py":
        from .code_python import parse_python

        return parse_python(path, doc_type, parser_profile_for_path(path))
    from .code_symbols import LANGUAGE_BY_SUFFIX, parse_code

    if suffix in LANGUAGE_BY_SUFFIX:
        blocks = parse_code(path, doc_type, parser_profile_for_path(path))
        if blocks is not None:
            return blocks
        # tree-sitter missing / syntax tree failed: fall back to one block for the whole file, profile
        # unchanged (the block-level profile determines chunk_uid)
    if suffix in TEXT_SUFFIXES or path.name in TEXT_FILENAMES or suffix in LANGUAGE_BY_SUFFIX or extensionless_script(path):
        # When the syntax tree is unavailable, every supported code suffix falls back to whole-file
        # text; parsing is no longer refused because the fallback whitelist lacks an entry (final
        # review S02)
        blocks = parse_text(path, doc_type, parser_profile_for_path(path))
        if suffix in {".nf", ".cwl", ".wdl", ".dot"} or path.name in WORKFLOW_FILENAMES or extensionless_script(path):
            for block in blocks:
                block.block_type = "code"
                block.title = path.name
                block.metadata["source_locator"] = {"kind": "lines", "start": 1, "end": len(block.text.splitlines())}
        return blocks
    # csv/xlsx/xls and html never reach here: parse_job routes them to
    # parse_native_table / parse_html_dom before falling back to this router.
    # Deterministic failure: a format with no parse route gives the same result however often it is
    # retried, so do not go through 5 rounds of backoff (2026-09-06 health check B6)
    raise NonRetryableParseError(f"parser route for {suffix or doc_type} is not enabled in this phase")
