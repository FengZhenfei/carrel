#!/usr/bin/env python3
"""Prepare the supported code grammars while building the application image."""
import os

import tree_sitter_language_pack as pack

from kb_pipeline.parsers.code_symbols import LANGUAGE_BY_SUFFIX


def main():
    if not os.getenv("TREE_SITTER_LANGUAGE_PACK_CACHE_DIR", "").strip():
        raise SystemExit("TREE_SITTER_LANGUAGE_PACK_CACHE_DIR must name the image's parser cache")
    # Markdown comment handling uses these grammars even in documents with no
    # fenced source code. Runtime images must not need to download them.
    languages = sorted(set(LANGUAGE_BY_SUFFIX.values()) | {"markdown", "markdown_inline", "html"})
    print(f"Preparing {len(languages)} parser grammars", flush=True)
    missing = sorted(set(languages) - set(pack.downloaded_languages()))
    if missing:
        pack.download(missing)
    # A missing or incompatible grammar must fail the build, rather than silently
    # falling back to plain text when a user first imports a code file.
    for language in languages:
        pack.get_parser(language).parse(b"")
    print(f"Prepared {len(languages)} parser grammars in {pack.cache_dir()}")


if __name__ == "__main__":
    main()
