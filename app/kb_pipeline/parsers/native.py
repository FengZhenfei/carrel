from __future__ import annotations

from pathlib import Path

from ..models import ParsedBlock
from .common import read_text_smart


def parse_text(path: Path, doc_type: str, parser_profile: str | None = None) -> list[ParsedBlock]:
    text = read_text_smart(path)
    if parser_profile is None:
        parser_profile = f"{doc_type}-native-v1" if doc_type else "text-native-v1"
    return [
        ParsedBlock(
            parser="native",
            parser_profile=parser_profile,
            doc_type=doc_type,
            block_type="text",
            text=text,
            block_id="text-0001",
        )
    ]
