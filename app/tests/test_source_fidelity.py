"""Synthetic source-fidelity invariants, independent of any enrolled knowledge base."""
from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from kb_pipeline.chunking.chunker import blocks_to_chunks, split_code
from kb_pipeline.chunking.context import evidence_text
from kb_pipeline.models import ParsedBlock
from kb_pipeline.parsers.code_python import python_symbol_blocks
from kb_pipeline.parsers.router import parse_native
from kb_pipeline.parsers.common import markdown_to_blocks
from kb_pipeline.utils import count_tokens


def chunks(blocks, budget=64):
    return blocks_to_chunks(kb_id="test", file_key=1, content_version="v", parser_profile="test",
                            blocks=blocks, max_tokens=budget, overlap_tokens=0)


class SourceFidelityTests(unittest.TestCase):
    def test_markdown_literal_code_and_long_fences(self):
        raw = '    text = "a\u2028b. c"  \n\tvalue = "' + ('ab  中!\t' * 400) + '"\n```\n# still code\n'
        md = '# Guide\n\nBefore.\n\n````python\n' + raw + '````\n\nAfter.\n'
        blocks = markdown_to_blocks(md, parser="md", parser_profile="md", doc_type="md")
        result = chunks(blocks, 64)
        code = [c for c in result if c.block.block_type == "code"]
        self.assertEqual("".join(c.text for c in code), raw)
        self.assertTrue(all(count_tokens(c.text) <= 64 for c in code))
        self.assertTrue(any("Before." in evidence_text(c.text, c.block.metadata) for c in result))
        self.assertTrue(any("After." in c.text for c in result))
        self.assertTrue(all(c.block.metadata["section_path"] == ["Guide"] for c in code))
        self.assertTrue(all(c.block.metadata["code_language"] == "python" for c in code))

    def test_partition_preserves_every_character(self):
        rng = random.Random(4381)
        texts = ["\t  " + "x = '句子。 spaces. kept!'  \n" * 80,
                 '    payload = "' + "ab  中!\t" * 500 + '"\n',
                 "first\r\n\tsecond\r\n", " \n\n  ",
                 "".join(rng.choice("abc中文  \n\t:{}()!") for _ in range(5000))]
        for text in texts:
            for budget in (8, 32, 127, 400):
                with self.subTest(budget=budget, size=len(text)):
                    parts = split_code(text, budget)
                    self.assertEqual("".join(parts), text)
                    self.assertTrue(all(count_tokens(p) <= budget for p in parts))

    def test_real_symbol_boundaries_keep_indent_literals_and_locations(self):
        source = ('import os\n\nclass Worker:\n    note = "first. second!"\n\n'
                  '    def run(self):\n        message = "keep. this. intact"\n' +
                  ''.join(f'        value_{i} = {i}  # unchanged. punctuation!\n' for i in range(45)) +
                  '        return message\n\n    def stop(self):\n        return None\n\nflag = True\n')
        result = chunks(python_symbol_blocks(source))
        lines = source.split("\n")
        self.assertGreater(len(result), 3)
        for chunk in result:
            self.assertFalse(chunk.text.startswith("TITLE:"))
            loc = chunk.block.metadata["source_locator"]
            ranges = loc.get("ranges", [loc])
            fragments = []
            for r in ranges:
                original = "\n".join(lines[r["start"]-1:r["end"]])
                if r["start"] == r["end"]:
                    original = original[r["start_column"]:r["end_column"]]
                else:
                    original = original[r["start_column"]:len(original)-len(lines[r["end"]-1])+r["end_column"]]
                fragments.append(original)
            # Separators between adjacent source runs are actual source line breaks.
            self.assertEqual(chunk.text.strip("\n"), "\n".join(fragments).strip("\n"))
        self.assertTrue(any('        message = "keep. this. intact"' in c.text for c in result))
        self.assertTrue(any('    def stop(self)' in c.text for c in result))

    def test_one_long_code_block_reassembles_without_titles_or_whitespace_changes(self):
        raw = "    if ready:\n" + "        emit('one. two!')  \n" * 100
        block = ParsedBlock(parser="fixture", parser_profile="fixture", doc_type="code", block_type="code",
                            text=raw, title="not source text", block_id="code", metadata={"symbol":{"kind":"function"}})
        self.assertEqual("".join(c.text for c in chunks([block])), raw)

    def test_document_listing_caption_survives_outside_the_code(self):
        source = "    return 'unchanged. code'\n"
        block = ParsedBlock(parser="fixture", parser_profile="fixture", doc_type="pdf", block_type="code",
                            text=source, caption="Listing: output operation", block_id="listing", page_idx=3)
        result = chunks([block])
        self.assertTrue(any("Listing: output operation" in evidence_text(c.text, c.block.metadata) for c in result))
        self.assertEqual("".join(c.text for c in result if c.block.block_type=="code"), source)
        self.assertTrue(all(c.block.page_idx==3 for c in result))

    def test_invalid_python_stays_code(self):
        source = "    this is not valid python: (\n" + "\tbroken = 'a. b.'\n" * 30
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"unfinished.py"; path.write_text(source)
            blocks = parse_native(path)
        self.assertTrue(all(b.block_type == "code" for b in blocks))
        self.assertEqual("".join(c.text for c in chunks(blocks)), source)

    def test_plain_prose_does_not_take_the_code_route(self):
        text = "A sentence. Another sentence.\n第三句话。第四句话。"
        block = ParsedBlock(parser="fixture", parser_profile="fixture", doc_type="txt", block_type="text", text=text, block_id="prose")
        result = chunks([block], 200)
        self.assertEqual(result[0].text, "A sentence.\nAnother sentence.\n第三句话。\n第四句话。")
        self.assertNotIn("code_fragment", result[0].block.metadata)
