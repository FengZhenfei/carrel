"""Regression tests for all 5 Codex review findings (Patch 12) plus the pre-existing Light keyword sync fix.
Synthetic fixtures only; no real document filenames or body text."""
from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
import uuid
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from kb_pipeline import db
from kb_pipeline.chunking.chunker import blocks_to_chunks
from kb_pipeline.embedding.client import EmbeddingClient, EmbeddingConfigError
from kb_pipeline.embedding.visual import VisualEmbedConfigError, VisualEmbeddingClient
from kb_pipeline.models import ParsedBlock, SourceFile, UnifiedChunk
from kb_pipeline.parsers import docx_enhanced, mineru_pdf
from kb_pipeline.parsers.common import html_table_to_markdown
from kb_pipeline.parsers.errors import (
    FAILURE_CONFIG,
    FAILURE_SCOPE_EMBEDDING,
    FAILURE_SCOPE_KEY,
    FAILURE_SCOPE_MINERU,
    FAILURE_SCOPE_VISUAL_EMBEDDING,
    FAILURE_VERDICT_KEY,
)
from kb_pipeline.parsers.mineru_pdf import MinerUServiceError
from kb_pipeline.pipeline import scheduler, worker
from kb_pipeline.storage.light import LightStorage
from kb_pipeline.storage.standard import StandardStorage
from kb_pipeline.vector.layout import TEXT_VECTOR, VectorLayout
from kb_server import model_endpoints, service


def _source_file(kb_id: str, file_key: int, rel_path: str, version: str = "v1") -> SourceFile:
    return SourceFile(
        kb_id=kb_id,
        collection=f"col_{kb_id}",
        source_root=f"dir_{kb_id}",
        source_type="local",
        file_key=file_key,
        source_path=f"/{rel_path}",
        rel_path=rel_path,
        dir="",
        filename=Path(rel_path).name,
        physical_path=f"/tmp/{rel_path}",
        size=100,
        mtime=1700000000,
        mime_type="text/markdown" if rel_path.endswith(".md") else "application/pdf",
        checksum=version,
    )


def _insert_kb(con: sqlite3.Connection, kb_id: str) -> None:
    con.execute(
        "INSERT INTO kb_sources (kb_id, collection, source_root, status, first_seen_at, last_seen_at, config_json) "
        "VALUES (?, ?, ?, 'active', 1, 1, '{}')",
        (kb_id, f"col_{kb_id}", f"dir_{kb_id}"),
    )


class Item1MergedCellAttributionTests(unittest.TestCase):
    def test_long_shared_rowspan_note_survives_per_row_and_across_table_chunk_splits(self) -> None:
        note = (
            "URL format validation required: must start with http:// or https://, supports relative or absolute "
            "paths, and displays an inline validation error whenever the entered value does not match the format. "
            "Applies to every row in this parameter group regardless of chunk boundary."
        )
        rows_html = [
            "<tr><th>Field</th><th>Required</th><th>Validation Rule</th></tr>",
            f"<tr><td>field_0</td><td>Yes</td><td rowspan='6'>{note}</td></tr>",
            *[f"<tr><td>field_{i}</td><td>Yes</td></tr>" for i in range(1, 6)],
        ]
        md = html_table_to_markdown("<table>" + "".join(rows_html) + "</table>")
        self.assertIsNotNone(md)
        for i in range(6):
            self.assertIn(f"| field_{i} | Yes | {note} |", md)

        # Force the table to split across multiple chunks and verify every split chunk preserves the note
        block = ParsedBlock(
            parser="mineru",
            parser_profile="p",
            doc_type="pdf",
            block_type="table",
            text=md,
            table_markdown=md,
            block_id="t1",
            page_idx=1,
            metadata={"section_path": ["API Settings"]},
        )
        chunks = blocks_to_chunks(
            kb_id="kb",
            file_key=1,
            content_version="v1",
            parser_profile="p",
            blocks=[block],
            max_tokens=160,
            overlap_tokens=0,
        )
        self.assertGreaterEqual(len(chunks), 2)
        seen_fields: set[str] = set()
        for ch in chunks:
            for i in range(6):
                tag = f"field_{i}"
                if tag in ch.text:
                    seen_fields.add(tag)
                    self.assertIn("http:// or https://", ch.text, f"{tag} lost shared note in split chunk")
        self.assertEqual(seen_fields, {f"field_{i}" for i in range(6)})


class Item2UIControlsAndScreenshotHeadingsTests(unittest.TestCase):
    def test_embedded_screenshot_ocr_labels_stay_searchable_body_while_native_headings_survive(self) -> None:
        import pymupdf
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "manual.pdf"
            buf = io.BytesIO()
            Image.new("RGB", (300, 300), (240, 240, 240)).save(buf, format="PNG")
            img_bytes = buf.getvalue()

            doc = pymupdf.open()
            # Page 1: native heading + enough native prose so document is digital
            p1 = doc.new_page(width=600, height=800)
            p1.insert_text((30, 60), "Step 2: Create Connector", fontsize=16)
            p1.insert_text(
                (30, 100),
                "Follow the instructions below to register a connector in the console. " * 4,
                fontsize=11,
            )
            # Page 2: embedded UI screenshot at y=80..560 (norm y=100..700), native heading below at y=660 (norm 825)
            p2 = doc.new_page(width=600, height=800)
            p2.insert_image(pymupdf.Rect(30, 80, 360, 560), stream=img_bytes)
            p2.insert_text((30, 660), "Step 3: Verify and Call", fontsize=16)
            p2.insert_text((30, 700), "Invoke the connector using the standard API.", fontsize=11)
            doc.save(str(pdf_path))
            doc.close()

            content = [
                {"type": "text", "text": "Step 2: Create Connector", "text_level": 1, "page_idx": 0, "bbox": [50, 60, 400, 85]},
                {"type": "text", "text": "Follow the instructions below.", "page_idx": 0, "bbox": [50, 115, 500, 140]},
                # Inside screenshot on page 2 (norm rect x=50..600, y=100..700): no native PDF text!
                {"type": "text", "text": "New Connector", "text_level": 1, "page_idx": 1, "bbox": [80, 130, 240, 155]},
                {"type": "text", "text": "<- Back to templates", "text_level": 1, "page_idx": 1, "bbox": [80, 180, 280, 205]},
                {"type": "text", "text": "Selected template: Example Legal Search", "page_idx": 1, "bbox": [90, 230, 420, 255]},
                {"type": "text", "text": "On this page", "text_level": 1, "page_idx": 1, "bbox": [450, 130, 560, 150]},
                # Real native heading & body below screenshot on page 2
                {"type": "text", "text": "Step 3: Verify and Call", "text_level": 1, "page_idx": 1, "bbox": [48, 810, 360, 838]},
                {"type": "text", "text": "Invoke the connector using the standard API.", "page_idx": 1, "bbox": [48, 860, 460, 885]},
            ]
            with patch.object(mineru_pdf, "call_mineru_sync", return_value={"content_list": content}):
                blocks = mineru_pdf.mineru_pdf_blocks(
                    mineru_url="http://x", path=pdf_path, cache_dir=Path(tmp) / "cache"
                )

            by_text = {b.text: b for b in blocks}
            # UI labels remain searchable body text, NOT headings, and stay under Step 2
            for ui_txt in ("New Connector", "<- Back to templates", "On this page"):
                self.assertIn(ui_txt, by_text)
                self.assertEqual(by_text[ui_txt].block_type, "text")
                self.assertEqual(by_text[ui_txt].metadata["section_path"], ["Step 2: Create Connector"])
            self.assertEqual(
                by_text["Selected template: Example Legal Search"].metadata["section_path"],
                ["Step 2: Create Connector"],
            )
            # Real native heading on the same page survives and owns its body
            self.assertEqual(by_text["Step 3: Verify and Call"].block_type, "title")
            self.assertEqual(
                by_text["Invoke the connector using the standard API."].metadata["section_path"],
                ["Step 3: Verify and Call"],
            )

    def test_scanned_pdf_and_native_heading_over_banner_image_are_not_demoted(self) -> None:
        import pymupdf
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            buf = io.BytesIO()
            Image.new("RGB", (300, 300), (220, 220, 220)).save(buf, format="PNG")
            img_bytes = buf.getvalue()

            # 1) Pure scanned PDF (no native text): headings inside image must NOT be demoted
            scanned = Path(tmp) / "scanned.pdf"
            doc = pymupdf.open()
            p = doc.new_page(width=600, height=800)
            p.insert_image(pymupdf.Rect(30, 40, 400, 400), stream=img_bytes)
            doc.save(str(scanned))
            doc.close()

            content = [
                {"type": "text", "text": "Chapter 1 Overview", "text_level": 1, "page_idx": 0, "bbox": [80, 100, 350, 130]},
                {"type": "text", "text": "Body text from scan.", "page_idx": 0, "bbox": [80, 160, 400, 200]},
            ]
            with patch.object(mineru_pdf, "call_mineru_sync", return_value={"content_list": content}):
                blocks = mineru_pdf.mineru_pdf_blocks(
                    mineru_url="http://x", path=scanned, cache_dir=Path(tmp) / "c1"
                )
            self.assertEqual(blocks[0].block_type, "title")
            self.assertEqual(blocks[1].metadata["section_path"], ["Chapter 1 Overview"])

            # 2) Digital PDF with native heading text drawn ON TOP of a decorative banner image: preserved
            banner = Path(tmp) / "banner.pdf"
            doc = pymupdf.open()
            p = doc.new_page(width=600, height=800)
            p.insert_image(pymupdf.Rect(30, 40, 500, 200), stream=img_bytes)
            p.insert_text((50, 100), "Product Overview", fontsize=18)
            p.insert_text((50, 260), "Detailed technical overview of the platform. " * 6, fontsize=11)
            doc.save(str(banner))
            doc.close()

            content2 = [
                {"type": "text", "text": "Product Overview", "text_level": 1, "page_idx": 0, "bbox": [80, 105, 360, 135]},
                {"type": "text", "text": "Detailed technical overview.", "page_idx": 0, "bbox": [80, 320, 500, 360]},
            ]
            with patch.object(mineru_pdf, "call_mineru_sync", return_value={"content_list": content2}):
                blocks2 = mineru_pdf.mineru_pdf_blocks(
                    mineru_url="http://x", path=banner, cache_dir=Path(tmp) / "c2"
                )
            self.assertEqual(blocks2[0].block_type, "title")
            self.assertEqual(blocks2[1].metadata["section_path"], ["Product Overview"])

    def test_recurring_card_footer_control_demoted_while_visual_summary_and_parent_steps_preserved(self) -> None:
        # Card layout: 4 cards, each has real recurring subsection "Summary" + right-shifted footer control "Collapse"
        # followed by tiny avatar icon (area < 2000) and next card heading at left margin.
        card_items = []
        for idx in range(4):
            y0 = 50 + idx * 200
            card_items += [
                {"type": "text", "text": f"Case {idx + 1}", "text_level": 1, "page_idx": 0, "bbox": [100, y0, 260, y0 + 15]},
                {"type": "text", "text": "Summary", "text_level": 1, "page_idx": 0, "bbox": [80, y0 + 25, 160, y0 + 38]},
                {"type": "text", "text": f"Deployment details for case {idx + 1}.", "page_idx": 0, "bbox": [80, y0 + 45, 300, y0 + 90]},
                {"type": "text", "text": "Collapse", "text_level": 1, "page_idx": 0, "bbox": [240, y0 + 105, 290, y0 + 118]},
                {"type": "image", "img_path": "", "page_idx": 0, "bbox": [80, y0 + 185, 100, y0 + 205]},  # tiny 20x20 avatar
            ]
        with tempfile.TemporaryDirectory() as tmp, patch.object(mineru_pdf, "call_mineru_sync", return_value={"content_list": card_items}):
            blocks = mineru_pdf.mineru_pdf_blocks(mineru_url="http://x", path=Path(tmp) / "cards.pdf", cache_dir=Path(tmp) / "c")
        collapse_blocks = [b for b in blocks if b.text == "Collapse"]
        self.assertEqual(len(collapse_blocks), 4)
        self.assertTrue(all(b.block_type == "text" for b in collapse_blocks))
        self.assertTrue(all("Collapse" not in (b.metadata.get("section_path") or []) for b in blocks))
        # Real recurring subsection "Summary" stays a title
        summary_blocks = [b for b in blocks if b.text == "Summary"]
        self.assertEqual(len(summary_blocks), 4)
        self.assertTrue(all(b.block_type == "title" for b in summary_blocks))

        # Positive controls: recurring "Visual Summary" followed by large figure (area >> 2000)
        # AND recurring "Procedure" followed directly by same-column child heading below it.
        pos_items = []
        for idx in range(4):
            pos_items += [
                {"type": "text", "text": f"Chapter {idx + 1}", "text_level": 1, "page_idx": idx, "bbox": [60, 60, 260, 85]},
                {"type": "text", "text": "Procedure", "text_level": 1, "page_idx": idx, "bbox": [60, 110, 180, 130]},
                {"type": "text", "text": f"Step {idx + 1}.1 Setup", "text_level": 1, "page_idx": idx, "bbox": [60, 145, 240, 165]},
                {"type": "text", "text": "Run the setup command.", "page_idx": idx, "bbox": [60, 180, 460, 210]},
                {"type": "text", "text": "Visual Summary", "text_level": 1, "page_idx": idx, "bbox": [60, 240, 220, 260]},
                {"type": "image", "text": "diagram caption", "page_idx": idx, "bbox": [60, 280, 660, 680]},  # large 600x400 diagram
            ]
        with tempfile.TemporaryDirectory() as tmp, patch.object(mineru_pdf, "call_mineru_sync", return_value={"content_list": pos_items}):
            pos_blocks = mineru_pdf.mineru_pdf_blocks(mineru_url="http://x", path=Path(tmp) / "book.pdf", cache_dir=Path(tmp) / "c2")
        self.assertEqual(sum(b.block_type == "title" and b.text == "Visual Summary" for b in pos_blocks), 4)
        self.assertEqual(sum(b.block_type == "title" and b.text == "Procedure" for b in pos_blocks), 4)
        img_paths = [b.metadata.get("section_path") for b in pos_blocks if b.block_type == "image"]
        self.assertTrue(all("Visual Summary" in p for p in img_paths))


class Item3RestorePartialVectorGCTests(unittest.TestCase):
    def _seed_deleted_file_with_two_chunks(self, con: sqlite3.Connection, kb_id: str, file_key: int) -> tuple[sqlite3.Row, list[str]]:
        sf = _source_file(kb_id, file_key, f"doc_{file_key}.md", "v1")
        db.upsert_file(con, sf, status="indexed")
        fid = db.file_id_for(kb_id, file_key)
        con.execute("UPDATE files SET indexed_version='v1', indexed_parser_profile='native-v1' WHERE file_id=?", (fid,))
        row = db.get_file_by_id(con, fid)
        p1, p2 = str(uuid.uuid4()), str(uuid.uuid4())
        b1 = ParsedBlock("native", "native-v1", "md", "text", "chunk one text", "b1", page_idx=1)
        b2 = ParsedBlock("native", "native-v1", "md", "text", "chunk two text", "b2", page_idx=1)
        db.replace_chunks(
            con,
            file_id=fid,
            collection=sf.collection,
            content_version="v1",
            chunks=[
                UnifiedChunk(chunk_uid=f"u1_{file_key}", chunk_index=0, text="chunk one text", block=b1),
                UnifiedChunk(chunk_uid=f"u2_{file_key}", chunk_index=1, text="chunk two text", block=b2),
            ],
            point_ids=[p1, p2],
        )
        db.mark_chunks_deleted(con, fid)
        db.enqueue_job(
            con,
            ingest_run_id="scan-restore",
            file_id=fid,
            kb_id=kb_id,
            collection=sf.collection,
            file_key=file_key,
            job_type="metadata_update",
            payload={"reactivate": True},
            dedupe_key=f"meta:{fid}",
        )
        con.commit()
        return row, [p1, p2]

    def _chunk_payload(self, row: sqlite3.Row, chunk_uid: str, idx: int, text: str) -> dict:
        return {
            "kb_id": str(row["kb_id"]),
            "file_key": int(row["file_key"]),
            "doc_id": f"{row['kb_id']}:{int(row['file_key'])}",
            "rel_path": str(row["rel_path"]),
            "source_path": str(row["source_path"]),
            "filename": str(row["filename"]),
            "content_version": str(row["content_version"]),
            "chunk_uid": chunk_uid,
            "chunk_index": idx,
            "block_id": f"b{idx + 1}",
            "text": text,
            "is_active": False,
            "inactive_at": 1700000000,
        }

    def test_preexisting_light_keywords_sync_document_returns_collection_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = SimpleNamespace(
                state_db=Path(tmp) / "state.db",
                cache_dir=Path(tmp) / "cache",
                runtime_dir=Path(tmp) / "runtime",
                storage_backend="light",
            )
            store = LightStorage(settings)
            # Missing collection must not crash and must include collection key
            empty = store.keywords.sync_document("col_missing", "kb1:1")
            self.assertEqual(empty, {"collection": "col_missing", "doc_id": "kb1:1", "inserted_rows": 0, "deleted_rows": 0})
            # Existing collection returns collection key matching Standard contract
            layout = VectorLayout(text_size=4, visual_size=4)
            store.chunks.ensure("col_kb1", layout)
            pid = str(uuid.uuid4())
            store.chunks.write(
                "col_kb1",
                [pid],
                [{TEXT_VECTOR: [1.0, 0.0, 0.0, 0.0]}],
                [{"doc_id": "kb1:1", "file_key": 1, "kb_id": "kb1", "content_version": "v1",
                  "chunk_uid": "u1", "chunk_index": 0, "text": "hello", "is_active": True}],
                max_bytes=1024 * 1024,
            )
            res = store.keywords.sync_document("col_kb1", "kb1:1")
            self.assertEqual(res["collection"], "col_kb1")
            self.assertEqual(res["doc_id"], "kb1:1")
            self.assertEqual(res["inserted_rows"], 1)

    def test_light_restore_three_states_all_partial_none(self) -> None:
        for mode in ("all", "partial", "none"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                state = Path(tmp) / "state.db"
                db.init_db(state)
                settings = SimpleNamespace(
                    state_db=state,
                    cache_dir=Path(tmp) / "cache",
                    runtime_dir=Path(tmp) / "runtime",
                    storage_backend="light",
                    parse_enabled=True,
                    parse_job_lease_seconds=60,
                    job_max_retries=3,
                    job_retry_base_seconds=10,
                    job_retry_max_seconds=60,
                )
                store = LightStorage(settings)
                layout = VectorLayout(text_size=4, visual_size=4)
                with db.connect(state) as con:
                    _insert_kb(con, "kb1")
                    row, (p1, p2) = self._seed_deleted_file_with_two_chunks(con, "kb1", 1)
                    store.chunks.ensure(str(row["collection"]), layout)
                    present_ids = [p1, p2] if mode == "all" else [p1] if mode == "partial" else []
                    if present_ids:
                        store.chunks.write(
                            str(row["collection"]),
                            present_ids,
                            [{TEXT_VECTOR: [1.0, 0.0, 0.0, 0.0]} for _ in present_ids],
                            [self._chunk_payload(row, f"u{i+1}_1", i, f"chunk {i+1}") for i in range(len(present_ids))],
                            max_bytes=1024 * 1024,
                        )
                    with patch.object(worker, "get_storage", return_value=store):
                        res = worker.run_once(con, settings=settings)
                    self.assertTrue(res.startswith("metadata-done:"), (mode, res))
                    f_after = db.get_file_by_id(con, str(row["file_id"]))
                    active_ledger = con.execute(
                        "SELECT COUNT(*) FROM chunks WHERE file_id=? AND status='active'", (row["file_id"],)
                    ).fetchone()[0]
                    parse_jobs = con.execute(
                        "SELECT COUNT(*) FROM jobs WHERE file_id=? AND job_type='parse' AND status='queued'",
                        (row["file_id"],),
                    ).fetchone()[0]
                    readable = store.chunks.payloads(str(row["collection"]), [p1, p2], active_only=True)

                if mode == "all":
                    self.assertEqual(str(f_after["status"]), "indexed")
                    self.assertEqual(str(f_after["indexed_version"]), "v1")
                    self.assertEqual(active_ledger, 2)
                    self.assertEqual(len(readable), 2)
                    self.assertEqual(parse_jobs, 0)
                else:
                    self.assertEqual(str(f_after["status"]), "seen")
                    self.assertIsNone(f_after["indexed_version"])
                    self.assertEqual(active_ledger, 0)
                    self.assertEqual(len(readable), 0)
                    self.assertEqual(parse_jobs, 1)

    def test_standard_qdrant_restore_three_states_all_partial_none(self) -> None:
        from qdrant_client import QdrantClient

        for mode in ("all", "partial", "none"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                state = Path(tmp) / "state.db"
                db.init_db(state)
                settings = SimpleNamespace(
                    state_db=state,
                    qdrant_url="http://in-memory",
                    qdrant_api_key=None,
                    opensearch_url="http://in-memory",
                    storage_backend="standard",
                    parse_enabled=True,
                    parse_job_lease_seconds=60,
                    job_max_retries=3,
                    job_retry_base_seconds=10,
                    job_retry_max_seconds=60,
                )
                q = QdrantClient(":memory:")
                store = StandardStorage(settings, vector_client=q)
                layout = VectorLayout(text_size=4, visual_size=4)
                with db.connect(state) as con:
                    _insert_kb(con, "kb1")
                    row, (p1, p2) = self._seed_deleted_file_with_two_chunks(con, "kb1", 2)
                    store.chunks.ensure(str(row["collection"]), layout)
                    present_ids = [p1, p2] if mode == "all" else [p1] if mode == "partial" else []
                    if present_ids:
                        store.chunks.write(
                            str(row["collection"]),
                            present_ids,
                            [{TEXT_VECTOR: [1.0, 0.0, 0.0, 0.0]} for _ in present_ids],
                            [self._chunk_payload(row, f"u{i+1}_2", i, f"chunk {i+1}") for i in range(len(present_ids))],
                            max_bytes=1024 * 1024,
                        )
                    with patch.object(worker, "get_storage", return_value=store), \
                            patch.object(
                                store.keywords,
                                "sync_document",
                                return_value={"collection": str(row["collection"]), "doc_id": "kb1:2",
                                              "deleted_rows": 0, "inserted_rows": len(present_ids)},
                            ):
                        res = worker.run_once(con, settings=settings)
                    self.assertTrue(res.startswith("metadata-done:"), (mode, res))
                    f_after = db.get_file_by_id(con, str(row["file_id"]))
                    active_ledger = con.execute(
                        "SELECT COUNT(*) FROM chunks WHERE file_id=? AND status='active'", (row["file_id"],)
                    ).fetchone()[0]
                    parse_jobs = con.execute(
                        "SELECT COUNT(*) FROM jobs WHERE file_id=? AND job_type='parse' AND status='queued'",
                        (row["file_id"],),
                    ).fetchone()[0]
                    readable = store.chunks.payloads(str(row["collection"]), [p1, p2], active_only=True)

                if mode == "all":
                    self.assertEqual(str(f_after["status"]), "indexed")
                    self.assertEqual(str(f_after["indexed_version"]), "v1")
                    self.assertEqual(active_ledger, 2)
                    self.assertEqual(len(readable), 2)
                    self.assertEqual(parse_jobs, 0)
                else:
                    self.assertEqual(str(f_after["status"]), "seen")
                    self.assertIsNone(f_after["indexed_version"])
                    self.assertEqual(active_ledger, 0)
                    self.assertEqual(len(readable), 0)
                    self.assertEqual(parse_jobs, 1)


class Item4ScopedConfigFailureRequeueTests(unittest.TestCase):
    def _base_cfg(self, state: Path, tmp: Path):
        return SimpleNamespace(
            state_db=state,
            mirror_root=tmp,
            parse_job_lease_seconds=60,
            job_max_retries=3,
            job_retry_base_seconds=10,
            job_retry_max_seconds=60,
            parse_enabled=True,
            embedding_base_url="http://127.0.0.1:9001/v1",
            embedding_model_id="bge-m3",
            embedding_api_key="bad-key",
            embedding_dim=1024,
            embedding_batch=16,
            embedding_concurrency=4,
            visual_embedding_enabled=False,
            visual_embedding_base_url="",
            visual_embedding_model_id="",
            visual_embedding_api_key="",
            visual_embedding_dim=1024,
            visual_embedding_batch=8,
            visual_embedding_concurrency=2,
            reranker_base_url="",
            reranker_model_id="",
            reranker_api_key="",
            visual_reranker_base_url="",
            visual_reranker_model_id="",
            visual_reranker_api_key="",
        )

    def test_text_embedding_401_classified_as_config_and_requeued_on_embedding_save_only(self) -> None:
        # 1) Real OpenAI SDK client receiving HTTP 401 -> EmbeddingConfigError (1 attempt, no retries)
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(401, json={"error": {"message": "Invalid API key"}}, request=request)

        from openai import AuthenticationError, OpenAI

        emb = EmbeddingClient(
            base_url="http://127.0.0.1:9001/v1",
            api_key="bad-key",
            model_id="bge-m3",
            dim=4,
            batch_size=8,
            retry=5,
            sleep_seconds=0.0,
        )
        emb.client = OpenAI(
            api_key="bad-key",
            base_url="http://127.0.0.1:9001/v1",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with self.assertRaises(AuthenticationError) as ctx:
            emb.embed(["hello world"])
        self.assertEqual(calls, 1)  # no wasted inner retries on HTTP 401
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertTrue(getattr(ctx.exception, "config_error", False))
        self.assertEqual(worker._failure_verdict(ctx.exception), FAILURE_CONFIG)
        self.assertEqual(worker._failure_scope(ctx.exception), FAILURE_SCOPE_EMBEDDING)

        # 2) Positive & negative integration through real worker + model_endpoints.save_endpoint
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.db"
            db.init_db(state)
            cfg = self._base_cfg(state, Path(tmp))
            with db.connect(state) as con:
                _insert_kb(con, "kb1")
                f_emb = _source_file("kb1", 1, "emb_fail.md", "v1")
                f_mineru = _source_file("kb1", 2, "mineru_fail.pdf", "v1")
                _, j_emb = scheduler.schedule_file(con, ingest_run_id="r1", file=f_emb)
                _, j_mineru = scheduler.schedule_file(con, ingest_run_id="r1", file=f_mineru)
                con.commit()

                # Fail f_emb via actual EmbeddingConfigError from 401; fail f_mineru via MinerU 401
                with patch.object(worker, "process_parse_job", side_effect=ctx.exception):
                    r1 = worker.run_once(con, settings=cfg)
                self.assertTrue(r1.startswith("failed:"), r1)
                with patch.object(worker, "process_parse_job", side_effect=MinerUServiceError("HTTP 401", status_code=401)):
                    r2 = worker.run_once(con, settings=cfg)
                self.assertTrue(r2.startswith("failed:"), r2)
                con.commit()

                p_emb = json.loads(con.execute("SELECT payload_json FROM jobs WHERE job_id=?", (j_emb,)).fetchone()[0])
                p_min = json.loads(con.execute("SELECT payload_json FROM jobs WHERE job_id=?", (j_mineru,)).fetchone()[0])
                self.assertEqual(p_emb[FAILURE_VERDICT_KEY], FAILURE_CONFIG)
                self.assertEqual(p_emb[FAILURE_SCOPE_KEY], FAILURE_SCOPE_EMBEDDING)
                self.assertEqual(p_min[FAILURE_VERDICT_KEY], FAILURE_CONFIG)
                self.assertEqual(p_min[FAILURE_SCOPE_KEY], FAILURE_SCOPE_MINERU)

            # Negative case: saving a disabled Reranker must NOT wake MinerU or embedding failed jobs
            rev_rr = model_endpoints._revision(model_endpoints.endpoint_values(cfg)["reranker"])
            kicked: list[int] = []
            with patch.object(model_endpoints, "load_settings", return_value=cfg), \
                    patch.object(model_endpoints, "_probe", side_effect=lambda _r, c: c), \
                    patch.object(service, "kick_worker", side_effect=lambda: kicked.append(1)):
                model_endpoints.save_endpoint("reranker", {"enabled": False, "revision": rev_rr})
            self.assertEqual(kicked, [])
            with db.connect(state) as con:
                self.assertEqual(con.execute("SELECT COUNT(*) FROM jobs WHERE status='queued'").fetchone()[0], 0)

            # Positive case: fixing embedding credentials wakes ONLY f_emb, leaving MinerU-failed PDF untouched
            rev_emb = model_endpoints._revision(model_endpoints.endpoint_values(cfg)["embedding"])
            with patch.object(model_endpoints, "load_settings", return_value=cfg), \
                    patch.object(model_endpoints, "_probe", side_effect=lambda _r, c: c), \
                    patch.object(model_endpoints, "_check_embedding_change", return_value=None), \
                    patch.object(service, "kick_worker", side_effect=lambda: kicked.append(2)):
                model_endpoints.save_endpoint("embedding", {"api_key": "good-key", "revision": rev_emb})
            self.assertEqual(kicked, [2])
            with db.connect(state) as con:
                queued = {r[0] for r in con.execute("SELECT file_id FROM jobs WHERE status='queued'").fetchall()}
            self.assertEqual(queued, {db.file_id_for("kb1", 1)})

    def test_visual_embedding_401_classified_as_visual_embedding_config(self) -> None:
        resp = SimpleNamespace(status_code=401, text="Unauthorized")
        client = VisualEmbeddingClient(
            base_url="http://127.0.0.1:9002/v1",
            api_key="bad",
            model_id="vl-embed",
            dim=4,
            retry=5,
        )
        with patch.object(client.session, "post", return_value=resp) as post_mock:
            with self.assertRaises(VisualEmbedConfigError) as ctx:
                client._request([[{"role": "user", "content": "x"}]])
        self.assertEqual(post_mock.call_count, 1)
        self.assertEqual(worker._failure_verdict(ctx.exception), FAILURE_CONFIG)
        self.assertEqual(worker._failure_scope(ctx.exception), FAILURE_SCOPE_VISUAL_EMBEDDING)


class Item5WordFootnoteDedupTests(unittest.TestCase):
    @staticmethod
    def _make_docx(path: Path, body_xml: str, footnotes_xml: str) -> Path:
        w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("word/document.xml", f'<w:document xmlns:w="{w}"><w:body>{body_xml}</w:body></w:document>')
            zf.writestr("word/footnotes.xml", f'<w:footnotes xmlns:w="{w}">{footnotes_xml}</w:footnotes>')
        return path

    def test_unrelated_body_phrase_does_not_drop_footnote_but_real_duplicate_is_deduped(self) -> None:
        para = lambda t, m="": f"<w:p><w:r><w:t>{t}</w:t></w:r>{m}</w:p>"
        ref = lambda n: f'<w:r><w:footnoteReference w:id="{n}"/></w:r>'
        foot = lambda n, t: f'<w:footnote w:id="{n}">{para(t)}</w:footnote>'

        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_docx(
                Path(tmp) / "note.docx",
                para("第一段正文里刚好写了仅供参考。") + para("第二段引用的指标说明", ref(2)),
                foot(2, "仅供参考。"),
            )
            # 1) Recogniser did NOT deliver the footnote: phrase in paragraph 1 must NOT drop paragraph 2's footnote
            blocks_without_note = [
                ParsedBlock("mineru", "p", "docx", "text", "第一段正文里刚好写了仅供参考。", "b1", metadata={"section_path": ["一"]}),
                ParsedBlock("mineru", "p", "docx", "text", "第二段引用的指标说明", "b2", metadata={"section_path": ["一"]}),
            ]
            notes = docx_enhanced.extract_docx_note_blocks(path, blocks_without_note)
            self.assertEqual([n.text for n in notes], ["FOOTNOTE: 仅供参考。"])
            placed = docx_enhanced.attach_note_blocks(blocks_without_note, notes)
            self.assertEqual(placed[1].text, "第二段引用的指标说明\nFOOTNOTE: 仅供参考。")

            # 2) Recogniser DID deliver the footnote (appears a 2nd time in blocks): deduplicated
            blocks_with_note = [
                ParsedBlock("mineru", "p", "docx", "text", "第一段正文里刚好写了仅供参考。", "b1", metadata={"section_path": ["一"]}),
                ParsedBlock("mineru", "p", "docx", "text", "第二段引用的指标说明\n仅供参考。", "b2", metadata={"section_path": ["一"]}),
            ]
            self.assertEqual(docx_enhanced.extract_docx_note_blocks(path, blocks_with_note), [])

            # 3) Paragraph 1 missing from blocks, yet recogniser appended the footnote to paragraph 2: still deduplicated
            blocks_missing_p1 = [
                ParsedBlock("mineru", "p", "docx", "text", "第二段引用的指标说明\n仅供参考。", "b2", metadata={"section_path": ["一"]}),
            ]
            self.assertEqual(docx_enhanced.extract_docx_note_blocks(path, blocks_missing_p1), [])

            # 4) Adversarial H1/H2/H3: merged 800-token block (with section_spans), repeated table cell,
            #    VLM caption, chart, and an unrelated FOOTNOTE: line must NOT drop paragraph 2's footnote
            merged_text = "第一段正文里刚好写了仅供参考。\n\n第二段引用的指标说明"
            s1_end = len("第一段正文里刚好写了仅供参考。")
            s2_start = s1_end + 2
            merged_block = ParsedBlock(
                "mineru", "p", "docx", "text", merged_text, "m1",
                metadata={
                    "section_path": ["一"],
                    "section_spans": [
                        {"start": 0, "end": s1_end, "section_path": ["一"]},
                        {"start": s2_start, "end": len(merged_text), "section_path": ["一"]},
                    ],
                },
            )
            table_with_unrelated_fn = ParsedBlock(
                "mineru", "p", "docx", "table",
                "| 说明 |\n| --- |\n| 仅供参考。 |\n| 仅供参考。 |\nFOOTNOTE: 单位：mm",
                "t1",
                metadata={"section_path": ["一"]},
            )
            img_with_caption = ParsedBlock(
                "mineru", "p", "docx", "image", "VISUAL SUMMARY: 仅供参考。", "img1",
                visual_ref="/tmp/x.png", metadata={"section_path": ["一"]},
            )
            notes_adv = docx_enhanced.extract_docx_note_blocks(
                path, [merged_block, table_with_unrelated_fn, img_with_caption]
            )
            self.assertEqual([n.text for n in notes_adv], ["FOOTNOTE: 仅供参考。"])


class Patch12AdversarialRegressionTests(unittest.TestCase):
    def test_scanned_page_with_digital_footer_and_rotated_page_preserve_ocr_headings(self) -> None:
        import pymupdf
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            buf = io.BytesIO()
            Image.new("RGB", (400, 500), (235, 235, 235)).save(buf, format="PNG")
            img_bytes = buf.getvalue()

            # Scanned pages (standard 1-inch margins => w_f~0.76, h_f~0.82) + digital Bates footer across 12 pages
            pdf_path = Path(tmp) / "scanned_with_footer.pdf"
            doc = pymupdf.open()
            for i in range(12):
                pg = doc.new_page(width=600, height=800)
                pg.insert_image(pymupdf.Rect(72, 72, 528, 728), stream=img_bytes)
                pg.insert_text((250, 770), f"Page {i + 1} of 12", fontsize=10)
            doc.save(str(pdf_path))
            doc.close()

            content = [
                {"type": "text", "text": "Scanned Chapter 1", "text_level": 1, "page_idx": 0, "bbox": [140, 140, 420, 175]},
                {"type": "text", "text": "Scanned body text.", "page_idx": 0, "bbox": [140, 200, 480, 250]},
                {"type": "page_footer", "text": "Page 1 of 12", "page_idx": 0, "bbox": [400, 950, 600, 975]},
            ]
            with patch.object(mineru_pdf, "call_mineru_sync", return_value={"content_list": content}):
                blocks = mineru_pdf.mineru_pdf_blocks(mineru_url="http://x", path=pdf_path, cache_dir=Path(tmp) / "c")
            self.assertEqual(blocks[0].block_type, "title")
            self.assertEqual(blocks[1].metadata["section_path"], ["Scanned Chapter 1"])

    def test_two_column_and_centered_recurring_parent_headings_preserved(self) -> None:
        # 2-column wrap (col 1 bottom body -> col 2 top recurring parent heading) AND centered recurring parent heading
        items = []
        for pg in range(3):
            items += [
                {"type": "text", "text": f"Section {pg + 1}", "text_level": 1, "page_idx": pg, "bbox": [50, 50, 200, 75]},
                {"type": "text", "text": "Column 1 bottom prose.", "page_idx": pg, "bbox": [50, 820, 450, 880]},
                # Column 2 top recurring parent heading (x0=520 > 50, y0=80 < 820: NOT trailing!)
                {"type": "text", "text": "Col2 Steps", "text_level": 1, "page_idx": pg, "bbox": [520, 80, 680, 105]},
                {"type": "text", "text": f"Step {pg + 1}.1", "text_level": 1, "page_idx": pg, "bbox": [520, 120, 650, 145]},
                {"type": "text", "text": "Run action.", "page_idx": pg, "bbox": [520, 160, 900, 200]},
                # Centered recurring parent heading followed by left-aligned bold list item (mis_marked -> body!)
                {"type": "text", "text": "Checklist", "text_level": 1, "page_idx": pg, "bbox": [430, 300, 570, 325]},
                {"type": "text", "text": "- Check permissions", "text_level": 1, "page_idx": pg, "bbox": [60, 350, 300, 375]},
            ]
        with tempfile.TemporaryDirectory() as tmp, patch.object(mineru_pdf, "call_mineru_sync", return_value={"content_list": items}):
            blocks = mineru_pdf.mineru_pdf_blocks(mineru_url="http://x", path=Path(tmp) / "cols.pdf", cache_dir=Path(tmp) / "c")
        self.assertEqual(sum(b.block_type == "title" and b.text == "Col2 Steps" for b in blocks), 3)
        self.assertEqual(sum(b.block_type == "title" and b.text == "Checklist" for b in blocks), 3)

    def test_visual_embed_nan_is_retryable_while_dim_mismatch_is_config(self) -> None:
        nan_resp = SimpleNamespace(status_code=200, json=lambda: {"data": [{"index": 0, "embedding": [float("nan"), 0.0, 0.0, 0.0]}]})
        dim_resp = SimpleNamespace(status_code=200, json=lambda: {"data": [{"index": 0, "embedding": [1.0, 0.0]}]})
        client = VisualEmbeddingClient(
            base_url="http://127.0.0.1:9002/v1", api_key="", model_id="vl", dim=4, retry=2,
        )
        with patch.object(client.session, "post", return_value=nan_resp) as pm, patch("time.sleep"):
            with self.assertRaises(RuntimeError) as ctx:
                client._request([[{"role": "user", "content": "x"}]])
        self.assertNotIsInstance(ctx.exception, VisualEmbedConfigError)
        self.assertEqual(pm.call_count, 2)  # retried!
        self.assertEqual(worker._failure_verdict(ctx.exception), "retryable")

        with patch.object(client.session, "post", return_value=dim_resp) as pm2:
            with self.assertRaises(VisualEmbedConfigError):
                client._request([[{"role": "user", "content": "x"}]])
        self.assertEqual(pm2.call_count, 1)  # not retried!

    def test_graph_only_kb_update_and_unused_llm_do_not_wake_unrelated_config_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.db"
            db.init_db(state)
            cfg = Item4ScopedConfigFailureRequeueTests()._base_cfg(state, Path(tmp))
            with db.connect(state) as con:
                _insert_kb(con, "kb1")
                f1 = _source_file("kb1", 1, "mineru_err.pdf", "v1")
                _, j1 = scheduler.schedule_file(con, ingest_run_id="r1", file=f1)
                con.execute(
                    "UPDATE jobs SET status='failed', payload_json=?, finished_at=? WHERE job_id=?",
                    (json.dumps({FAILURE_VERDICT_KEY: FAILURE_CONFIG, FAILURE_SCOPE_KEY: FAILURE_SCOPE_MINERU}), db.now_ts(), j1),
                )
                con.commit()

            kicked: list[int] = []
            with patch.dict("os.environ", {"KB_RUNTIME": ""}), \
                    patch.object(service, "settings", return_value=cfg), \
                    patch.object(service, "chunk_limits", return_value={"max_tokens_min": 100, "max_tokens_cap": 2000, "effective_max_model_len": 8192}), \
                    patch.object(service, "kick_worker", side_effect=lambda: kicked.append(1)):
                service.update_kb_config("kb1", {"graph_enabled": False})
            with patch.object(service, "settings", return_value=cfg), \
                    patch.object(service, "_probe_llm", return_value=None), \
                    patch.object(service, "kick_worker", side_effect=lambda: kicked.append(2)):
                service.save_llm({
                    "name": "unused-vlm",
                    "base_url": "http://127.0.0.1:8000/v1",
                    "model_id": "vl",
                    "protocol": "openai",
                    "multimodal": True,
                    "concurrency": 2,
                })
            self.assertEqual(kicked, [])
            with db.connect(state) as con:
                self.assertEqual(con.execute("SELECT COUNT(*) FROM jobs WHERE status='queued'").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
