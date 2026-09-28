from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import sqlite3

from .. import db
from .. import search_fts
from ..chunking.chunker import blocks_to_chunks, embedding_context_for, embedding_input
from ..chunking.diagnose import chunk_diagnostics, summarize_line
from ..config import Settings
from ..embedding.client import EmbeddingClient
from ..embedding.visual import VisualEmbeddingClient
from ..models import ParsedBlock, UnifiedChunk
from ..models import KBSource
from ..parsers.common import parser_profile_for_path
from ..parsers.docx_enhanced import parse_docx_enhanced
from ..parsers.html_dom import parse_html_dom
from ..parsers.image_file import parse_image_file
from ..parsers.native_table import parse_native_table
from ..parsers.pdf_enhanced import parse_pdf_enhanced
from ..parsers.pptx_enhanced import parse_pptx_enhanced
from ..parsers.router import TEXT_FILENAMES, TEXT_SUFFIXES, parse_native
from ..parsers.table_check import table_ambiguity_summary
from ..parsers.visual_blocks import vlm_failure_summary, went_through_vlm
from ..utils import count_tokens, detect_lang, dir_ancestors_from_rel_path, infer_doc_type
from ..vector.layout import TEXT_VECTOR, VISUAL_VECTOR
from ..vector.qdrant import client as qdrant_client
from ..vector.qdrant import collection_exists, ensure_collection
from ..vector.qdrant import mark_old_versions_inactive, mark_stale_file_points_inactive, upsert_chunks
from ..vector.qdrant import validate_collection_layout
from ..vision.images import IMAGE_SUFFIXES
from ..vision.vlm import prompt_with_language


TABLE_SUFFIXES = {".xlsx", ".xls", ".csv"}


# The definition moved to parsers.errors (the router raises it too); the name is kept here so the worker and
# the tests keep importing it from this module
from ..parsers.errors import NonRetryableParseError  # noqa: E402,F401



class JobCancelled(RuntimeError):
    """The user closed or deleted this knowledge base mid-parse. Not a failure: no retry is counted, no
    failure is recorded, the worker marks the job cancelled directly. The checkpoint is stage() -- every
    phase boundary and every per-image VLM callback passes through it, so the worst-case wait is "the
    current step", not "the current knowledge base"."""


def process_parse_job(con: sqlite3.Connection, settings: Settings, job: sqlite3.Row) -> str:
    file_row = db.get_file_by_id(con, str(job["file_id"]))
    if file_row is None:
        raise NonRetryableParseError(f"file row not found for {job['file_id']}")

    started = time.time()
    path = Path(str(file_row["physical_path"]))
    if not path.exists():
        raise NonRetryableParseError(f"physical file not found: {path}")

    max_bytes = settings.max_file_bytes
    if max_bytes and int(file_row["size"] or 0) > max_bytes:
        # A few-hundred-MB .json/.log/.csv dropped into a knowledge base by mistake makes the worker eat
        # several GB of memory, spend tens of minutes in tiktoken, or simply get killed by the OOM killer --
        # and the latter turns into a crash loop.
        raise NonRetryableParseError(
            f"File exceeds the single-file limit ({int(file_row['size'])} bytes > {max_bytes}) and was skipped; "
            "raise KB_MAX_FILE_BYTES if it must be ingested"
        )

    source = _source_for_file(settings, str(file_row["kb_id"]))
    verify_source_file(settings, source, path)
    parser_profile = str(job["parser_profile"] or _profile_for_path(path))
    content_version = str(file_row["content_version"])
    cache_dir = parse_cache_dir(settings, str(file_row["kb_id"]), int(file_row["file_key"]), content_version)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[parse] start job={job['job_id']} file={file_row['source_path']} size={file_row['size']} profile={parser_profile}", flush=True)

    def event(text: str) -> None:
        # Timeline milestone (console job details); failing to record it must not take the parse down
        try:
            db.add_job_event(con, str(job["job_id"]), "info", text)
        except Exception:
            pass

    event(f"Parsing started, size={file_row['size']}, profile={parser_profile}")

    last_stage_write = [0.0]
    last_cancel_check = [0.0]

    def check_cancelled(force: bool = False) -> None:
        # A failed read (lock contention etc.) must not take the parse down: cancellation is best-effort, the
        # next checkpoint looks again.
        now = time.time()
        if not force and now - last_cancel_check[0] < 1.5:
            return
        last_cancel_check[0] = now
        try:
            owner = job["locked_by"] if "locked_by" in job.keys() else None
            cancelled = db.job_cancel_requested(con, str(job["job_id"]), str(owner) if owner else None)
        except Exception:
            return
        if cancelled:
            raise JobCancelled(f"job {job['job_id']} cancelled by operator")

    def stage(text: str, *, force: bool = False) -> None:
        # Progress checkpoint for the web console; failures must never take
        # a parse down with them.
        # Every checkpoint is an UPDATE+COMMIT. The VLM callback fires once per image (concurrency 8), which
        # can mean several write locks per second contending with the console's polling reads -- rate-limited
        # to 1.5 seconds.
        check_cancelled(force)
        now = time.time()
        if not force and now - last_stage_write[0] < 1.5:
            return
        last_stage_write[0] = now
        try:
            db.set_job_stage(con, str(job["job_id"]), text)
        except Exception:
            pass

    stage("Parsing document", force=True)
    blocks = _parse_blocks(settings, source, path, cache_dir, parser_profile, file_row, stage_cb=stage)
    print(f"[parse] blocks={len(blocks)} file={file_row['source_path']}", flush=True)
    event(f"Parsed into {len(blocks)} blocks")
    table_stats = table_ambiguity_summary(blocks)
    if table_stats["flagged"]:
        print(f"[parse] table_ambiguous={table_stats['flagged']} verified={table_stats['verified']} "
              f"file={file_row['source_path']}", flush=True)
        event(f"Merged cells in {table_stats['flagged']} tables: {table_stats['verified']} split after screenshot check, "
              f"{table_stats['unverified']} unverified (unverified values are excluded from trusted parameters)")
    if not blocks:
        if _is_intentionally_empty(path):
            return _finish_empty_document(con, settings, job, file_row, content_version, parser_profile)
        raise NonRetryableParseError("parser returned no blocks")

    # When the VLM is down, each image's exception is swallowed per image and the document is still stored
    # as "parsed successfully", while content_version is unchanged => later scans judge it unchanged and the
    # missing image descriptions are never filled in automatically. Once the failure ratio exceeds the
    # threshold, the whole job takes the normal retry path (images that succeeded are cached, so a retry
    # does not burn the GPU again).
    vlm_stats = vlm_failure_summary(blocks)
    if vlm_stats["vlm_failed"]:
        ratio = vlm_stats["vlm_failed"] / max(1, vlm_stats["visual_blocks"])
        print(
            f"[parse] vlm_failed={vlm_stats['vlm_failed']}/{vlm_stats['visual_blocks']} "
            f"file={file_row['source_path']}",
            flush=True,
        )
        if ratio >= settings.vlm_failure_retry_ratio:
            raise RuntimeError(
                f"VLM description failed for {vlm_stats['vlm_failed']}/{vlm_stats['visual_blocks']} images; "
                "this parse was not stored so that no document is left without image descriptions"
            )

    stage("Chunking", force=True)
    chunks = blocks_to_chunks(
        kb_id=str(file_row["kb_id"]),
        file_key=int(file_row["file_key"]),
        content_version=content_version,
        parser_profile=parser_profile,
        blocks=blocks,
        max_tokens=source.max_tokens,
        overlap_tokens=source.overlap_tokens,
    )
    print(f"[parse] chunks={len(chunks)} file={file_row['source_path']}", flush=True)
    if not chunks:
        if _is_intentionally_empty(path):
            return _finish_empty_document(con, settings, job, file_row, content_version, parser_profile)
        raise NonRetryableParseError("chunker returned no chunks")
    # Chunking acceptance check: advisory only, never blocking; the verdict is stored with the files row for
    # the console
    chunk_diag = chunk_diagnostics(chunks, max_tokens=source.max_tokens, blocks=blocks)
    print(f"[parse] chunk-check {summarize_line(chunk_diag)} file={file_row['source_path']}", flush=True)
    event(f"{len(chunks)} chunks · " + ("check passed" if chunk_diag["ok"] else "chunking notes: " + "; ".join(r["message"] for r in chunk_diag["reasons"])))

    stage(f"Text embedding ({len(chunks)} blocks)")
    embedder = EmbeddingClient(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        model_id=settings.embedding_model_id,
        dim=settings.embedding_dim,
        batch_size=settings.embedding_batch,
        retry=settings.embedding_retry,
        sleep_seconds=settings.embedding_sleep_seconds,
    )
    # Embedding prefix: document name + section path go into the vector, the text stays unchanged. Short chunks
    # rely on it to carry context and get hit (plan batch 1, item 4)
    doc_name = str(file_row["rel_path"] or file_row["filename"] or "").rsplit(".", 1)[0].replace("/", " > ")
    for chunk in chunks:
        chunk.embedding_context = embedding_context_for(chunk, doc_name=doc_name)
    text_vectors = embedder.embed([embedding_input(chunk) for chunk in chunks])
    print(f"[parse] embeddings={len(text_vectors)} batches={(len(chunks) + settings.embedding_batch - 1) // settings.embedding_batch}", flush=True)
    event(f"{len(text_vectors)} text vectors")

    # Second leg of the serial visual path: every picture the VLM described
    # now also gets a pixel-level vector. Runs after the description pass and
    # the text embeddings, one vector per picture (the first chunk of a block).
    stage("Visual embedding", force=True)
    visual_vectors = _embed_visual_chunks(settings, chunks, cache_dir)
    named_vectors: list[dict[str, list[float]]] = []
    for chunk, text_vector in zip(chunks, text_vectors, strict=True):
        vector: dict[str, list[float]] = {TEXT_VECTOR: text_vector}
        visual = visual_vectors.get(chunk.chunk_uid)
        if visual is not None:
            vector[VISUAL_VECTOR] = visual
        named_vectors.append(vector)

    q = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
    point_ids = [_point_id(chunk.chunk_uid) for chunk in chunks]
    # The file row is gone = this KB (or this file) was deleted mid-parse. This used to fall back to the old
    # in-memory row and keep writing, and when the collection was already dropped, ensure_collection
    # recreated it, leaving an unclaimed zombie collection. Just stop.
    write_row = db.get_file_by_id(con, str(job["file_id"]))
    if write_row is None:
        print(
            f"[parse] gone-skip job={job['job_id']} file={file_row['source_path']}; "
            "file row disappeared mid-parse (kb deleted?)",
            flush=True,
        )
        return f"parse-gone-skipped:{job['job_id']}"
    if str(write_row["content_version"]) != content_version:
        print(
            f"[parse] stale-skip job={job['job_id']} parsed_version={content_version} "
            f"current_version={write_row['content_version']} file={write_row['source_path']}",
            flush=True,
        )
        return f"parse-stale-skipped:{job['job_id']}"
    if str(write_row["status"]) == "deleted":
        print(
            f"[parse] deleted-skip job={job['job_id']} parsed_version={content_version} "
            f"file={write_row['source_path']}",
            flush=True,
        )
        return f"parse-deleted-skipped:{job['job_id']}"
    if str(write_row["source_path"]) != str(file_row["source_path"]):
        print(
            f"[parse] metadata refreshed before upsert old_path={file_row['source_path']} "
            f"new_path={write_row['source_path']}",
            flush=True,
        )
    stage("Writing vectors", force=True)
    collection = str(write_row["collection"])
    if not collection_exists(q, collection):
        print(f"[qdrant] collection missing; creating collection={collection}", flush=True)
        print(ensure_collection(q, collection, settings.vector_layout), flush=True)
    else:
        validate_collection_layout(q, collection, settings.vector_layout)

    payloads = [
        _payload_for_chunk(
            settings,
            write_row,
            chunk,
            len(chunks),
            visual_model=settings.visual_embedding_model_id if chunk.chunk_uid in visual_vectors else None,
        )
        for chunk in chunks
    ]
    upsert_chunks(
        q,
        collection,
        point_ids,
        named_vectors,
        payloads,
        max_bytes=settings.qdrant_upsert_max_bytes,
    )
    mark_old_versions_inactive(
        q,
        collection,
        kb_id=str(write_row["kb_id"]),
        file_key=int(write_row["file_key"]),
        active_content_version=content_version,
    )
    stale_points = mark_stale_file_points_inactive(
        q,
        collection,
        kb_id=str(write_row["kb_id"]),
        file_key=int(write_row["file_key"]),
        active_chunk_uids={chunk.chunk_uid for chunk in chunks},
    )
    if stale_points:
        print(f"[parse] inactive stale qdrant points={stale_points} file={file_row['source_path']}", flush=True)


    stage("Keyword indexing")
    # The keyword index is retrieval's second leg, not a precondition for indexing. The Qdrant points are
    # already written at this point: letting an OpenSearch fault fail the whole job would only rerun the full
    # MinerU + embedding 5 times and still leave the "Qdrant has the new version, SQLite records the old one"
    # inconsistency. Degrade to one failures record + a separate (idempotent) fts_sync job.
    try:
        fts_result = search_fts.sync_doc_from_qdrant(
            url=settings.opensearch_url,
            qdrant=q,
            collection=collection,
            doc_id=f"{write_row['kb_id']}:{write_row['file_key']}",
        )
        print(
            f"[fts] sync-doc collection={collection} doc_id={fts_result['doc_id']} "
            f"deleted={fts_result.get('deleted_rows')} inserted={fts_result.get('inserted_rows')}",
            flush=True,
        )
    except Exception as exc:
        print(f"[fts] sync deferred collection={collection} error={exc!r}", flush=True)
        event(f"Keyword indexing failed, a retry task was queued: {exc!r}"[:600])
        # The degraded branch must assign fts_result too: the return string at the end of the function refers
        # to it, and leaving it out means a NameError, with the job recorded as failed and retried 5 times --
        # while Qdrant and SQLite were actually both written (2026-09-06 health check B1)
        fts_result = {"inserted_rows": 0, "deferred": True}
        _defer_fts_sync(con, job, write_row, collection=collection, content_version=content_version, exc=exc)
    stage("Done", force=True)
    db.replace_chunks(
        con,
        file_id=str(write_row["file_id"]),
        collection=str(write_row["collection"]),
        content_version=content_version,
        chunks=chunks,
        point_ids=point_ids,
    )
    db.mark_file_indexed(con, str(write_row["file_id"]), content_version, parser_profile, chunk_diag=chunk_diag)
    elapsed = int(time.time() - started)
    print(
        f"[parse] done job={job['job_id']} chunks={len(chunks)} points={len(point_ids)} "
        f"visual_vectors={len(visual_vectors)} elapsed={elapsed}s",
        flush=True,
    )
    event(f"Stored {len(point_ids)} points ({len(visual_vectors)} visual vectors), took {elapsed}s")
    return (
        f"parse-done:{job['job_id']}:chunks={len(chunks)}:points={len(point_ids)}"
        f":visual={len(visual_vectors)}:fts={fts_result['inserted_rows']}:elapsed={elapsed}s"
    )


def select_visual_chunks(chunks: list[UnifiedChunk]) -> list[UnifiedChunk]:
    """Chunks that should carry the visual vector: the first chunk of every
    visual block whose picture actually went through the VLM. One picture,
    one vector -- a slide split into several text chunks must not show up N
    times in a visual search."""
    selected: list[UnifiedChunk] = []
    seen_blocks: set[str] = set()
    for chunk in chunks:
        block = chunk.block
        if block.block_type not in VISUAL_BLOCK_TYPES or not block.visual_ref:
            continue
        if block.block_id in seen_blocks:
            continue
        seen_blocks.add(block.block_id)
        if not went_through_vlm(block):
            continue
        if not Path(block.visual_ref).exists():
            # enrich_blocks_with_vlm verified this file moments ago; losing it
            # here means the parse cache is being mutated under us. Failing
            # loudly beats indexing a point whose payload claims visual
            # provenance it does not have.
            raise RuntimeError(f"visual crop vanished during parse: {block.visual_ref}")
        selected.append(chunk)
    return selected


def _embed_visual_chunks(settings: Settings, chunks: list[UnifiedChunk], cache_dir: Path) -> dict[str, list[float]]:
    targets = select_visual_chunks(chunks)
    if not targets:
        return {}
    if not settings.visual_embedding_enabled:
        print(f"[visual-embed] disabled; skipping images={len(targets)}", flush=True)
        return {}
    client = VisualEmbeddingClient(
        base_url=settings.visual_embedding_base_url,
        api_key=settings.visual_embedding_api_key,
        model_id=settings.visual_embedding_model_id,
        dim=settings.visual_embedding_dim,
        instruction=settings.visual_embedding_instruction,
        concurrency=settings.visual_embedding_concurrency,
        retry=settings.visual_embedding_retry,
        timeout=settings.visual_embedding_timeout_seconds,
        max_pixels=settings.image_max_pixels,
    )
    jobs: list[tuple[Path, Path | None]] = []
    for chunk in targets:
        image_path = Path(str(chunk.block.visual_ref))
        sha = str(chunk.block.metadata.get("visual_sha256") or "")
        # Keyed by the picture's own hash, not the parse location: the cache
        # identity inside the file already pins model/dim/instruction, so an
        # edited document (new content_version) or the same figure reused in
        # another document hits the cache instead of re-embedding.
        if sha:
            cache_json = settings.cache_dir / "vlm-cache" / sha[:2] / f"{sha}.embed.json"
        else:
            cache_json = cache_dir / "visual" / f"{chunk.block.block_id}-nohash.embed.json"
        jobs.append((image_path, cache_json))
    vectors = client.embed_images(jobs)
    result: dict[str, list[float]] = {}
    for chunk, vector in zip(targets, vectors, strict=True):
        if vector is None:
            chunk.block.metadata["visual_embed_status"] = "content_rejected"
            continue
        result[chunk.chunk_uid] = vector
    return result


def _is_intentionally_empty(path: Path) -> bool:
    if path.suffix.lower() not in {*TEXT_SUFFIXES, ".csv"} and path.name not in TEXT_FILENAMES:
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size == 0:
        return True          # a truly empty file need not be read
    if size > 1_000_000:
        return False         # above one MB it cannot be "whitespace only"; do not read it whole
    return not path.read_text(encoding="utf-8", errors="ignore").strip()



def _defer_fts_sync(con, job, write_row, *, collection: str, content_version: str, exc: Exception) -> str:
    """The keyword index could not be written (OpenSearch fault): queue a separate fts_sync compensation job,
    with the failure record attached to that compensation job.

    The compensation job used to borrow the metadata_update type with an fts-retry:… key, while the scan's
    failed-job recovery only recognizes metadata_update:<file>:<fingerprint> keys -- so once retries were
    exhausted it was never queued again; the failure record also hung on the main parse job and was marked
    resolved as soon as the main job finished: healthy UI, missing index (Codex review F06)."""
    fts_job_id = db.enqueue_job(
        con,
        ingest_run_id=str(job["ingest_run_id"] or "") or None,
        file_id=str(write_row["file_id"]),
        kb_id=str(write_row["kb_id"]),
        collection=collection,
        file_key=int(write_row["file_key"]),
        job_type="fts_sync",
        priority=60,
        payload={"reason": "fts sync retry after OpenSearch failure", "content_version": content_version},
        dedupe_key=db.fts_sync_dedupe_key(str(write_row["file_id"]), content_version),
    )
    db.add_failure(
        con,
        file_id=str(write_row["file_id"]),
        job_id=fts_job_id,
        stage="fts-sync",
        error_type=type(exc).__name__,
        error_message=str(exc),
    )
    return fts_job_id

def _finish_empty_document(
    con: sqlite3.Connection,
    settings: Settings,
    job: sqlite3.Row,
    file_row: sqlite3.Row,
    content_version: str,
    parser_profile: str,
) -> str:
    write_row = db.get_file_by_id(con, str(job["file_id"]))
    if write_row is None:
        return f"parse-gone-skipped:{job['job_id']}"
    if str(write_row["content_version"]) != content_version:
        return f"parse-stale-skipped:{job['job_id']}"
    if str(write_row["status"]) == "deleted":
        return f"parse-deleted-skipped:{job['job_id']}"

    q = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
    collection = str(write_row["collection"])
    stale_points = 0
    if collection_exists(q, collection):
        stale_points = mark_stale_file_points_inactive(
            q,
            collection,
            kb_id=str(write_row["kb_id"]),
            file_key=int(write_row["file_key"]),
            active_chunk_uids=set(),
        )
    try:
        fts_result = search_fts.sync_doc_from_qdrant(
            url=settings.opensearch_url,
            qdrant=q,
            collection=collection,
            doc_id=f"{write_row['kb_id']}:{write_row['file_key']}",
        )
    except Exception as exc:
        # Same degradation as the main path: the keyword index is not a precondition for indexing; on an
        # OpenSearch fault record one failures row + queue an idempotent retry job, and the empty document is
        # still marked indexed (2026-09-06 health check B5)
        print(f"[fts] sync deferred (empty document) collection={collection} error={exc!r}", flush=True)
        _defer_fts_sync(con, job, write_row, collection=collection, content_version=content_version, exc=exc)
        fts_result = {"inserted_rows": 0, "deferred": True}
    db.mark_chunks_inactive(con, str(write_row["file_id"]))
    db.mark_file_indexed(con, str(write_row["file_id"]), content_version, parser_profile)
    print(
        f"[parse] empty document indexed file={write_row['source_path']} "
        f"inactive_points={stale_points} fts_rows={fts_result['inserted_rows']}",
        flush=True,
    )
    return f"parse-empty:{job['job_id']}:inactive_points={stale_points}"


def vlm_progress_callback(stage_cb):
    """VLM progress → stage label "Describing images (VLM done/total)". The last image is written forcibly:
    the 1.5-second checkpoint rate limit would swallow it and the timeline would stop at "165/166"; when
    stage_cb does not accept the force parameter (other callers such as the preview), fall back to a plain
    checkpoint."""
    if stage_cb is None:
        return None

    def cb(done: int, total: int) -> None:
        text = f"Describing images (VLM {done}/{total})"
        if done >= total:
            try:
                stage_cb(text, force=True)
                return
            except TypeError:
                pass
        stage_cb(text)
    return cb


def _parse_blocks(
    settings: Settings,
    source: KBSource,
    path: Path,
    cache_dir: Path,
    parser_profile: str,
    file_row: sqlite3.Row,
    stage_cb=None,
) -> list[ParsedBlock]:
    suffix = path.suffix.lower()

    vlm_progress = vlm_progress_callback(stage_cb)
    vlm_options = {
        "temperature": settings.vlm_temperature,
        "top_p": settings.vlm_top_p,
        "max_tokens": settings.vlm_max_tokens,
        "structured": settings.vlm_structured_output,
        "max_pixels": settings.image_max_pixels,
    }
    caption_cache_root = settings.cache_dir / "vlm-cache"
    # When the KB's output language (from label extraction) is not Chinese, image descriptions are written in
    # that language too; Chinese / never extracted leaves the prompt unchanged
    vlm_prompt = prompt_with_language(source.vlm_prompt, source.graph_language)
    if suffix in IMAGE_SUFFIXES:
        return parse_image_file(
            path=path,
            cache_dir=cache_dir,
            vlm_base_url=settings.vlm_base_url,
            vlm_api_key=settings.vlm_api_key,
            vlm_model_id=settings.vlm_model_id,
            vlm_concurrency=settings.vlm_concurrency,
            vlm_options=vlm_options,
            caption_cache_root=caption_cache_root,
            vlm_prompt=vlm_prompt,
            progress_cb=vlm_progress,
        )
    if suffix == ".pdf":
        return parse_pdf_enhanced(
            mineru_url=settings.mineru_url,
            vlm_base_url=settings.vlm_base_url,
            vlm_api_key=settings.vlm_api_key,
            vlm_model_id=settings.vlm_model_id,
            vlm_concurrency=settings.vlm_concurrency,
            path=path,
            cache_dir=cache_dir,
            merge_target_tokens=source.block_merge_tokens,
            timeout=settings.mineru_timeout_seconds,
            vlm_options=vlm_options,
            caption_cache_root=caption_cache_root,
            vlm_prompt=vlm_prompt,
            progress_cb=vlm_progress,
        )
    if suffix == ".pptx":
        return parse_pptx_enhanced(
            mineru_url=settings.mineru_url,
            vlm_base_url=settings.vlm_base_url,
            vlm_api_key=settings.vlm_api_key,
            vlm_model_id=settings.vlm_model_id,
            vlm_concurrency=settings.vlm_concurrency,
            path=path,
            cache_dir=cache_dir,
            timeout=settings.mineru_timeout_seconds,
            vlm_options=vlm_options,
            caption_cache_root=caption_cache_root,
            vlm_prompt=vlm_prompt,
            progress_cb=vlm_progress,
        )
    if suffix == ".docx":
        return parse_docx_enhanced(
            mineru_url=settings.mineru_url,
            vlm_base_url=settings.vlm_base_url,
            vlm_api_key=settings.vlm_api_key,
            vlm_model_id=settings.vlm_model_id,
            vlm_concurrency=settings.vlm_concurrency,
            path=path,
            cache_dir=cache_dir,
            merge_target_tokens=source.block_merge_tokens,
            timeout=settings.mineru_timeout_seconds,
            vlm_options=vlm_options,
            caption_cache_root=caption_cache_root,
            vlm_prompt=vlm_prompt,
            progress_cb=vlm_progress,
        )
    if suffix == ".doc":
        raise NonRetryableParseError("legacy .doc files are not supported")
    if suffix == ".html":
        print(f"[parser] html dom start file={path.name}", flush=True)
        blocks = parse_html_dom(path)
        print(f"[parser] html dom done blocks={len(blocks)} file={path.name}", flush=True)
        return blocks
    if suffix in TABLE_SUFFIXES:
        return parse_native_table(path, max_tokens=source.max_tokens, overlap_tokens=source.overlap_tokens)
    return parse_native(path, "")


def verify_source_file(settings: Settings, source, path: Path) -> None:
    """Re-check, at read time, what the scanner checked at scan time: the knowledge base directory is still
    admitted by the link policy, and the file's real location is still inside that directory. Between the
    scan and the parse the directory may have been swapped for a symbolic link, or the file itself replaced
    by one (security review F04 follow-up). The boundary is the admitted root, resolved only after the
    policy check, never the target of whatever the root points at now."""
    from ..discovery import directory_admitted
    from ..localfs.scanner import inside_boundary

    admitted, why = directory_admitted(Path(settings.mirror_root), str(source.source_root))
    if not admitted:
        if why == "linked":
            raise NonRetryableParseError(
                f"Knowledge base directory {source.source_root!r} is a symbolic link; "
                "set KB_MIRROR_ALLOW_LINKED_DIRS=1 to read linked directories"
            )
        raise NonRetryableParseError(f"Knowledge base directory {source.source_root!r} is missing")
    root = (Path(settings.mirror_root) / str(source.source_root)).resolve()
    if not inside_boundary(path, root):
        raise NonRetryableParseError(f"File {path} is outside the enrolled directory (symbolic link); not parsed")


def _source_for_file(settings: Settings, kb_id: str):
    for source in settings.sources.values():
        if source.kb_id == kb_id:
            return source
    raise RuntimeError(f"unknown kb_id={kb_id}")


def _profile_for_path(path: Path) -> str:
    return parser_profile_for_path(path)


def _point_id(chunk_uid: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"qdrant-point:{chunk_uid}"))


VISUAL_BLOCK_TYPES = {"image", "chart", "slide"}


def _payload_for_chunk(
    settings: Settings,
    row: sqlite3.Row,
    chunk: UnifiedChunk,
    chunk_total: int,
    *,
    visual_model: str | None = None,
) -> dict[str, Any]:
    block = chunk.block
    section_path = [str(x) for x in (block.metadata.get("section_path") or []) if str(x).strip()]
    payload: dict[str, Any] = {
        # identity
        "kb_id": row["kb_id"],
        "doc_id": f"{row['kb_id']}:{row['file_key']}",
        "chunk_uid": chunk.chunk_uid,
        "chunk_index": chunk.chunk_index,
        "chunk_total": chunk_total,
        "block_id": block.block_id,
        # location — source_path / rel_path / dir all feed metadata-exact routing
        "source_path": row["source_path"],
        "rel_path": row["rel_path"],
        "filename": row["filename"],
        "dir": row["dir"],
        "dir_ancestors": dir_ancestors_from_rel_path(str(row["rel_path"])),
        "page_idx": block.page_idx,
        # Page range of tables / passages merged across pages (recorded in metadata by pdf_enhanced): fact
        # provenance and the UI are no longer limited to pointing at the start page (Codex review F09)
        "page_start": block.metadata.get("page_start", block.page_idx) if block.page_idx is not None else block.metadata.get("page_start"),
        "page_end": block.metadata.get("page_end", block.page_idx) if block.page_idx is not None else block.metadata.get("page_end"),
        "slide_idx": block.slide_idx,
        "sheet_name": block.sheet_name,
        "row_start": block.row_start,
        "row_end": block.row_end,
        "bbox": block.bbox,
        # structure
        "doc_type": block.doc_type or infer_doc_type(str(row["filename"]), str(row["mime_type"])),
        "block_type": block.block_type,
        "section_path": section_path,
        "section_depth": len(section_path),
        "embedding_context": getattr(chunk, "embedding_context", None),
        "title": block.title,
        "caption": block.caption,
        "visual_value_conflicts": block.metadata.get("visual_value_conflicts") or None,
        # Metadata from structured sources (code symbols / markdown frontmatter), used by rule-based graph
        # extraction and by retrieval
        "symbol": block.metadata.get("symbol"),
        "frontmatter": block.metadata.get("frontmatter"),
        # Table merged-cell flags and screenshot verification result (F02): fact extraction marks facts that took
        # an unverified merged value as untrusted, and the console shows a badge
        "table_flags": block.metadata.get("table_flags") or None,
        "table_repair": block.metadata.get("table_repair") or None,
        # content
        "text": chunk.text,
        "token_count": chunk.token_count or count_tokens(chunk.text),
        "lang": detect_lang(chunk.text),
        # provenance
        "content_version": row["content_version"],
        "file_size": int(row["size"]),
        "file_mtime": int(row["mtime"]),
        "parser_profile": block.parser_profile,
        "embedding_model": settings.embedding_model_id,
        "indexed_at": db.now_ts(),
        # lifecycle
        "is_active": True,
    }
    # Only genuinely visual blocks carry a visual_ref in the payload. MinerU
    # also hands back a crop for every table and equation, and the parsers keep
    # it on the block for provenance -- but a table's evidence is its HTML and
    # an equation's is its LaTeX, and neither goes through the VLM. Writing
    # visual_ref for them mislabels embedding_text_source as "visual_ref" and,
    # worse, would let any future visual-index pass that selects by visual_ref
    # sweep table crops into an image collection.
    if block.visual_ref and block.block_type in VISUAL_BLOCK_TYPES:
        payload["visual_ref"] = _relative_visual_ref(settings, block.visual_ref)
        if block.visual_summary and block.text:
            payload["embedding_text_source"] = "text_plus_visual_summary"
        elif block.visual_summary:
            payload["embedding_text_source"] = "visual_summary"
        else:
            payload["embedding_text_source"] = "visual_ref"
        # sha256 of the picture itself: lets consumers spot the same figure
        # reused across documents, independent of where the crop landed.
        if block.metadata.get("visual_sha256"):
            payload["visual_sha256"] = block.metadata["visual_sha256"]
        # Present only on the point that carries the `visual` named vector
        # (the first chunk of the block); doubles as the has-visual-vector flag.
        if visual_model:
            payload["visual_embedding_model"] = visual_model
    if block.visual_summary:
        payload["visual_summary"] = block.visual_summary
    for key in (
        "visual_entities",
        "visual_facts",
        "visual_keywords",
        "visual_text",
        "visual_kind",
        "visual_confidence",
        "vlm_model",
    ):
        if block.metadata.get(key):
            payload[key] = block.metadata[key]
    return {key: value for key, value in payload.items() if value is not None and value != ""}


def _relative_visual_ref(settings: Settings, visual_ref: str) -> str:
    """Store the parse-cache asset as a path relative to the cache root.

    Absolute paths bake the ingest machine into the payload -- the same
    portability flaw local_path had. Consumers join this key onto whatever
    cache root (or URL prefix) is right for them.
    """
    try:
        return str(Path(visual_ref).resolve().relative_to(settings.cache_dir.resolve()))
    except ValueError:
        return visual_ref


def parse_cache_dir(settings: Settings, kb_id: str, file_key: int, content_version: str) -> Path:
    """Cache directory for MinerU / VLM results, isolated by KB / file / content version. The chunk preview reads
    from here too."""
    return settings.cache_dir / "parse" / kb_id / str(file_key) / _safe_version(content_version)


def _safe_version(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)[:120]
