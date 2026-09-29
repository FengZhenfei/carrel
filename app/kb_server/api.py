from __future__ import annotations

import hashlib
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response

from . import service

router = APIRouter(prefix="/api")


def _wrap(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"not found: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/overview")
def overview() -> dict[str, Any]:
    return service.overview()


@router.get("/health")
def health() -> dict[str, Any]:
    return service.health()


_limits_cache: dict[str, Any] = {"at": 0.0, "value": None}


@router.get("/limits")
def limits() -> dict[str, Any]:
    # Every call would do a full load_settings + one probe of the embedding /models endpoint (5 s
    # timeout). The limits only change when the model changes, so a 60-second cache is enough.
    import time

    from kb_pipeline.limits import chunk_limits

    now = time.time()
    if _limits_cache["value"] is not None and now - float(_limits_cache["at"]) < 60.0:
        return dict(_limits_cache["value"])
    value = chunk_limits(service.settings().embedding_base_url)
    _limits_cache.update({"at": now, "value": value})
    return dict(value)


@router.post("/enroll")
def enroll(payload: dict[str, Any]) -> dict[str, Any]:
    name = str(payload.get("dir") or "").strip()
    if not name:
        raise HTTPException(status_code=422, detail="dir is required")
    config = payload.get("config")
    if isinstance(config, dict):
        _reject_bad_config(config)
    return _wrap(service.enroll, name, config if isinstance(config, dict) else None)


@router.post("/kbs/{kb_id}/adopt")
def adopt_kb(kb_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Recognise an unenrolled directory as the renamed form of this knowledge base (whose directory has
    disappeared): the id and data are kept and nothing is re-parsed."""
    return _wrap(service.adopt_kb, kb_id, str(payload.get("dir") or ""))


@router.delete("/kbs/{kb_id}")
def delete_kb(kb_id: str) -> dict[str, Any]:
    return _wrap(service.delete_kb, kb_id)


@router.post("/kbs/{kb_id}/graph_build")
def graph_build(kb_id: str) -> dict[str, Any]:
    return _wrap(service.trigger_graph_build, kb_id)


@router.post("/kbs/{kb_id}/graph_pause")
def graph_pause(kb_id: str) -> dict[str, Any]:
    return _wrap(service.pause_graph_build, kb_id)


@router.post("/kbs/{kb_id}/graph_append")
def graph_append(kb_id: str) -> dict[str, Any]:
    """"Merge new content": extract units only from new / changed documents, replay the previous
    version's merge decisions, reuse unchanged vectors, and switch over as a new version."""
    return _wrap(service.trigger_graph_append, kb_id)


@router.get("/kbs/{kb_id}/graph_corpus")
def graph_corpus(kb_id: str) -> dict[str, Any]:
    """Size of the ready documents (document / chunk / token counts). The console fetches it once when
    the knowledge graph is switched on."""
    return _wrap(service.graph_corpus_stats, kb_id)


@router.post("/kbs/{kb_id}/graph_schema")
def graph_schema(kb_id: str) -> dict[str, Any]:
    """"Extract / re-extract labels now": synchronously runs a full label extraction (several LLM calls,
    usually a minute or two); the extracted version is written into the version ring but does not take
    effect. The console fills the form with the result, and it takes effect only when the user reviews it
    and clicks save."""
    return _wrap(service.suggest_graph_schema, kb_id)


@router.delete("/kbs/{kb_id}/graph_schema/{version_id}")
def delete_graph_schema_version(kb_id: str, version_id: str) -> dict[str, Any]:
    """Delete one historical label version. The version ring is maintained by the server, so deletion
    can only go through here -- a client writing graph_schema_versions directly is rejected by
    _reject_bad_config."""
    return _wrap(service.delete_graph_schema_version, kb_id, version_id)


@router.post("/kbs/{kb_id}/parse_now")
def parse_now(kb_id: str) -> dict[str, Any]:
    return _wrap(service.parse_now, kb_id)


@router.delete("/kbs/{kb_id}/graph")
def delete_graph(kb_id: str) -> dict[str, Any]:
    return _wrap(service.delete_graph, kb_id)


@router.post("/kbs/{kb_id}/unenroll")
def unenroll(kb_id: str) -> dict[str, Any]:
    return _wrap(service.unenroll, kb_id)


@router.get("/kbs/{kb_id}/files")
def kb_files(kb_id: str, request: Request, response: Response):
    """File table. While parsing, the console polls it every 2 seconds and the content mostly has not
    changed: an ETag is derived from the content, and a request with a matching If-None-Match gets
    304, saving the transfer and redraw of the whole table (health check D5)."""
    rows = _wrap(service.kb_files, kb_id)
    digest = hashlib.sha1(json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    etag = f'W/"{digest[:20]}"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    response.headers["ETag"] = etag
    return rows


@router.get("/kbs/{kb_id}/files/{file_id}/chunks")
def file_chunks(kb_id: str, file_id: str) -> dict[str, Any]:
    """The chunks an indexed file currently has in the knowledge base plus the diagnostics from
    indexing (no re-chunking), for the file table's "chunking preview" drawer."""
    return _wrap(service.file_chunks, kb_id, file_id)


@router.get("/kbs/{kb_id}/graph_preview")
def graph_preview(kb_id: str, limit: int = 60, q: str = "", upper: str = "", key: str = "") -> dict[str, Any]:
    """A part of the current graph version (entities picked by degree + the relations between them;
    limit=0 returns the whole version; with key / q given, centred on that entity), for the "graph
    preview" drawing."""
    return _wrap(service.graph_preview, kb_id, limit=limit, q=q, upper=upper, key=key)


@router.get("/kbs/{kb_id}/graph_merges")
def graph_merges(kb_id: str, limit: int = 2000) -> dict[str, Any]:
    """Entity merge log of the current version: every merged pair (merged into which / origin / kind of
    basis) and every blocked pair (reason), for the merge drawer of the "graph preview"."""
    return _wrap(service.graph_merges, kb_id, limit=limit)


@router.get("/kbs/{kb_id}/config")
def get_kb_config(kb_id: str) -> dict[str, Any]:
    return _wrap(service.get_kb_config, kb_id)


@router.put("/kbs/{kb_id}/config")
def put_kb_config(kb_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    _reject_bad_config(payload)
    return _wrap(service.update_kb_config, kb_id, payload)


@router.post("/kbs/{kb_id}/reparse")
def reparse(kb_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    reason = str((payload or {}).get("reason") or "web console reparse")
    return _wrap(service.reparse_kb, kb_id, reason)


@router.get("/jobs/{job_id}")
def job_detail(job_id: str) -> dict[str, Any]:
    return _wrap(service.job_detail, job_id)


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    return _wrap(service.cancel_job, job_id)


@router.post("/files/retry")
def retry_file(payload: dict[str, Any]) -> dict[str, Any]:
    file_id = str(payload.get("file_id") or "").strip()
    if not file_id:
        raise HTTPException(status_code=422, detail="file_id is required")
    return _wrap(service.retry_file, file_id)


@router.post("/services/restart_all")
def restart_all_services(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Restart everything in one click (system service containers). force=true is the user's second
    confirmation after seeing the "system busy" prompt."""
    return _wrap(service.restart_all_services, bool((payload or {}).get("force")))


@router.post("/services/stop_all")
def stop_all_services(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Stop everything in one click (system service containers)."""
    return _wrap(service.stop_all_services, bool((payload or {}).get("force")))


@router.post("/services/{key}/restart")
def restart_service(key: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    # force=true is the user's second confirmation after seeing the "system busy" prompt
    force = bool((payload or {}).get("force"))
    return _wrap(service.restart_service, key, force)


@router.get("/llms")
def list_llms() -> list[dict[str, Any]]:
    return service.list_llms()


@router.post("/llms")
def save_llm(payload: dict[str, Any]) -> dict[str, Any]:
    return _wrap(service.save_llm, payload)


@router.delete("/llms/{name}")
def delete_llm(name: str) -> dict[str, Any]:
    """Delete one model registration. A knowledge base referencing it is no longer a reason to refuse
    -- the references are cleared, the return value says which ones, and the console reports that
    faithfully."""
    return _wrap(service.remove_llm, name)


def _reject_bad_config(payload: dict[str, Any]) -> None:
    """Type guard for config values. The payload is declared dict[str, Any], and max_tokens used to be
    allowed as a string / list / float; once written into config_json it only blew up in build_source,
    and what blew up then was the whole load_settings (500 across the site)."""
    int_keys = {"max_tokens", "overlap_tokens", "graph_unit_chunks", "graph_max_gleanings",
                "graph_tune_sample_size"}
    str_keys = {"vlm_prompt", "graph_rebuild_interval", "graph_rebuild_new_chunk_pct",
                "graph_rebuild_operator"}
    for key, value in (payload or {}).items():
        if value is None:
            continue
        if key in int_keys and not isinstance(value, int):
            raise HTTPException(status_code=422, detail=f"{key} must be an integer")
        if key in str_keys and not isinstance(value, str):
            raise HTTPException(status_code=422, detail=f"{key} must be a string")
        if key in ("graph_enabled", "graph_auto_append", "graph_rebuild_resuggest") and not isinstance(value, bool):
            raise HTTPException(status_code=422, detail=f"{key} must be a boolean")
        if key in ("graph_predicates", "graph_entity_types") and not isinstance(value, (list, str)):
            raise HTTPException(status_code=422, detail=f"{key} must be a list")
        if key == "graph_parent_types" and not isinstance(value, dict):
            raise HTTPException(status_code=422, detail="graph_parent_types must be an object")
        if key in ("graph_type_definitions", "graph_profile") and not isinstance(value, dict):
            raise HTTPException(status_code=422, detail=f"{key} must be an object")
        if key == "graph_examples" and not isinstance(value, str):
            raise HTTPException(status_code=422, detail="graph_examples must be a string")
        if key == "graph_llm" and not isinstance(value, dict):
            raise HTTPException(status_code=422, detail="graph_llm must be an object")
        if key == "graph_schema_active" and not isinstance(value, str):
            raise HTTPException(status_code=422, detail="graph_schema_active must be a string")
        # The version ring is maintained by the server during "extract labels". Letting the client
        # write it would turn "when this version was extracted, with which model and how large a
        # sample" into something anyone can make up -- and the whole value of rolling back to a
        # version rests on that record being trustworthy.
        if key == "graph_schema_versions":
            raise HTTPException(status_code=422, detail="Label versions are maintained by the server and cannot be written directly")
        # Paused is the trace of an action, not an option: it is written by "pause graph build" and
        # cleared by "build / rebuild now" and "enable knowledge graph". Letting a config save write it
        # would produce the inconsistent state "paused is ticked in the form but nothing is paused".
        if key == "graph_paused":
            raise HTTPException(status_code=422, detail="The paused state is maintained by the build operations and cannot be written directly")

