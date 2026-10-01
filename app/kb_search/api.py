from __future__ import annotations

import base64
import hmac
import io
import json
import sys
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from . import service

router = APIRouter()


def require_token(request: Request) -> None:
    """Static Bearer (Q01): when a token is configured it must be sent; without one only loopback
    requests are accepted."""
    _, ss, _ = service.runtime()
    if ss.token:
        header = (request.headers.get("authorization") or "").strip()
        given = header[7:].strip() if header.lower().startswith("bearer ") else ""
        if not given or not hmac.compare_digest(given.encode(), ss.token.encode()):      # comparison time does not depend on how many characters were guessed right
            raise HTTPException(status_code=401, detail="missing or invalid bearer token")
        return
    host = request.client.host if request.client else ""
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(status_code=401, detail="KB_SEARCH_TOKEN is not configured; only loopback requests are accepted")


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    kbs: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    hints: dict[str, Any] | None = None
    context: bool = True
    explain: bool = False
    image_b64: str | None = Field(default=None, max_length=12_000_000)   # image-to-image (Q17): the caller supplies an image, base64 or data URL


class ContextRequest(BaseModel):
    kb_id: str
    doc_id: str
    content_version: str | None = None
    chunk_from: int = Field(ge=0)
    chunk_to: int = Field(ge=0)


class NeighborsRequest(BaseModel):
    kb_id: str
    entity: str | None = Field(default=None, max_length=200)      # entity title or alias (case-insensitive)
    entity_id: str | None = Field(default=None, max_length=200)   # or the entity id directly (the id from a previous response)
    limit: int = Field(default=20, ge=1, le=100)
    types: list[str] | None = None                                 # only these predicates
    direction: str = Field(default="both", pattern="^(both|out|in)$")


class EntitiesRequest(BaseModel):
    kb_id: str
    types: list[str] | None = None                                 # only these entity types (case-insensitive; the types a knowledge base has are in /catalog and in the first page's types)
    parent_types: list[str] | None = None                          # or by upper class: entity / part / property / process / standard / document
    name: str | None = Field(default=None, max_length=200)         # a piece of text contained in the title or an alias
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0, le=1_000_000)


class FactsRequest(BaseModel):
    kb_id: str
    subject: str | None = Field(default=None, max_length=200)      # the subject's title or alias (case-insensitive)
    subject_id: str | None = Field(default=None, max_length=200)   # or the entity id directly
    prop: str | None = Field(default=None, max_length=200, alias="property")   # property name / symbol / canonical concept name; at least one of subject and property
    match: str = Field(default="auto", pattern="^(auto|exact|contains)$")      # auto: exact first, containment only when nothing matched
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0, le=1_000_000)


class CropRequest(BaseModel):
    kb_id: str
    point_id: str
    bbox: list[float] = Field(min_length=4, max_length=4)   # 0-1 fractions or 0-1000 per-mille; pixels are not accepted (any value up to 1000 is read as per-mille)
    pad: int = Field(default=16, ge=0, le=200)


def _wrap(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"not found: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except service.RetrievalUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))


def _decode_image(b64: str | None) -> bytes | None:
    if not b64:
        return None
    raw = b64.strip()
    if raw.startswith("data:"):
        raw = raw.split(",", 1)[1] if "," in raw else ""
    try:
        data = base64.b64decode(raw, validate=False)
    except Exception:
        raise HTTPException(status_code=422, detail="image_b64 is not valid base64")
    try:
        from PIL import Image

        Image.open(io.BytesIO(data)).verify()
    except Exception:
        # An unreadable image (corrupt, not an image, a format without an installed decoder) is rejected here;
        # otherwise the visual channel is silently skipped and the result comes from the question text alone
        raise HTTPException(status_code=422, detail="image_b64 is not an image this service can read (send PNG or JPEG)")
    return data


def _image_response(img: dict[str, Any]) -> Response:
    headers = {"X-Image-Source": str(img.get("source") or ""), "X-Image-Width": str(img.get("width") or 0), "X-Image-Height": str(img.get("height") or 0)}
    if img.get("box"):
        headers["X-Crop-Box"] = ",".join(str(v) for v in img["box"])
    return Response(content=img["bytes"], media_type=img["mime"], headers=headers)


@router.get("/health")
def health() -> dict[str, Any]:
    return service.health()


@router.get("/catalog", dependencies=[Depends(require_token)])
def catalog(refresh: bool = False) -> dict[str, Any]:
    return service.catalog(force=refresh)


def _log_search(out: dict[str, Any]) -> None:
    """Every search that returns a result leaves one line on the server: timings, degraded entries, widening
    and rerank state. If degradation were only written into the response, the degradation rate and slow
    requests could not be traced afterwards. The question text is not logged (in a health knowledge base the
    question itself is private); degraded entries hold only knowledge base names, channel names and
    exception types. The whole line is written at once (written in two parts, the lines of concurrent
    requests would run together); failing to write the log must not fail a search that is already done."""
    try:
        s = (out.get("retrieval_summary") if isinstance(out, dict) else None) or {}
        line = {"timings_ms": s.get("timings_ms"), "kbs": len(s.get("kbs") or []), "widened": (s.get("routing") or {}).get("widened"),
                "rerank": s.get("rerank"), "evidence_state": s.get("evidence_state"), "hits": (s.get("sources") or {}).get("hits"),
                "image_query": s.get("image_query"), "degraded": s.get("degraded") or []}
        sys.stdout.write("[search.request] " + json.dumps(line, ensure_ascii=False, default=str) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


@router.post("/search", dependencies=[Depends(require_token)])
def search(req: SearchRequest) -> dict[str, Any]:
    out = _wrap(service.search, req.question, kbs=req.kbs, top_k=req.top_k, hints=req.hints, with_context=req.context,
                explain=req.explain, image_bytes=_decode_image(req.image_b64))
    _log_search(out)
    return out


@router.post("/context", dependencies=[Depends(require_token)])
def context(req: ContextRequest) -> dict[str, Any]:
    return _wrap(service.context, req.kb_id, req.doc_id, req.chunk_from, req.chunk_to, content_version=req.content_version)


@router.get("/image/{kb_id}/{point_id}", dependencies=[Depends(require_token)])
def image(kb_id: str, point_id: str) -> Response:
    return _image_response(_wrap(service.image, kb_id, point_id))


@router.post("/crop", dependencies=[Depends(require_token)])
def crop(req: CropRequest) -> Response:
    return _image_response(_wrap(service.crop, req.kb_id, req.point_id, req.bbox, pad=req.pad))


@router.post("/graph/neighbors", dependencies=[Depends(require_token)])
def graph_neighbors(req: NeighborsRequest) -> dict[str, Any]:
    return _wrap(service.graph_neighbors, req.kb_id, entity=req.entity, entity_id=req.entity_id, limit=req.limit, types=req.types,
                 direction=req.direction)


@router.post("/graph/entities", dependencies=[Depends(require_token)])
def graph_entities(req: EntitiesRequest) -> dict[str, Any]:
    return _wrap(service.graph_entities, req.kb_id, types=req.types, parent_types=req.parent_types, name=req.name, limit=req.limit,
                 offset=req.offset)


@router.post("/graph/facts", dependencies=[Depends(require_token)])
def graph_facts(req: FactsRequest) -> dict[str, Any]:
    return _wrap(service.graph_facts, req.kb_id, subject=req.subject, subject_id=req.subject_id, prop=req.prop, match=req.match,
                 limit=req.limit, offset=req.offset)

