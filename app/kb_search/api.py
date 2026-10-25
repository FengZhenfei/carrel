from __future__ import annotations

import base64
import hmac
import io
import json
import sys
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

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


# 时间口径(关系、事实、文档状态共用,见 kb_pipeline.graph.currency):默认只给现行的;as_of 回看某一天
# (YYYY / YYYY-MM / YYYY-MM-DD,只给年或月就是那段时间的最后一天);history 全给,每行标 status
AS_OF_PATTERN = r"^\d{4}(-\d{1,2}(-\d{1,2})?)?$"


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    kbs: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    hints: dict[str, Any] | None = None
    context: bool = True
    explain: bool = False
    image_b64: str | None = Field(default=None, max_length=12_000_000)   # 图→图(Q17):调用方给图片,base64 或 data URL
    history: bool = False                                                 # 被取代 / 归档的文档、旧的说法也要
    as_of: str | None = Field(default=None, pattern=AS_OF_PATTERN)        # 回看某一天


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
    history: bool = False                                          # 被取代 / 过期的关系也要
    as_of: str | None = Field(default=None, pattern=AS_OF_PATTERN)


class EntitiesRequest(BaseModel):
    # 改版前的 parent_types 已经去掉:带着它来的请求直接 422,不悄悄忽略、列出一份没按上层类筛的清单
    model_config = ConfigDict(extra="forbid")

    kb_id: str
    types: list[str] | None = None                                 # 只要这些实体类型及其子类型(不分大小写;库里有哪些类型见 /catalog 或 /graph/schema)
    uppers: list[str] | None = None                                # 或按上层类:agent / entity / part / property / process / event / concept / document
    kind: str = Field(default="all", pattern="^(instance|class|all)$")        # 都要(默认)/ 只要具体的 / 只要类别名
    name: str | None = Field(default=None, max_length=200)         # 标题、别名或外部编号里包含这段文字
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0, le=1_000_000)


class FactsRequest(BaseModel):
    kb_id: str
    subject: str | None = Field(default=None, max_length=200)      # 主体的标题或别名(不分大小写)
    subject_id: str | None = Field(default=None, max_length=200)   # 或直接给实体 id
    prop: str | None = Field(default=None, max_length=200, alias="property")   # 属性名 / 符号 / 规范概念名
    match: str = Field(default="auto", pattern="^(auto|exact|contains)$")      # auto:先精确,一条都没有再按包含
    doc_id: str | None = Field(default=None, max_length=200)       # 只要出自这份文档的事实
    rel_path: str | None = Field(default=None, max_length=1000)    # 或按库内路径指定文档
    conflict_only: bool = False                                    # 只要带跨文档冲突组的事实;主体、属性、文档、本项至少给一个
    history: bool = False                                          # 序列里更早的档位、被取代 / 过期的也要
    as_of: str | None = Field(default=None, pattern=AS_OF_PATTERN)
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0, le=1_000_000)


class SchemaRequest(BaseModel):
    kb_id: str
    version: str | None = Field(default=None, max_length=100)      # 模式版本 id;不给就是建现行这版图用的那个
    format: str = Field(default="json", pattern="^(json|yaml)$")


class StatsRequest(BaseModel):
    kb_id: str


class PagesRequest(BaseModel):
    kb_id: str
    history: bool = False
    as_of: str | None = Field(default=None, max_length=20)
    kind: str | None = Field(default=None, pattern="^(subject|timeline|source|index)$")
    title: str | None = Field(default=None, max_length=200)        # the title contains this text (case-insensitive, whitespace ignored)
    id: str | None = Field(default=None, max_length=200)           # page id
    entity_id: str | None = Field(default=None, max_length=200)    # pages involving this entity
    doc_id: str | None = Field(default=None, max_length=200)       # pages involving this document
    rel_path: str | None = Field(default=None, max_length=1000)
    with_text: bool = True
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0, le=1_000_000)


class DocsRequest(BaseModel):
    kb_id: str
    dir: str | None = Field(default=None, max_length=500)          # only documents under this directory (subdirectories included), compared as a "dir/" prefix
    name: str | None = Field(default=None, max_length=200)         # rel_path contains this text (case-insensitive)
    include_deleted: bool = False
    limit: int = Field(default=500, ge=1, le=2000)
    offset: int = Field(default=0, ge=0, le=1_000_000)


class GrepRequest(BaseModel):
    kb_id: str
    phrases: list[str] = Field(min_length=1, max_length=8)
    rel_paths: list[str] | None = Field(default=None, max_length=500)
    doc_ids: list[str] | None = Field(default=None, max_length=500)
    fields: list[Literal["body", "title", "visual"]] | None = Field(default=None, min_length=1)
    limit: int = Field(default=30, ge=0, le=200)                   # hits returned per phrase; 0 returns counts only

    @field_validator("phrases")
    @classmethod
    def _phrases(cls, value: list[str]) -> list[str]:
        out = [str(p).strip() for p in value]
        if any(not 1 <= len(p) <= 200 for p in out):
            raise ValueError("each phrase must be 1-200 characters after trimming")
        return out


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
    requests could not be traced afterwards. The question text is not logged (questions can themselves contain
    private information); degraded entries hold only knowledge base names, channel names and
    exception types. The whole line is written at once (written in two parts, the lines of concurrent
    requests would run together); failing to write the log must not fail a search that is already done."""
    try:
        s = (out.get("retrieval_summary") if isinstance(out, dict) else None) or {}
        line = {"timings_ms": s.get("timings_ms"), "kbs": len(s.get("kbs") or []), "widened": (s.get("routing") or {}).get("widened"),
                "rerank": s.get("rerank"), "evidence_state": s.get("evidence_state"), "hits": (s.get("sources") or {}).get("hits"),
                "image_query": s.get("image_query"), "degraded": s.get("degraded") or []}
        if s.get("ranking"):
            line["ranking"] = s["ranking"]
        sys.stdout.write("[search.request] " + json.dumps(line, ensure_ascii=False, default=str) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


@router.post("/search", dependencies=[Depends(require_token)])
def search(req: SearchRequest) -> dict[str, Any]:
    out = _wrap(service.search, req.question, kbs=req.kbs, top_k=req.top_k, hints=req.hints, with_context=req.context,
                explain=req.explain, image_bytes=_decode_image(req.image_b64), history=req.history, as_of=req.as_of)
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
                 direction=req.direction, history=req.history, as_of=req.as_of)


@router.post("/graph/entities", dependencies=[Depends(require_token)])
def graph_entities(req: EntitiesRequest) -> dict[str, Any]:
    return _wrap(service.graph_entities, req.kb_id, types=req.types, uppers=req.uppers, kind=req.kind, name=req.name, limit=req.limit,
                 offset=req.offset)


@router.post("/graph/facts", dependencies=[Depends(require_token)])
def graph_facts(req: FactsRequest) -> dict[str, Any]:
    return _wrap(service.graph_facts, req.kb_id, subject=req.subject, subject_id=req.subject_id, prop=req.prop, match=req.match,
                 doc_id=req.doc_id, rel_path=req.rel_path, conflict_only=req.conflict_only, history=req.history, as_of=req.as_of,
                 limit=req.limit, offset=req.offset)


@router.post("/graph/schema", dependencies=[Depends(require_token)])
def graph_schema(req: SchemaRequest) -> dict[str, Any]:
    return _wrap(service.graph_schema, req.kb_id, version=req.version, fmt=req.format)


@router.post("/graph/stats", dependencies=[Depends(require_token)])
def graph_stats(req: StatsRequest) -> dict[str, Any]:
    return _wrap(service.graph_stats, req.kb_id)


@router.post("/graph/pages", dependencies=[Depends(require_token)])
def graph_pages(req: PagesRequest) -> dict[str, Any]:
    return _wrap(service.graph_pages, req.kb_id, kind=req.kind, title=req.title, page_id=req.id, entity_id=req.entity_id, doc_id=req.doc_id,
                 rel_path=req.rel_path, with_text=req.with_text, limit=req.limit, offset=req.offset, history=req.history, as_of=req.as_of)


# POST /docs shares its path with FastAPI's own API documentation page (GET /docs) under a different method; the
# two do not interfere
@router.post("/docs", dependencies=[Depends(require_token)])
def docs(req: DocsRequest) -> dict[str, Any]:
    return _wrap(service.docs, req.kb_id, dir=req.dir, name=req.name, include_deleted=req.include_deleted, limit=req.limit, offset=req.offset)


@router.post("/grep", dependencies=[Depends(require_token)])
def grep(req: GrepRequest) -> dict[str, Any]:
    return _wrap(service.grep, req.kb_id, req.phrases, fields=list(req.fields) if req.fields else None, rel_paths=req.rel_paths,
                 doc_ids=req.doc_ids, limit=req.limit)

