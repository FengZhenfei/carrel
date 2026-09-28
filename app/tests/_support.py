"""Shared test fixtures: repository file reading, fake blocks / chunks / units, a fake model client, and the
class-level fixtures shared by the test modules after the split."""
from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from types import SimpleNamespace

from kb_pipeline.graph.extract import entity_key
from kb_pipeline.graph.llm import ChatClient, LLMCache, LLMSpec
from kb_pipeline.graph.units import ChunkRef, Unit, write_units
from kb_pipeline.localfs.scanner import stable_int
from kb_pipeline.models import ParsedBlock, SourceFile


def _repo_file(rel: str) -> str:
    """Text of a file inside the repository. Structural tests (front/back-end constant alignment, systemd
    coverage, ...) read source files, and pytest runs from app/, where relative paths would not resolve."""
    return (Path(__file__).resolve().parents[2] / rel).read_text(encoding="utf-8")


def _local_file(rel_path: str, *, checksum: str = "same-checksum") -> SourceFile:
    parent = str(Path(rel_path).parent)
    if parent == ".":
        parent = ""
    return SourceFile(
        kb_id="project_materials",
        collection="kb_project",
        source_root="项目资料",
        source_type="local_mirror",
        file_key=stable_int(f"file:project_materials:{rel_path}"),
        source_path=f"项目资料/{rel_path}",
        rel_path=rel_path,
        filename=Path(rel_path).name,
        dir=parent,
        physical_path=f"/tmp/{rel_path}",
        mime_type="application/pdf",
        size=12,
        mtime=1,
        checksum=checksum,
    )


def _pin_tokenizer():
    """Pin the tokenizer to tiktoken's o200k_base; without it, skip every assertion that depends on it.

    On first use tiktoken downloads its encoding table over the network with no timeout; when that fails,
    count_tokens silently falls back to a heuristic, so the same token-budget assertion is really testing two
    different tokenizers on two machines.
    """
    from kb_pipeline import utils

    if utils._ENCODER is None:
        try:
            import tiktoken

            utils._ENCODER = tiktoken.get_encoding("o200k_base")
        except Exception:
            utils._ENCODER = False
    return utils._ENCODER is not False


def _block(block_id: str, text: str, **kw) -> ParsedBlock:
    base = dict(parser="mineru", parser_profile="pdf-mineru-table-vlm-v2", doc_type="pdf",
                block_type="text", text=text, block_id=block_id)
    base.update(kw)
    return ParsedBlock(**base)


def _mb(block_id: str, text: str, block_type: str = "text", **kw) -> ParsedBlock:
    base = dict(parser="mineru", parser_profile="mineru-3.4.4-pipeline-v1", doc_type="pdf",
                block_type=block_type, text=text, block_id=block_id, metadata={})
    base.update(kw)
    return ParsedBlock(**base)


def _text_of_tokens(n: int) -> str:
    """Build a Chinese string whose token count is exactly n. Boundaries in the tests are computed with the real
    tokenizer, not with approximations such as "one CJK character is roughly one token"."""
    from kb_pipeline.utils import count_tokens
    text = "书"
    while count_tokens(text) < n:
        text += "书"
    return text


def _fake_chat_client(responses):
    from kb_pipeline.graph.llm import ChatClient, LLMCache, LLMSpec

    queue = list(responses)

    def chat(spec, messages, **_):
        return queue.pop(0)

    return ChatClient(LLMSpec(name="m", base_url="http://x/v1", api_key="", model_id="m", protocol="openai"),
                      chat=chat, cache=LLMCache(None), backoff_base=0.0, backoff_max=0.0)


def _chunk(idx: int, text: str, *, block: str = "b1", section=("第一章",), block_type: str = "text",
           doc: str = "kb_003:1", tokens: int | None = None) -> ChunkRef:
    return ChunkRef(
        point_id=f"p{idx}", chunk_uid=f"kb_003:1:v1:pdf:{block}:{idx}", doc_id=doc, content_version="v1",
        chunk_index=idx, block_id=block, block_type=block_type, section_path=list(section), text=text,
        n_tokens=tokens if tokens is not None else max(1, len(text) // 2), rel_path="a.pdf", filename="a.pdf",
    )


SPEC = LLMSpec(name="m", base_url="http://x/v1", api_key="", model_id="model-a", protocol="openai")


def _client(responses, **kw) -> ChatClient:
    """responses: callable(messages) -> str, or a list popped in order."""
    if isinstance(responses, list):
        queue = list(responses)

        def chat(spec, messages, **_):
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
    else:
        def chat(spec, messages, **_):
            return responses(messages)
    kw.setdefault("cache", LLMCache(None))
    kw.setdefault("backoff_base", 0.0)
    kw.setdefault("backoff_max", 0.0)
    return ChatClient(SPEC, chat=chat, **kw)


def _unit(uid: str, doc: str = "d1", points=("p1",), order: int = 0) -> Unit:
    return Unit(unit_id=uid, doc_id=doc, rel_path=f"{doc}.pdf", section_path=["S"], block_ids=["b"],
                chunk_uids=[f"c{p}" for p in points], point_ids=list(points), n_tokens=10, text="t", order=order)


def _ext(entities, relations):
    return {"entities": [{"name": n, "key": entity_key(n), "type": t, "descriptions": [d], "mentions": 1, "types": {t: 1}}
                         for n, t, d in entities],
            "relations": [{"source": s, "target": t, "source_key": entity_key(s), "target_key": entity_key(t),
                           "predicate": p, "predicate_raw": p, "descriptions": [d], "strength": w}
                          for s, t, p, d, w in relations]}


def _bundle_dir(self, tmp: str):
    from kb_pipeline.graph.vectors import entity_id

    out = Path(tmp) / "work"
    out.mkdir()
    units = [_unit("u1", doc="kb_003:1", points=("p1", "p2")), _unit("u2", doc="kb_003:2", points=("p3",), order=1)]
    write_units(out / "units.jsonl", units)
    graph = {
        "entities": [
            {"key": "a", "title": "A", "type": "t", "parent_type": "p", "description": "da", "frequency": 2,
             "degree": 1, "pagerank": 0.6, "unit_ids": ["u1", "u2"], "aliases": ["a1"], "doc_ids": ["kb_003:1"]},
            {"key": "b", "title": "B", "type": "t", "parent_type": "", "description": "db", "frequency": 1,
             "degree": 1, "pagerank": 0.4, "unit_ids": ["u1"], "aliases": [], "doc_ids": ["kb_003:1"]},
        ],
        "relations": [
            {"source_key": "a", "target_key": "b", "source": "A", "target": "B", "predicate": "has_pin",
             "directed": True, "description": "A has B", "weight": 7.5, "strength_sum": 9, "evidence": 1,
             "cooccur": 1, "npmi": 0.3, "combined_degree": 2, "unit_ids": ["u1"], "type_violation": False},
        ],
        "mentions": [{"entity_key": "a", "point_id": "p1", "chunk_uid": "cp1", "count": 2},
                     {"entity_key": "a", "point_id": "p3", "chunk_uid": "cp3", "count": 0},
                     {"entity_key": "b", "point_id": "p2", "chunk_uid": "cp2", "count": 1},
                     {"entity_key": "ghost", "point_id": "p2", "chunk_uid": "cp2", "count": 1}],
        "stats": {},
    }
    (out / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    return out, graph, units, entity_id


class _CodexAudit20260906TestsSupport:
    """Fixtures shared by CodexAudit20260906Tests after it was split across the per-topic files."""

    @staticmethod
    def _chunks(blocks, max_tokens=120, overlap=0):
        from kb_pipeline.chunking.chunker import blocks_to_chunks
        return blocks_to_chunks(kb_id="kb", file_key=1, content_version="v", parser_profile="p",
                                blocks=blocks, max_tokens=max_tokens, overlap_tokens=overlap)

    NVSRAM_ROW_HTML = ("<table><tr><td>参数</td><td>说明</td><td>测试条件</td><td>最小值</td><td>典型值</td><td>最大值</td><td>单位</td></tr>"
                       "<tr><td> $V_{CC}$ </td><td>电源</td><td></td><td>2.7</td><td>3.0</td><td>3.6</td><td>V</td></tr>"
                       "<tr><td> $I_{CC1}$ </td><td>平均电流  $V_{CC}$ </td><td> $t_{RC} = 20 ns$  $t_{RC} = 25 ns$  $t_{RC} = 45 ns$ 无输出负载下取得的值( $I_{OUT} = 0 mA$ )</td>"
                       "<td>-</td><td>-</td><td>757557</td><td>mAmAmA</td></tr>"
                       "<tr><td> $I_{CC2}$ </td><td>存储电流</td><td>无需关注</td><td>-</td><td>-</td><td>20</td><td>mA</td></tr></table>")


class _CodexFinalTestsSupport:
    """Fixtures shared by CodexFinalTests after it was split across the per-topic files."""

    def _running_build(self, tmp: str):
        from kb_pipeline import db as dbm, discovery

        root = Path(tmp)
        state = root / "state.sqlite3"
        dbm.init_db(state)
        mirror = root / "mirror"
        (mirror / "库F").mkdir(parents=True)
        with dbm.connect(state) as con:
            s, _ = discovery.enroll(con, mirror, "库F")
            discovery.set_config(con, s.kb_id, {"graph_enabled": True})
            con.execute(
                "INSERT INTO graph_builds(graph_build_id, source_key, kb_id, source_collection, graph_version, "
                "status, started_at, worker_host, worker_pid, heartbeat_at) VALUES('gb', ?, ?, ?, 'v1', 'running', ?, ?, ?, ?)",
                (s.kb_id, s.kb_id, s.collection, int(time.time()), socket.gethostname(), os.getpid(), int(time.time())))
            con.commit()
        stub = SimpleNamespace(state_db=state, runtime_dir=root, graph_work_dir=root / "gw", mirror_root=mirror,
                               qdrant_url="http://q", qdrant_api_key="", opensearch_url="http://o", cache_dir=root / "cache")
        return stub, s
