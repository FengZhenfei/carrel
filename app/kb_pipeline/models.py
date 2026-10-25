from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GraphRebuildPolicy:
    interval_days: int | None = None
    # Two forms of the new-content condition, mutually exclusive: the console dropdown picks one. The ratio
    # suits KBs of stable size (rebuild after 10% growth); the absolute count suits KBs of very different
    # sizes -- 20% growth of a 1496-file KB is three hundred chunks, 20% of a 28-file KB is just a few.
    new_chunk_ratio: float | None = None
    new_chunk_count: int | None = None
    operator: str = "or"

    @property
    def enabled(self) -> bool:
        return (self.interval_days is not None
                or self.new_chunk_ratio is not None
                or self.new_chunk_count is not None)


@dataclass(frozen=True)
class KBSource:
    kb_id: str
    collection: str
    source_root: str
    source_type: str
    max_tokens: int
    overlap_tokens: int
    physical_base: Path | None = None
    # Per-KB override of the VLM caption instruction paragraph; the JSON
    # field guide is always appended by vision.vlm.compose_prompt.
    vlm_prompt: str | None = None
    graph_enabled: bool = False
    # The persistent intent written by "pause graph build"; the automatic rebuild must respect it.
    graph_paused: bool = False
    # New / changed documents are automatically appended to the current graph (without waiting for a full
    # rebuild); when off, they only enter the graph at the next full rebuild.
    graph_auto_append: bool = True
    # How many consecutive raw chunks each extraction unit merges (1 = no merging; see graph/units.py).
    graph_unit_chunks: int = 3
    # Gleaning rounds: how many "what else was missed" follow-ups per unit after the first extraction; 0–2,
    # default 1 (always two calls).
    graph_max_gleanings: int = 1
    # 模式字段:由现行模式版本(LinkML,见 kb_pipeline/ontology.py)展开出来的派生数据,
    # 不是你手写的配置:留空即回退到 ontology.default_model / 不加语言约束,
    # 删图时随图谱数据一并清空(见 maintenance._drop_graph_data)。
    # graph_entity_types 是抽取菜单(非 abstract 的类型);graph_parent_types 是每个类型的上层类;
    # graph_type_parents 是类型树(直接父类);graph_type_scopes 是逐类型判定的作用域。
    graph_entity_types: tuple[str, ...] = ()
    graph_language: str | None = None
    graph_predicates: tuple[dict[str, Any], ...] = ()
    graph_parent_types: dict[str, str] = field(default_factory=dict)
    graph_type_parents: dict[str, str] = field(default_factory=dict)
    graph_type_scopes: dict[str, str] = field(default_factory=dict)
    # 2026-09-07 模式层补齐(同样是派生数据,随标签版本一起生效):类型定义(抽取菜单里的一句话)、
    # 从本库语料生成的 few-shot、场景画像(主体类型 / 轴 / 结论段标题 / 延伸谓词)
    graph_type_definitions: dict[str, str] = field(default_factory=dict)
    graph_examples: str = ""
    graph_profile: dict[str, Any] = field(default_factory=dict)
    # 「立即/重新抽取标签」采样多少个片段喂给 LLM。选取方式固定为 random,不开放配置。
    graph_tune_sample_size: int = 0
    graph_rebuild_policy: GraphRebuildPolicy = field(default_factory=GraphRebuildPolicy)

    @property
    def block_merge_tokens(self) -> int:
        """PDF and DOCX text blocks are merged toward this target before chunking.
        Derived, not configured: both empirically tuned presets (400->800,
        library 800->1600) sit at exactly 2x max_tokens -- big enough that
        most chunk boundaries fall inside a merged run (overlap stitches
        those), small enough that page provenance stays tight."""
        return 2 * self.max_tokens


@dataclass
class SourceFile:
    kb_id: str
    collection: str
    source_root: str
    source_type: str
    file_key: int
    source_path: str
    rel_path: str
    filename: str
    dir: str
    physical_path: str
    mime_type: str
    size: int
    mtime: int
    checksum: str | None = None

    @property
    def doc_id(self) -> str:
        return f"{self.kb_id}:{self.file_key}"

    @property
    def content_version(self) -> str:
        return self.checksum or f"mtime:{self.mtime}:size:{self.size}"


@dataclass
class ParsedBlock:
    parser: str
    parser_profile: str
    doc_type: str
    block_type: str
    text: str
    block_id: str
    page_idx: int | None = None
    slide_idx: int | None = None
    sheet_name: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    bbox: list[float] | None = None
    title: str | None = None
    caption: str | None = None
    table_markdown: str | None = None
    latex: str | None = None
    visual_summary: str | None = None
    visual_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedChunk:
    chunk_uid: str
    chunk_index: int
    text: str
    block: ParsedBlock
    token_count: int = 0
    # Embedding prefix "document name > section path": goes into the vector only, not into the payload text
    # (see chunking.chunker.embedding_input)
    embedding_context: str | None = None
