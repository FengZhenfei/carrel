# 检索与 API

[首页](../README.zh-CN.md) · [English](retrieval.md) | 中文

检索服务从已索引的文档和图谱中获取证据。调用方智能体据此组织回答，按需补查上下文，并列出所依据的文件。

## 接入智能体

配套的 [carrel-search Skill](../skills/carrel-search/SKILL.md) 包含仅依赖 Python 标准库的客户端，以及检索、核验证据、作答、列出参考文件和沿图谱给出追问的操作说明。按所用智能体支持的方式安装该 Skill。

客户端默认连接 `http://127.0.0.1:9810`。服务位于其他主机时，在客户端环境中设置 `CARREL_SEARCH_BASE_URL` 和 `CARREL_SEARCH_TOKEN`；也支持 `~/.config/carrel-search/config.json` 及独立令牌文件。配置优先级、超时、图片输入和命令语法见 [客户端 API 参考](../skills/carrel-search/references/api.md)。

在仓库根目录可以这样发起本机检索：

```bash
python3 skills/carrel-search/scripts/carrel_search.py search \
  --question "资料中对交付和验收有哪些要求？"
```

每条命令输出一份精简结果：检索状态、来源正文及其所在的文件和位置、事实、页面摘要、图谱线索和参考文件。精简掉的是各种 ID、分数和调试字段，正文不删；一次工具输出装不下时分页，`show <调用编号> --page 2` 取下一页。完整响应保存在工作目录里，每一条都有编号，后续命令用 `--ref <调用编号>:<条目编号>` 指向它，不必重抄 ID：`context --ref 3fa2c1:S3` 回读某条来源前后的原文，`context --ref 3fa2c1:S3 --whole` 通读它所在的整份文档，`neighbors --ref 3fa2c1:H1` 从某个主体沿图谱走一跳，`show 3fa2c1:F2` 打印某一条的全部字段。加 `--json` 则输出完整响应。

Skill 检索一次就作答：先是答案，然后列出所依据的文件（从知识库一级目录开始的明文路径），最后给出几条带序号的追问，追问取自图谱线索。用户回一个序号，智能体就从对应的线索沿图谱走一跳，再答一次。要不要深挖由用户决定，首次回答不必等一轮多跳探索。

一次回答要多久，主要取决于智能体的推理档位，而不是检索服务：检索几秒内返回，其余时间是智能体在阅读和思考，最高档的耗时是中等偏上档位的数倍。Claude Code 可以在 `SKILL.md` 的 frontmatter 里加一行 `effort:`（例如 `effort: high`）给这个 Skill 单独定档，只对调用 Skill 的那一轮生效；其他智能体按会话选择档位。

令牌保存在客户端环境变量或独立令牌文件中。服务端使用 `KB_SEARCH_TOKEN`，客户端使用上述 `CARREL_` 变量。服务端没有设置令牌时，受保护接口只接受本机调用；`/health` 无需认证。

运行中的服务每分钟重读一次 `config/knowledge-base.env` 里的 `KB_SEARCH_*`，改了令牌或参数一分钟内生效，无需重启，换下的旧令牌随即失效。监听地址和端口仍需重启才生效；在服务自身环境中显式设置的键优先于文件。

## HTTP 接口

| 接口 | 用途 |
|---|---|
| `GET /health` | 服务健康、认证模式和知识库元数据 |
| `GET /catalog` | 知识库名称、领域、规模、文件样本及图谱状态；`?refresh=1` 刷新目录 |
| `POST /search` | 返回原文，以及可用的实体、关系、事实、编译页面和主体的一跳邻域 |
| `POST /context` | 按切块序号范围补取文档内容 |
| `GET /image/{kb_id}/{point_id}` | 获取检索切块对应的图片 |
| `POST /crop` | 裁剪该图片中的指定区域 |
| `POST /graph/neighbors` | 查询实体周围带证据的关系，每条关系带一段原文摘录 |
| `POST /graph/entities` | 按类型、上层类或名称列出实体，带总数和翻页 |
| `POST /graph/facts` | 列出某个主体的结构化事实、某个属性在各主体上的事实、某份文档里的事实，或只列带跨文档冲突组的事实，带总数和翻页 |
| `POST /graph/pages` | 按类别、标题、实体或文档列出编译页面（主体页、时间线页、来源页、索引页），可带全文 |
| `POST /docs` | 列出一个知识库的全部文档，带类型、入库状态、切块数和最近一次未解决的失败 |
| `POST /grep` | 统计若干短语出现在多少切块、哪些文档里，附前几条命中和逐字核对 |

控制台在端口 9800 使用独立的 `/api` 管理接口，智能体通常连接端口 9810 的检索服务。

目录缓存 `KB_SEARCH_CATALOG_TTL` 秒（默认 60），开启或关闭知识库时立即重建。只有库里还有活跃文档时 `has_graph` 才为 true。建目录时某个库的向量库读不出来（集合尚未创建按空库处理），该条目的 `chunks` 和 `has_graph` 为 `null` 并带 `degraded` 说明，检索照常尝试它的图谱，这份目录约 15 秒后重建。

### 检索

```bash
curl -sS http://127.0.0.1:9810/search \
  -H "Authorization: Bearer $KB_SEARCH_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"question": "资料中对交付和验收有哪些要求？", "top_k": 8}'
```

示例中的 shell 变量需包含服务端配置的令牌；未设置令牌的本机部署可省略 Authorization 请求头。

`question` 必填。可选的 `kbs` 用于限定知识库，不传时自动路由；实际编号通过 `/catalog` 获取。`top_k` 范围为 1–50。`context` 是是否补取邻近内容的布尔开关；`explain` 返回检索诊断信息。图片查询可以提供 `image_b64`，或使用客户端的 `--image` 参数；服务读不出图片的上传会以 HTTP 422 拒绝（请发送 PNG 或 JPEG）。

每次请求有一个时间预算 `KB_SEARCH_REQUEST_BUDGET` 秒（默认 45，设为 `0` 表示不限），调用方的超时应大于它。每个阶段可用的时间取它自身的超时与剩余预算中较小的一个。预算不足时不再放宽到其他知识库；预算用完后也不再重排（保持融合顺序）、不补取邻近内容。这些情况都会记入 `retrieval_summary.degraded`。如果向量通道和关键词通道在所有知识库上都失败，或候选原文一条也取不回来，导致没有任何结果，`/search` 返回 HTTP 503，而不是空结果。

`hints` 支持四个字段：

| 字段 | 行为 |
|---|---|
| `doc_ids` | 按文档硬过滤 |
| `rel_paths` | 使用 API 返回的路径进行硬过滤 |
| `content_version` | 按内容版本硬过滤 |
| `block_types` | 软偏好，例如优先选择表格 |

多个硬过滤条件取交集，并约束附带的派生证据：包含范围外来源的页面会被省略，存在硬范围过滤时不附加关系。`hints_used`、`hints_ignored` 和 `hints_scope` 记录实际生效的条件。

### 补取原文与图片

将检索结果中的 `kb_id`、`doc_id`、`content_version` 传给 `/context`：

```json
{
  "kb_id": "<检索结果中的编号>",
  "doc_id": "<检索结果中的文档 ID>",
  "content_version": "<检索结果中的内容版本>",
  "chunk_from": 5,
  "chunk_to": 7
}
```

范围使用切块序号，传入返回的内容版本即可回查同一版文档。

`/image` 在镜像 PDF 版本与切块一致、且图像支持可用时使用 PDF 原图，否则可回退到解析缓存。`X-Image-Source` 表示实际来源。`/crop` 接受 `kb_id`、`point_id`、`bbox` 和可选的 `pad`。两个接口都使用 `sources` 中返回的 UUID 形式的 `point_id`，其他取值以 HTTP 422 拒绝。`bbox` 使用 0–1 比例或 0–1000 千分比，不接受像素：大于 1000 的数值直接拒绝，不超过 1000 的数值一律按千分比读取。实际裁剪的范围由 `X-Crop-Box` 返回。先查看图片，再确定裁剪范围，详见 [图片接口参考](../skills/carrel-search/references/api.md)。

### 查询图谱关系

`POST /graph/neighbors` 接受 `kb_id`、实体名称 `entity` 或 `entity_id`，以及可选的 `types`、`direction`、`limit`。同名实体有歧义时，应优先使用已返回的实体 ID。多跳追查由多次请求组成，每一步都需核对相关来源。

- 有向关系的 `in`、`out` 表示关系方向；无向关系可用 `direction=both` 查询两个存储方向。
- 默认返回 20 条，最多 100 条，按权重排序；`count` 记录本次返回的关系数，`total` 是同样筛选条件下的总数，`has_more` 表示是否还有未返回的。
- 接口一次返回有数量上限的邻域，可通过 `types` 缩小范围或调整 `limit`。
- `predicates` 列出这个实体各类关系的条数，不受 `limit` 和 `types` 影响，据此可知还有哪些种类可以按谓语去取。每个对端带 `degree`（它自己的关系数），实体带 `facts`（名下的事实条数）。
- 每条关系带一段原文摘录：排在最前的证据切块带 `excerpt`（200 字以内，取原文里提到对端的那几句）和 `excerpt_match`（`both` 表示两端的名字都在这一片里，`other` 表示只找到对端，`center` 或 `none` 表示没有定位到对端）。摘录按名字在原文里定位，不调用模型。提到对端的文字切块优先于图片切块；取自图片切块的摘录标 `visual`，因为那段文字是视觉模型的描述。
- 关系描述汇总关联证据，其来源位置可用于 `/context` 原文查询。

### 检索结果里的主体邻域

`/search` 返回 `neighborhoods`：问题所问的实体在图谱里通向哪里的预览，智能体据此决定是否调用 `/graph/neighbors`，不必先逐个查一遍。

- 主体首先是问题里点名的实体：标题或别名整段出现在问题里，不分大小写、不计空格。不含字母数字的两三个字的名字，只有当这个实体同时是图路的命中时才算。剩下的名额用图路匹配度最高的实体补足。`KB_SEARCH_NEIGHBORHOODS` 设置总数（默认 4，设为 `0` 关闭）。
- 每个主体带 `named`、`relations`（关系总数）、`facts`、`predicates`（各类关系的条数）和 `neighbors`：权重最高的 8 条关系，含谓语、方向和对端，同一个对端只列一条。
- 这一块只是线索，不带证据。带硬过滤 `hints` 的限定查询不返回它；构建失败时只在 `retrieval_summary.degraded` 里记录，不影响检索。
- 一个知识库全部实体的名字按图谱版本保留在检索进程里，重启或图谱换版后读取一次。

### 列举实体和事实

问题要求列出某一类的全部内容时（“所有产品”“这个器件的全部参数”）使用这两个接口；普通问题只用 `/search` 即可。

`POST /graph/entities` 接受 `kb_id`，以及可选的 `types`、`parent_types`（上层类 `entity`、`part`、`property`、`process`、`standard`、`document`）、`name`（标题或别名中包含的文字）、`limit`（默认 50，最多 200）和 `offset`。实体按重要程度排序，每个实体带 `docs`（提及它最多的文档）和 `doc_count`。第一页同时返回 `types`，即该知识库各类型的实体数量。

`POST /graph/facts` 接受 `kb_id`、主体名称 `subject` 或 `subject_id`、可选的 `property`（属性名、符号或概念名）、`match`（`auto`、`exact` 或 `contains`）、以 `doc_id` 或 `rel_path` 指定的文档、`conflict_only`、`limit` 和 `offset`。主体、属性、文档、`conflict_only` 至少提供一个，多个条件取交集。返回行的结构与 `/search` 的 `specs` 相同，并带 `evidence` 列表，其中的位置可用于 `/context`；每行有值时另带 `subject_id`（事实所属的实体）、`doc_id`（事实出自的文档）和 `concept_key`，取不到值的字段直接省略（不挂在任何实体下的事实没有 `subject_id`，没有记录来源文档的事实没有 `doc_id`）。

- 属性命中后，同一规范概念下的各种写法一并返回。`match=auto` 先精确匹配，没有结果时才按包含匹配；`property.matched` 说明实际采用的方式。
- 提供主体或文档时，第一页同时返回 `properties`，即这个范围里的属性及数量。
- 按 `rel_path` 指定的文档经状态库换成 `doc_id`；同一路径先后有过几个文件时，取没有删除的那个。路径不存在返回 HTTP 404，`doc_id` 与 `rel_path` 指向不同文档返回 HTTP 422。指定文档时，属性匹配也只在这份文档里进行。
- `conflict_only=true` 只保留带跨文档冲突组的事实，同一组的事实排在一起。文档与冲突条件会回显在 `filters` 里。
- `total`、`offset` 和 `has_more` 描述完整结果；用更大的 `offset` 请求下一页。
- 实体名称不区分大小写、不计空格。名称没有匹配时 `found=false`，`candidates` 列出相近实体，可按 ID 选择。
- 这些接口列出的是图谱在抽取时登记的内容，不能证明文档里没有其他内容。

### 编译页面

`POST /graph/pages` 用于整页读取编译页面，不必等检索恰好命中。它接受 `kb_id`，以及可选的 `kind`（`subject`、`timeline`、`source` 或 `index`）、`title`（标题中包含的文字，不区分大小写、不计空格）、`id`（页面 ID）、`entity_id`、以 `doc_id` 或 `rel_path` 指定的文档、`with_text`（默认 true）、`limit`（默认 20，最多 100）和 `offset`。

- 响应带 `kb_name`、`graph_version`、`kinds`、`pages`，以及与 `/graph/entities` 相同的翻页字段 `total`、`offset`、`limit`、`count` 和 `has_more`。
- 页面先按上述类别顺序、再按标题排序。`kinds` 是整个知识库各类页面的数量，不受筛选条件影响。
- 每个页面带 `n`（在完整结果中的序号）、`id`、`kind`、`title`、`summary`、`text`（`with_text=false` 时省略，`text_chars` 始终给出它的长度）、`series`、`docs`（文档路径，已无法对应的位置为 `null`）及对应的 `doc_ids`、`entity_ids`（可直接用于 `/graph/neighbors` 和 `/graph/facts`）、`concept_keys` 和 `path`（页面在编译视图里的文件路径，例如 `index.md` 或 `subjects/` 下的某个路径）。
- 页面正文存储时截到 8000 字以内，每个页面最多记录 32 份文档、32 个实体，因此按文档或实体筛选可能漏掉涉及更多文档或实体的页面。
- 知识库没有页面集合时返回 HTTP 404；向量库无法访问时返回 HTTP 503。

### 文档清单与短语计数

问题需要完整的文档清单，或某个短语出现的次数，而不是最相关的几段原文时，使用这两个接口。

`POST /docs` 接受 `kb_id`，以及可选的 `dir`（该目录及其子目录下的文档）、`name`（`rel_path` 中包含的文字，不区分大小写）、`include_deleted`、`limit`（默认 500，最多 2000）和 `offset`。文档清单读自状态库，按 `rel_path` 排序。响应带 `kb_name`、`totals`、`docs`，以及翻页字段 `total`、`offset`、`limit`、`count` 和 `has_more`。每份文档带 `n`（在完整结果中的序号）、`doc_id`、`rel_path`、`filename`、`dir`、`doc_type`、`mime_type`、`size`、`mtime`、`content_version`、`status`、`indexed`、`chunk_total`、`parser_profile` 和 `first_seen_at`，有记录时另带 `diag`（切块验收的摘要）和 `last_error`（最近一次未解决的失败）。

- `chunk_total` 只计现行内容版本的活跃切块，与检索能看到的一致；解析版本就是现行版本并且切出了切块时，`indexed` 为 true。`totals` 汇总符合筛选条件的全部文档，不受翻页影响，含 `files`、`indexed`、`not_indexed` 和 `chunks` 四项。
- 失败消息截到 300 字以内，去掉堆栈和服务器路径。响应里只有知识库内的相对路径。
- `POST /docs` 与交互式接口文档页（`GET /docs`）同一路径、不同方法，互不影响。

`POST /grep` 接受 `kb_id`、`phrases`（1–8 个，去掉首尾空白后每个 1–200 字）、用于限定范围的可选 `rel_paths` 和 `doc_ids`（各最多 500 个）、`fields`（`body`、`title`、`visual` 中的若干个，默认三个都查）和 `limit`（每个短语返回的命中数，默认 30，最多 200；`0` 只返回计数）。

- 响应带 `kb_name`、`fields`（实际查的字段）、`scope`（`rel_paths` 与 `doc_ids` 各限定了多少个）、`note`（计数的读法）和 `results`，每个短语一项。
- 每个短语返回 `total_chunks`、`docs`（命中的每份文档及其切块数，切块多的在前）和按路径、切块序号排序的 `hits`。每条命中带位置、命中的字段 `field`，以及短语前后的一小段片段，不返回整段原文。
- 计数采用关键词索引分析器的短语匹配。中日韩文字按二元组建索引，匹配可能比字面短语宽松，因此每条返回的命中另做一次逐字核对（经 Unicode NFKC 归一并去掉空白后比较）：每条命中带 `literal`，每个短语带 `literal_checked` 和 `literal_true`。总数没有逐条核对。
- 只计现行内容版本的切块。关键词索引同步还没跟上的文档，计数可能与现状有出入。`docs_truncated` 表示命中的文档超过 1000 份。
- 关键词索引尚未建立时按 0 命中处理并在 `degraded` 里记录；无法访问时返回 HTTP 503。

响应中带知识库的目录名（`/search` 为 `kb_names`，其余为 `kb_name`）。引用文档时写作 `<目录名>/<rel_path>`，后接 `place`：来源和证据切块自带的短定位（页码、幻灯片或工作表行；不分页的文档为最深一级标题）。

## 理解返回证据

| 字段 | 使用方式 |
|---|---|
| `sources` | 原文、来源位置、接受标记及可用的视觉描述 |
| `entities`、`relationships` | 图谱候选与关联，关键关系需回查来源 |
| `neighborhoods` | 问题点名的实体权重最高的几条关系，是沿图谱追查的线索 |
| `specs` | 带单位、条件、时间、来源与冲突标记的结构化事实；`evidence` 是事实出自的切块，带可用于 `/context` 的位置，能分辨时正文里写着该值的那一块排在最前（`located`） |
| `pages` | 编译的主体页、时间线页或来源页，`compiled=true` 表示派生内容 |
| `doc_aggs` | 本次结果涉及的文档 |
| `retrieval_summary` | 路由、选择、证据状态和降级诊断 |

首先检查 `retrieval_summary.evidence_state`：`accepted`、`diagnostic` 或 `unranked`。当 `no_relevant_content=true` 时，返回的候选不足以支持答案。重排不可用时可能返回 `unranked`；服务错误与不可用通道会在诊断字段中单独报告。

事实、页面、实体和关系还需检查 `verified` 和 `sources_active`；实体和关系按代表性来源点核验，每份文档一个。文档删除或重新解析后，图谱要到下一个版本才会跟上；在此之前，来源全部失效的条目标为 `verified=false`，不能作为现行依据，这类事实不给线索，这类页面不附正文。来源点未能核验的条目不做标记。冲突标记用于识别记录间的差异；比较测量结果时，可结合主体、单位、日期和条件字段。返回的事实按序列归组。

`text_truncated` 表示可能需要补取原文，`stitched.pieces` 给出拼接片段的位置。来源正文受 `KB_SEARCH_CONTEXT_TOKENS` 限制，编译页面另受 `KB_SEARCH_PAGE_TEXT_TOKENS` 限制，这两个预算独立于 `top_k`。被省略的条件、表头或上下文影响结论时，可调用 `/context` 补取。

不可用阶段通过 `retrieval_summary.degraded` 报告。服务可能退回融合顺序、关键词候选或剩余检索通道；响应会记录可用通道和可引用的来源位置。常见条目：

| 条目 | 含义 |
|---|---|
| `visual_query: …` | 图片查询的图片向量没有算出来，结果只来自问题文字 |
| `catalog: …` | 建目录时读不到向量库，切块数与图谱状态未知 |
| `lexical_profile: …` | 选择知识库时缺少词法证据 |
| `<kb>:graph`、`<kb>:visual`、`rerank: TimeoutError` | 该阶段失败、超时或没有剩余预算；重排退回融合顺序 |
| `budget: widen skipped` | 剩余预算不足，没有放宽到其他知识库 |
| `budget: context skipped`、`context: …` | 没有补取邻近切块或表头 |
| `budget: neighborhoods skipped`、`neighborhoods: …` | 没有生成主体邻域 |
| `budget: spec evidence skipped`、`<kb>:spec_evidence: …` | 事实没有带来源切块，或该知识库的切块只有 ID |
| `<kb>:backfill: …` | 候选原文取不回来，没有原文的候选被丢弃 |
| `widen: rerank: …` | 放宽后的重排失败，保留首轮「全部低于下限」的结论 |

## 检索评测

题集与结果放在 `runtime/eval/`，不提交到 Git。准备好题集后执行：

```bash
cd app
.venv/bin/kb search eval --set ../runtime/eval/my-set.json \
  --out ../runtime/eval/result.json --auto
```

`--auto` 使用自动路由，不使用题目预先指定的知识库。评测输出切片与文档命中率、MRR、预期答案检查、文档覆盖、负例表现、延迟、错误，以及相对上次结果的变化。重新解析会改变切块的 point ID，因此开跑前会核对切片金标：金标切片一个都不在了的题目计入 `stale_gold`，不计入 hit@k 和 MRR，文档金标与预期答案照常判定。要恢复切片口径，重新运行 `make-set`。

`kb search make-set` 可调用知识库的抽取模型生成题集初稿。审核问题和预期证据后，可在选定的资料与模型配置上运行评测。
