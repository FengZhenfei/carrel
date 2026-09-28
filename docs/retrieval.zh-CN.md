# 检索与 API

[首页](../README.zh-CN.md) · [English](retrieval.md) | 中文

检索服务从已索引的文档和图谱中获取证据。调用方智能体据此组织回答，按需补查上下文，并引用来源。

## 接入智能体

配套的 [carrel-search Skill](../skills/carrel-search/SKILL.md) 包含仅依赖 Python 标准库的客户端，以及检索、核验证据、追查关系和引用的操作说明。按所用智能体支持的方式安装该 Skill。

客户端默认连接 `http://127.0.0.1:9810`。服务位于其他主机时，在客户端环境中设置 `CARREL_SEARCH_BASE_URL` 和 `CARREL_SEARCH_TOKEN`；也支持 `~/.config/carrel-search/config.json` 及独立令牌文件。配置优先级、超时、图片输入和命令语法见 [客户端 API 参考](../skills/carrel-search/references/api.md)。

在仓库根目录可以这样发起本机检索：

```bash
python3 skills/carrel-search/scripts/carrel_search.py search \
  --question "资料中对交付和验收有哪些要求？"
```

令牌保存在客户端环境变量或独立令牌文件中。服务端使用 `KB_SEARCH_TOKEN`，客户端使用上述 `CARREL_` 变量。服务端没有设置令牌时，受保护接口只接受本机调用；`/health` 无需认证。

## HTTP 接口

| 接口 | 用途 |
|---|---|
| `GET /health` | 服务健康、认证模式和知识库元数据 |
| `GET /catalog` | 知识库名称、领域、规模、文件样本及图谱状态；`?refresh=1` 刷新目录 |
| `POST /search` | 返回原文，以及可用的实体、关系、事实和编译页面 |
| `POST /context` | 按切块序号范围补取文档内容 |
| `GET /image/{kb_id}/{point_id}` | 获取检索切块对应的图片 |
| `POST /crop` | 裁剪该图片中的指定区域 |
| `POST /graph/neighbors` | 查询实体周围带证据的关系 |

控制台在端口 9800 使用独立的 `/api` 管理接口，智能体通常连接端口 9810 的检索服务。

### 检索

```bash
curl -sS http://127.0.0.1:9810/search \
  -H "Authorization: Bearer $KB_SEARCH_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"question": "资料中对交付和验收有哪些要求？", "top_k": 8}'
```

示例中的 shell 变量需包含服务端配置的令牌；未设置令牌的本机部署可省略 Authorization 请求头。

`question` 必填。可选的 `kbs` 用于限定知识库，不传时自动路由；实际编号通过 `/catalog` 获取。`top_k` 范围为 1–50。`context` 是是否补取邻近内容的布尔开关；`explain` 返回检索诊断信息。图片查询可以提供 `image_b64`，或使用客户端的 `--image` 参数。

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

`/image` 在镜像 PDF 版本与切块一致、且图像支持可用时使用 PDF 原图，否则可回退到解析缓存。`X-Image-Source` 表示实际来源。`/crop` 接受 `kb_id`、`point_id`、`bbox` 和可选的 `pad`；坐标使用归一化比例或千分比。先查看图片，再确定裁剪范围，详见 [图片接口参考](../skills/carrel-search/references/api.md)。

### 查询图谱关系

`POST /graph/neighbors` 接受 `kb_id`、实体名称 `entity` 或 `entity_id`，以及可选的 `types`、`direction`、`limit`。同名实体有歧义时，应优先使用已返回的实体 ID。多跳追查由多次请求组成，每一步都需核对相关来源。

- 有向关系的 `in`、`out` 表示关系方向；无向关系可用 `direction=both` 查询两个存储方向。
- 默认返回 20 条，最多 100 条，按权重排序；`count` 记录本次返回的关系数。
- 接口一次返回有数量上限的邻域，可通过 `types` 缩小范围或调整 `limit`。
- 关系描述汇总关联证据，其来源位置可用于 `/context` 原文查询。

## 理解返回证据

| 字段 | 使用方式 |
|---|---|
| `sources` | 原文、来源位置、接受标记及可用的视觉描述 |
| `entities`、`relationships` | 图谱候选与关联，关键关系需回查来源 |
| `specs` | 带单位、条件、时间、来源与冲突标记的结构化事实 |
| `pages` | 编译的主体页、时间线页或来源页，`compiled=true` 表示派生内容 |
| `doc_aggs` | 本次结果涉及的文档 |
| `retrieval_summary` | 路由、选择、证据状态和降级诊断 |

首先检查 `retrieval_summary.evidence_state`：`accepted`、`diagnostic` 或 `unranked`。当 `no_relevant_content=true` 时，返回的候选不足以支持答案。重排不可用时可能返回 `unranked`；服务错误与不可用通道会在诊断字段中单独报告。

事实与页面还需检查 `verified` 和 `sources_active`。冲突标记用于识别记录间的差异；比较测量结果时，可结合主体、单位、日期和条件字段。返回的事实按序列归组。

`text_truncated` 表示可能需要补取原文，`stitched.pieces` 给出拼接片段的位置。来源正文受 `KB_SEARCH_CONTEXT_TOKENS` 限制，编译页面另受 `KB_SEARCH_PAGE_TEXT_TOKENS` 限制，这两个预算独立于 `top_k`。被省略的条件、表头或上下文影响结论时，可调用 `/context` 补取。

不可用阶段通过 `retrieval_summary.degraded` 报告。服务可能退回融合顺序、关键词候选或剩余检索通道；响应会记录可用通道和可引用的来源位置。

## 检索评测

题集与结果放在 `runtime/eval/`，不提交到 Git。准备好题集后执行：

```bash
cd app
.venv/bin/kb search eval --set ../runtime/eval/my-set.json \
  --out ../runtime/eval/result.json --auto
```

`--auto` 使用自动路由，不使用题目预先指定的知识库。评测输出切片与文档命中率、MRR、预期答案检查、文档覆盖、负例表现、延迟、错误，以及相对上次结果的变化。

`kb search make-set` 可调用知识库的抽取模型生成题集初稿。审核问题和预期证据后，可在选定的资料与模型配置上运行评测。
