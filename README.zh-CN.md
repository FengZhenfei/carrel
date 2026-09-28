# Carrel

[English](README.md) | 中文

**Ontology-Augmented Generation for AI agents.** Carrel 是一个本地知识库:把目录里的文档解析、切块、
向量化后写进 Qdrant 和 OpenSearch,可选地为每个知识库建一张按本体分层的实体图(LLM 抽取 → 归并 →
Qdrant 实体 / 关系 / 事实集合 + Neo4j 投影),再通过一个只出证据、不做生成的检索服务把这些资产交给
调用它的 agent。全部由定时任务驱动,配一个局域网内的 Web 控制台。名字取自图书馆里的单人书桌:
一张你的 agent 在你私人藏书里工作的书桌。

一条命令部署:`./deploy.sh` 会探测机器、拉起 MinerU / Qdrant / OpenSearch / Neo4j 四个容器、装好应用。
文本向量、图片描述和建图用的 LLM 只认 OpenAI 兼容地址,本地或公网服务都行。

## 目录

- [设计原则](#设计原则)
- [功能](#功能)
- [架构](#架构)
- [运行环境](#运行环境)
- [快速开始](#快速开始)
- [配置](#配置)
- [使用](#使用)
- [入库管线](#入库管线)
- [实体图](#实体图)
- [定时任务与维护](#定时任务与维护)
- [仓库布局](#仓库布局)
- [开发与测试](#开发与测试)
- [排障](#排障)
- [安全边界](#安全边界)
- [许可](#许可)

## 设计原则

1. **流程全自动化,尽量不要人为干预。** 为此放弃了手工改标签、手工关联实体、手工管理图谱这类功能:
   标签由样本自动抽取,图谱由定时任务增量并入或整库重建,坏了就重建而不是修补。
2. **所有功能高度通用。** 提示词、切块、抽取、归并都不针对某一个知识库的资料特性做倾向性处理,
   代价是产出物会带一些噪声;系统只保证不把噪声当成事实(证据冲突会被标记而不是被抹平)。

审查修复时也按这两条取舍:优先修「代码在改错 / 在丢状态」的问题,不为模型噪声堆逻辑;
无人值守下能自己收敛的现象不额外处理。

## 功能

- **多格式解析。** PDF / DOCX / PPTX 走 MinerU VLM 引擎并附加表格修复与图片描述;
  XLSX / XLS / CSV 走原生表格解析;图片文件(png / jpg / jpeg / webp / bmp / gif / tif / tiff)
  由视觉模型描述;HTML / Markdown / 纯文本 / 配置文件按结构切块;
  代码文件(JavaScript / TypeScript / Go / Java / Rust / C / C++ / C# / PHP / Ruby / Swift /
  Kotlin / Scala / Bash / Lua / PowerShell / Python)用 tree-sitter 按符号切块并抽取调用关系。
- **双向量。** 每个切片带 1024 维 `text` 向量;每张被描述过的图片再带 2048 维 `visual` 向量,
  二者作为同一个点的两个命名向量存进 Qdrant。
- **关键词索引。** 切片同步写入 OpenSearch(内置 `cjk` 分析器,二字词可直接命中)。
- **实体图。** 按知识库可选开启:自动从样本抽取实体类型 / 谓词 / 示例,LLM 抽取实体、关系与
  带单位的测量事实,确定性归并、属性概念归一、证据冲突标记,产物按版本落到 Qdrant 与 Neo4j,
  用别名切换版本;新增 / 变更文档定时增量并入,达到策略条件时整库重建。
- **控制台。** 开库 / 关库、切块与图谱策略、模型注册表、文件与任务进度、切块预览、图谱预览、
  容器健康与重启,中英文切换,原生 JS 无构建链。
- **无人值守。** 定时器全部用相对时间(不读墙钟),维护任务忙时让路并计数,任务租约 / 崩溃回收 /
  取消归属都有所有者校验,建图用 flock 互斥。

## 架构

```text
你的同步工具(rsync / Syncthing / NFS / 手工拷贝,不限)
   ▼
runtime/mirror/<目录>            ← 每个开启的目录 = 一个知识库 kb_NNN
   │  scan.timer(每分钟):新增 / 修改 / 删除 → 任务队列(SQLite)
   ▼
worker.timer(每 5 分钟):解析 → 切块 → 向量化 → 写入
   ├─ MinerU(8765)+ 多模态模型(OpenAI 兼容)  解析、图片描述
   ├─ 文本向量服务(OpenAI 兼容)              1024-d text 向量
   ├─ 跨模态向量服务(可选,本地 vLLM)         2048-d visual 向量
   ├─ Qdrant(6333)                           每库一个 collection
   └─ OpenSearch(9200)                       每库一个关键词索引
   │
   ▼  graph-rebuild.timer(每 2 小时,仅开启图谱的库)
切片 → 抽取单元 → 公网 LLM 抽取 → 归并 / 概念归一 / 事实核对
   ├─ Qdrant 实体 / 关系 / 事实集合(按版本,别名切换)
   └─ Neo4j(7687)投影(按版本)
   │
   ▼
Web 控制台(9800,局域网唯一入口)
```

基础设施由 [`deployment/compose/docker-compose.yml`](deployment/compose/docker-compose.yml) 管理:
MinerU、Qdrant、OpenSearch、Neo4j 四个容器默认启动,五个 vLLM 模型服务在 `local-models` profile 里,
想在本机跑模型的人才开。建图用的 LLM 在控制台的模型注册表里登记,密钥只存在本机的状态库里。

## 运行环境

| 平台 | 能到什么程度 |
|---|---|
| Linux + NVIDIA GPU | 全部功能本地跑;MinerU 走 vlm-engine,模型服务可以本地起 |
| Linux 无 GPU | MinerU 走 CPU 的 pipeline 后端,向量 / 多模态 / LLM 走公网接口 |
| macOS(Docker Desktop) | 容器只能用 CPU,应用原生跑;视觉两路关掉,定时任务用二期的进程内调度或自行安排 |
| Windows | 通过 WSL2 等于 Linux 路径;原生 Windows 不支持 |

- Docker 25+ 与 Compose v2;有 GPU 的 Linux 主机还要 NVIDIA 容器工具包(CDI 或 nvidia runtime 均可)。
- Python 3.12+(管线、控制台与检索服务装在 `app/.venv`)。`deploy.sh --with-pdf-images` 会多装可选的
  `pdf-images`(PyMuPDF,AGPL-3.0),检索服务才能按原分辨率从 PDF 里取图;不装就用解析器缓存的图,见 `NOTICE.md`。
- 文本向量、图片描述、建图三路各一个 OpenAI 兼容接口:公网供应商,或本机 `local-models` profile。
  重排是可选项(`/v1/rerank`,vLLM / TEI / Jina / Cohere 的格式),跨模态向量与重排只有本地 vLLM 提供。
- 资料怎么进 `runtime/mirror/` 由你决定:rsync、Syncthing、NFS、手工拷贝都行,见下文「同步工具的契约」。

## 快速开始

```bash
git clone <this repo> carrel && cd carrel
./deploy.sh
```

`deploy.sh` 做的事(幂等,重跑等于升级):

1. 探测机器:架构、GPU 与容器工具包、内存档位、Docker Hub / PyPI / HuggingFace 是否可达(不可达时自动换镜像源与 ModelScope)。
2. 生成 `deployment/compose/.env`(是否带 GPU、解析镜像口味、镜像名、内存档位、随机 Neo4j 密码)
   和 `config/knowledge-base.env`(已有的文件不覆盖)。
3. 构建解析镜像并拉起四个容器;MinerU 容器首次启动时按它看到的硬件选 vlm-engine 或 pipeline 后端,
   下载对应模型到 `models/mineru/`。
4. 建 `app/.venv`、安装应用、初始化状态库;Linux 上装好并启用 systemd 用户单元。

跑完打开它打印的控制台地址(默认 `http://127.0.0.1:9800`,局域网访问见「安全边界」)。把资料目录放到 `runtime/mirror/<目录名>/`,在控制台「开启」它,
系统分配编号 `kb_NNN`,扫描与解析随即自动进行。要建图的库先在顶栏注册一个 LLM,再在配置页打开「图谱」开关,
标签抽取、建图、增量并入都由定时任务接手。

`./deploy.sh detect` 只打印探测结果;`--with-local-models` 顺带起五个本地模型服务(权重先放到 `models/`);
`--cpu` 强制 CPU 口味;`down` 停容器;`purge [--images] [--data]` 卸载。没有 systemd 的机器
(macOS)脚本会打印手动启动控制台、检索服务和一轮入库的命令。

### 同步工具的契约

Carrel 只管 `runtime/mirror/` 之后的事,文件怎么到那里不限:

1. 内容放进 `runtime/mirror/<目录名>/`(`KB_MIRROR_ROOT` 可改),一个目录一个库。
2. 扫描不碰落地不到 `KB_MIN_FILE_AGE_SECONDS`(默认 180 秒)的文件,写到一半的文件不会被误收。
3. 同步工具想让扫描完全让路,推送期间持有目录 `runtime/state/mirror_sync.lock.d` 即可(可选)。
   软链接不会被跟到目录之外;想把别处的文件夹挂成一个库,设 `KB_MIRROR_ALLOW_LINKED_DIRS=1`。
4. mirror 是事实源:文件消失走软删除加保留期;整库突然清空由 `KB_MASS_DELETE_GUARD_*` 挡住,
   用 rsync `--delete` 的人要知道这一点。

## 配置

### 全局配置 `config/knowledge-base.env`

模板是 [`config/knowledge-base.env.example`](config/knowledge-base.env.example),按组说明:

| 组 | 主要键 | 说明 |
|---|---|---|
| 根目录 | `KB_LOCAL_BASE_DIR` `KB_MIRROR_ROOT` `KB_STATE_DB` `KB_RUNTIME_DIR` `KB_CACHE_DIR` `KB_LOG_DIR` `KB_GRAPH_WORK_DIR` | 仓库、镜像目录、状态库(SQLite,WAL)、运行态、解析缓存、日志、建图工作区 |
| 存储 | `QDRANT_URL` `OPENSEARCH_URL` `NEO4J_URI` `NEO4J_USER` `NEO4J_PASSWORD` | 全部 loopback;Neo4j 密码只在这个 600 权限的文件里 |
| 解析 | `MINERU_SERVICE_URL` `MINERU_BACKEND=auto` `MINERU_LANG` `KB_PARSE_ENABLED` `KB_MIN_FILE_AGE_SECONDS` `KB_JOB_MAX_RETRIES` `KB_JOB_RETRY_*` `KB_*_JOB_LEASE_SECONDS` | `MINERU_BACKEND=auto` 用解析容器自己按硬件选的后端;`KB_PARSE_ENABLED=0` 是总闸,worker 会把解析任务挂起;文件写入后要静置 `KB_MIN_FILE_AGE_SECONDS` 才入队 |
| 保留期 | `QDRANT_INACTIVE_RETENTION_DAYS` `KB_CACHE_ROTATION_KEEP` `KB_LOG_ROTATION_KEEP_MONTHS` `KB_VLM_CACHE_MAX_AGE_DAYS` | 软删点的撤销窗口(默认 7 天)、缓存 / 日志轮换份数、图片描述缓存寿命 |
| 文本向量 | `EMBEDDING_*` | 任意 OpenAI 兼容 `/v1/embeddings`;默认 qwen3-embedding-0.6b @ 1024 维,维度建库后不能改 |
| 图片描述 | `VLM_*` `KB_IMAGE_MAX_PIXELS` | 任意 OpenAI 兼容多模态 chat 接口;像素预算要与模型允许的 `max_pixels` 一致 |
| 视觉向量 | `VISUAL_EMBEDDING_*` | 可选,只有本地 vLLM 提供(`messages` 形态请求);默认关,开了是 qwen3-vl-embedding-2b @ 2048 维 |
| 重排序 | `RERANKER_BASE_URL` `VISUAL_RERANKER_BASE_URL` | 检索侧可选;入库不调用 |
| 控制台纳管 | `KB_CONSOLE_SERVICES` | 服务状态里探活与重启的行,默认 `database,mineru`;本机跑模型服务时再加上模型行 |
| 控制台 | `KB_WEB_HOST` `KB_WEB_PORT` | 默认 `0.0.0.0:9800` |
| 图谱 | `QDRANT_GRAPH_COLLECTION_RETENTION_DAYS` `NEO4J_GRAPH_RETENTION_DAYS` `GRAPH_GC_KEEP_VERSIONS` `GRAPH_NEO4J_IMPORT_*` `KB_GRAPH_LLM_CONCURRENCY` `KB_GRAPH_LLM_TIMEOUT` `KB_GRAPH_CIRCUIT_FAILS` | 旧版本保留 14 天、每库保留 2 个版本;公网 LLM 并发、单次超时、连续失败熔断 |

向量维度在建 collection 时写死,改 `EMBEDDING_DIM` / `VISUAL_EMBEDDING_DIM` 意味着重建全部 collection。

### 单库配置(控制台「配置」页)

每个知识库一行 `kb_sources` 记录,策略存在 `config_json` 里,只在控制台改:

| 组 | 键 | 说明 |
|---|---|---|
| 切块 | `max_tokens` `overlap_tokens` | 上限由文本向量服务的上下文决定,控制台会给提示 |
| 图谱开关 | `graph_enabled` `graph_auto_append` `graph_profile` | 开关、是否允许定时增量并入、场景画像 |
| 标签 | `graph_entity_types` `graph_parent_types` `graph_type_definitions` `graph_predicates` `graph_examples` `graph_language` | 由「抽标签」按 `graph_tune_sample_size` 个样本自动生成,可选一个历史版本生效;不建议手改 |
| 抽取 | `graph_unit_chunks` `graph_max_gleanings` | 每几个连续切片合成一个抽取单元(默认 3,1 = 不合并);每单元最多补抽几轮 |
| 模型 | `graph_llm.extract` `graph_llm.summarize` `graph_llm.tune` | 每一步各选注册表里的一个模型 |
| 重建策略 | `graph_rebuild_interval` `graph_rebuild_new_chunk_pct` `graph_rebuild_new_chunk_count` `graph_rebuild_operator` | 距上次整库重建满 N 天,或新切片占比 / 条数超阈值(`or` / `and`)则整库重建,否则增量并入 |

`graph_chunk_size`、`graph_mode` 等旧键服务端静默丢弃(摘要树模式已于 2026-09-05 删除,建图只有实体图一种)。

## 使用

### 控制台

三个标签页,侧栏列出全部知识库与统一进度条:

- **配置**:切块参数、图谱开关与标签、每步模型、重建策略;按钮有「保存配置」(只写策略)、
  「立即解析」(踢一轮扫描与 worker)、「整库重新解析」、「抽标签」、「立即建图」、「增量并入」、
  「暂停建图」、「删除图谱」、「关闭知识库」。
- **文件**:每个文件的解析状态、切片数、失败原因、重试;抽屉里可以看切块预览与任务时间线。
- **图谱**:当前版本的实体 / 关系预览,可按实体类型筛选、看归并记录。

顶栏是模型注册表(增删公网 LLM,密钥只写不回显)、容器健康(逐个重启 / 全部停止 / 全部重启)、
中英文切换。「保存配置」只写策略,切块参数改了要点「整库重新解析」才会生效;
解析器版本号变化时受影响的文件自动重排,不需要手动重解析。

关闭知识库后进入保留期(默认 7 天),到期由 GC 彻底删除;期间重新开启即恢复。
目录改名会自动沿用原编号;识别不出来时可以在控制台手动「沿用」旧库。

### 命令行

`app/.venv/bin/kb`(下文简写 `kb`),配置从 `config/knowledge-base.env` 读:

| 命令 | 用途 |
|---|---|
| `kb config` / `kb status` / `kb health` | 打印生效配置、队列与知识库状态、各服务健康 |
| `kb init-db` | 建 SQLite 状态库 |
| `kb scan [--source kb_NNN] [--requeue-failed] [--rehash]` | 扫描镜像目录并排队(定时器每分钟自动跑) |
| `kb worker --once --max-jobs N --max-seconds S` | 消费队列(定时器自动跑;**不要**与 systemd worker 并行手动起第二个) |
| `kb qdrant ensure-collections` / `ensure-graph-collections` | 补建集合 |
| `kb fts init` / `status` / `rebuild` / `sync-doc` / `search` | OpenSearch 索引维护与试搜 |
| `kb graph build` / `append` / `check-rebuild [--execute] [--force-full]` | 整库建图、增量并入、按策略检查(定时器用的就是 check-rebuild) |
| `kb graph adopt-current` / `neo4j-import` / `neo4j-status` / `neo4j-delete` | 版本基线、Neo4j 投影 |
| `kb graph query` / `factcheck` / `status` | 图召回原型、事实级对照评测、建图状态 |
| `kb cleanup status` / `weekly` / `monthly` / `qdrant-gc` / `qdrant-graph-gc` / `neo4j-graph-gc` / `parse-assets-gc` | 各种 GC(定时器自动跑) |
| `kb reset --source kb_NNN --yes` | 清空某库的状态 / 缓存 / 索引并重建 collection(不带 `--yes` 只打印计划) |

整库推倒重来(三条命令即可;第一条不带 `--yes` 只打印计划):

```bash
cd app
./.venv/bin/kb reset --all --yes --force      # 清空所有库的状态 / 解析缓存 / 索引 / 图谱,重建 collection
./.venv/bin/kb scan --verbose                 # 按镜像重新登记文件并排队解析
systemctl --user start --no-block carrel-worker.service   # 踢一轮 worker;之后由定时器接管解析、建图
```

按需解析某个库,用「扫描 + 踢一次 worker 单元」而不是手动起 worker:

```bash
kb scan --source kb_001 && systemctl --user start --no-block carrel-worker.service
```

### HTTP 接口

控制台前端用的接口都在 `app/kb_server/`,前缀 `/api`:`overview`、`health`、`limits`、`enroll`、
`kbs/{kb_id}/…`(`config`、`files`、`jobs`、`parse_now`、`reparse`、`graph_build`、`graph_append`、
`graph_pause`、`graph_schema`、`graph_preview`、`graph_merges`、`graph_builds`、`chunk_preview`、
`unenroll`、`adopt`、`graph` 删除)、`jobs/{id}/cancel`、`files/retry`、`services/{key}/restart`、
`llms`。写接口要求同源。

## 入库管线

1. **扫描**(`scan.timer`,每分钟)。只处理控制台开启过的目录;新文件要静置
   `KB_MIN_FILE_AGE_SECONDS` 才入队;按内容校验和与大小判变更,只改时间戳不重解析;
   文件删除只是软删(保留期内重新出现即恢复),全部已开启目录同时消失时不会自动下线知识库。
   同步工具持有 `runtime/state/mirror_sync.lock.d` 时跳过本轮。
2. **解析**(`worker.timer`,每 5 分钟,带时间预算)。每个任务有租约、所有者与重试退避,
   崩溃的任务由下一轮回收,取消优先于重试。解析器带版本号(当前 PDF 档案是
   `pdf-mineru-table-vlm-v13`),版本变化时受影响的文件自动重排。
3. **切块**。按块结构切,表格、列表、跨页表格片段会先合并,再按 `max_tokens` / `overlap_tokens` 切;
   代码按符号切。切片写主库(SQLite)。
4. **向量化与写入**。送去嵌入的文本是「文档名 > 章节路径」前缀加正文,负载里的 `text` 仍是原正文;
   图片先由 VLM 描述再算 `visual` 向量(描述与向量都有跨库缓存);
   点写 Qdrant(旧版本软删,7 天后 GC 硬删),同步写 OpenSearch。
5. **解析产物**。MinerU 的中间产物与图片存在 `runtime/parse_cache/`,夜间 GC 只留现行版本用到的。

## 实体图

建图直接读主库切片,没有独立的建图语料投影:

1. **抽标签**。从样本切片让 LLM 建议实体类型、父类(六类上层本体)、谓词与示例,存成一个标签版本;
   建图前自动跑一次,有人抢先时后来者被判「已被取代」而不是覆盖。
2. **抽取**。每 `graph_unit_chunks` 个连续切片合成一个抽取单元;LLM 输出实体、关系、带单位的测量事实。
   每个单元的结果按(库、单元、配置指纹)缓存在状态库 `graph_extractions` 里,语料没变的单元下次一次
   LLM 都不调;格式不对的回复会带纠错提示重试,三次仍不对才记为空。
3. **归并**。确定性归并同名 / 别名实体(别名一律二道复核),属性概念归一(单位别名表 `measure_units.py`),
   同一实体同一属性的值互相矛盾时标记 `evidence_conflict` 而不是二选一;时间序列按画像主体归位。
4. **落库**。实体 / 关系 / 事实向量按版本写 Qdrant 集合(`confidence`、`evidence_conflict`、`cmps`、
   `comparable` 随 payload 走),再导入 Neo4j;都成功后切别名,旧版本按保留期 GC。
5. **增量并入**(`graph-rebuild.timer`,每 2 小时)。只抽新增 / 变更文档的单元,回放上一版的归并结论
   (包括挡下的对),沿用没变的向量;删除的文档随之下线。达到重建策略条件时改走整库重建,基线是上一次整库重建。
   不做「只处理增量文档」的真增量:归并、共现权重、推边、谓词体检都是整库统计,删改文档要做反向操作,
   而这些确定性部分整库重算只要几十秒,不值得引入撤销逻辑;并入产出的是完整的新版本,再原子切别名。

工作区 `runtime/graph/`:`work/<库>/<版本>/` 放单元与合并后的图,`cache/<库>.sqlite` 是 LLM 响应缓存。
建图互斥用 `runtime/state/graph_build.lock.d/lock` 上的 flock,锁随进程消亡自动释放,
控制台「暂停建图」会等确认进程退出后才放行。

## 定时任务与维护

| 单元 | 节奏 | 做什么 |
|---|---|---|
| `carrel-scan.timer` | 启用后 30 秒,之后每分钟 | 扫描目录,发现新增 / 修改 / 删除并排队 |
| `carrel-worker.timer` | 启用后 1 分钟,之后每 5 分钟 | 消费队列:解析 → 切块 → 向量化 → 写入 |
| `carrel-graph-rebuild.timer` | 启用后 10 分钟,之后每 2 小时 | 增量并入;达到策略条件整库重建 |
| `carrel-qdrant-gc.timer` | 启用后 30 分钟,之后每 24 小时 | 点级 GC、失活知识库到期硬删、旧图版本清理、任务历史清理 |
| `carrel-cache-weekly.timer` | 启用后 1 小时,之后每 7 天 | 缓存轮换(整机的 uv / pip / Docker 清理只在 `KB_HOST_HOUSEKEEPING=1` 时做) |
| `carrel-logs-monthly.timer` | 启用后 2 小时,之后每 30 天 | 日志轮换 |
| `carrel-web.service` | 常驻 | 控制台 |

定时器全部是相对时间(`OnActiveSec` / `OnUnitActiveSec`),不读墙钟、不看时区:换机器、重置、随时启用
都是同一套节奏。维护类任务撞上系统繁忙时让路(退出码 75),连续让路超过上限才把单元标成失败,
策略在 `scripts/lib/kb-maint-defer.sh`。全部单元设有启动超时。

自动轮转覆盖的东西不需要手动清:软删点 7 天、图版本 14 天、任务与入库历史 30 天、
解析产物与图片描述缓存按夜间 / 每周 GC。手动维护只剩两件:`backups/`(重置前的注册表快照,含密钥)
和 `runtime/state/` 下的锁目录(不要删)。

## 仓库布局

```text
app/
  kb_pipeline/        管线包(CLI 入口 kb_pipeline.cli:main)
    pipeline/         scan / worker / parse_job / 状态迁移
    parsers/          router、MinerU(pdf/docx/pptx)、native_table、html_dom、code_symbols、visual_blocks …
    chunking/  embedding/  vector/  vision/  localfs/
    graph/            build、extract、facts、concepts、reconcile、compile、temporal、lock、neo4j_import …
    measure_units.py  单位别名与规范化(解析侧与建图侧共用)
    maintenance.py    GC、停建图、删图、服务重启
  kb_server/          FastAPI 控制台(app.py / service.py / static/{index.html,app.js,i18n.js})
  tests/              按模块分文件:test_parsers / chunking / scanner_worker / state_config / console_api / console_service /
                      graph_schema / graph_build / graph_extract / graph_facts / llm_client / maintenance_ops / search;_support.py 是公用夹具
  kb_search/          检索服务(只出证据,不做生成)
deploy.sh             一条命令部署 / 探测 / 停止 / 卸载
config/               knowledge-base.env(.example)
deployment/
  compose/            四个基础容器 + 可选本地模型服务的 compose 工程、解析镜像(两种口味)与它的入口脚本
  systemd/            用户级 timer / service 单元(模板,install-systemd.sh 渲染成本机路径)
scripts/              各 timer 的包装脚本(scan / worker-once / graph-rebuild-check / cleanup)、install-systemd.sh、lib/
runtime/(不入库)    mirror/ state/ graph/ parse_cache/ qdrant_data/ neo4j/ opensearch/ mineru-output/
models/(不入库)     解析模型与本地模型服务的权重
logs/、backups/(不入库)
LICENSE、NOTICE.md    MIT 与第三方声明
```

## 检索服务(kb_search)

召回层做成常驻服务 `app/kb_search/`(systemd `carrel-search.service`,默认端口 9810,静态 Bearer `KB_SEARCH_TOKEN`),
策略全在服务端,客户端只把问题原样交过来,服务只返回证据、不生成答案;问题理解与答案组织都在调用方的 agent 里做,
服务端的查询路径只有打分模型(嵌入 8101、交叉编码器 8102、跨模态嵌入 8103),没有任何生成步骤。

给 agent 用的客户端在 [`skills/carrel-search/`](skills/carrel-search/SKILL.md):一个只依赖标准库的 Python 脚本加一份 SKILL.md,装进 Claude Code / Codex 这类 agent 后,agent 自己理解问题、按需补查并组织回答;默认地址 `http://127.0.0.1:9810`,地址与 token 用 `CARREL_SEARCH_BASE_URL` / `CARREL_SEARCH_TOKEN` 或 `~/.config/carrel-search/config.json` 指定。

| 接口 | 作用 |
|---|---|
| `GET /health` | 免鉴权;库列表、鉴权方式、Qdrant 连通性 |
| `GET /catalog` | 知识库目录:名字、领域、主体类型、类型表、规模、文件名样本、有没有图(没开图谱的库照样有目录项);`?refresh=1` 重建 |
| `POST /search` | `{question, kbs?, top_k?, hints?, context?, explain?, image_b64?}` → 编号的 Sources / Entities / Relationships / Specs / Pages、doc_aggs、retrieval_summary |
| `POST /context` | `{kb_id, doc_id, content_version?, chunk_from, chunk_to}` → 某文档序号区间的切片(追问用) |
| `GET /image/{kb_id}/{point_id}` | 图片切片的原图:镜像 PDF 的 sha256 与切片的 content_version 对得上才用它(嵌图按摆放位置匹配优先,其次按 bbox 高清渲染),否则解析缓存;失活的点不给;响应头 `X-Image-Source` 说明来源 |
| `POST /crop` | `{kb_id, point_id, bbox, pad?}` → 按调用方给的框(0–1 比例或 0–1000 千分比)从原图裁出局部,服务端 只做确定性裁剪 |
| `POST /graph/neighbors` | `{kb_id, entity 或 entity_id, limit?, types?, direction?}` → 一个实体在现行图谱里的一跳关系:谓语、方向、权重、对端实体、证据切片(可直接喂 /context);同名多个走关系最多的、其余在 matches,找不到给向量候选。多跳由 agent 逐步走 |

一次查询:向量路 + BM25 路在所有库上并行探测(五库几十毫秒)→ 按证据选库(调用方指定了 `kbs` 就不路由)→
图路(一跳扩展;0 / 1 / 2 / 3 / 5 跳实测候选集与指标相同、只差延迟,多跳关联交给 agent 用 `/graph/neighbors` 逐步走)、视觉路只在选中的库上跑(没开图谱的库自然没有图路)→ 每库 RRF 融合、多主体 / 多文档分桶交错 → 8102 交叉编码器重排
(阈值 0.1、地板 0.07)→ 最终分 = 重排分 × 0.8 + 问题记号覆盖率 × 0.2(重排分在表格库里成片饱和到 0.99,覆盖率分出同一张表的哪一行),
样板片、低置信图片再降权 → 名额保底 + MMR(相似度不算表头行)选出 top-k、按最终分呈现 → 短片拼邻居、表格续片带表头、邻域回填、
重叠去重、token 预算 → 证据包。任一环节不可用都只降级并写进 `retrieval_summary.degraded`(重排挂退回融合序;嵌入挂
只走关键词路 + 图路词法种子;Neo4j 挂跳图路;OpenSearch 挂跳关键词路;都带 `low_confidence`)。

库路由不靠画像、图谱或任何题材假设,新建什么主题的库、开不开图谱都一样:每个库的证据 = 向量路前 3 片余弦均值
与库级词法证据(问题按 cjk 分析器切成记号,数每个记号在各库的出现密度,按跨库稀有度加权;BM25 原分跨索引不可比,
题材词在本题材的库里 IDF 反而最低)按 7:3 合成,与最高分差在 `KB_SEARCH_ROUTE_GAP`(0.2)内的库一起查,最多
`KB_SEARCH_ROUTE_MAX_KBS`(2)个;所有库向量证据都低于 `KB_SEARCH_ROUTE_FLOOR`(0.45)只标 `routing.weak`。
选中的库拿不出像样的证据(没有候选,或重排分全在阈值之下)时自动放宽到全库再排一次(`routing.widened`)。

证据里的事实带线索串(「主体 · 属性 = 值 单位 @ 条件 · 时间」)、来源切片编号、同序列的其它值与冲突标注;编译页
(主体页 / 时间线页 / 来源页)标 `compiled=true`,正文只给时间线页与带序列行的主体页、按预算截;图片切片带视觉描述、
图中文字、事实与冲突标注,可再取原图或裁图。所有参数在 `app/kb_search/config.py`,每个数值旁写了来源与标定依据。

调用方要读的几个状态字段:`retrieval_summary.evidence_state` 是 accepted / diagnostic / unranked
三态,全部候选都低于重排地板时是 diagnostic,此时 `no_relevant_content = true`,Sources 照样返回但每条 `accepted = false`,
事实与页面带 `state = diagnostic`,只能当诊断线索;重排挂了是 unranked。事实和页面的来源点会核一遍活跃状态
(`verified` / `sources_active`),来源全失活的事实不给线索;实体、关系、事实带稳定 `id`,`graph_versions` 记本次用的图版本。
短片拼邻居后 `stitched.pieces` 列出每一段的 point_id、序号、页码,逐段可引用。Sources 的正文总量受 `KB_SEARCH_CONTEXT_TOKENS`
硬约束,超出的命中片截成带位置的摘录并标 `text_truncated`,全文用 `/context` 取;编译页正文另按 `KB_SEARCH_PAGE_TEXT_TOKENS`
计,`sources.page_tokens` 单独报。带图的请求视觉路在全库探测并参与选库,视觉路候选不受文字重排阈值约束、按视觉分保底
`KB_SEARCH_VISUAL_QUOTA` 条。近似去重只在同文档同版本、序号相差不超过两片之间做,数字(含正负号、小数点)对不上的不算重复,
不同文档、不相邻的相似句都留。Sources 正文合计严格不超过预算:每条命中先预留一个保底摘录额度(80 token 与「预算 / 条数」取小),
剩余按序分,拼接片被截时 `stitched.pieces` 标出哪些段还在。
`hints` 只收四个通用、可确定性执行的键:`doc_ids` / `rel_paths` / `content_version` 是硬过滤(推进向量路、关键词路、视觉路的
查询过滤,图路候选回填后按同样的限定筛;派生证据同样受限 —— 事实要有范围内的活跃来源点,跨文档编译的页面只要有范围外来源就不附加,
实体按来源文档筛,关系一律不附加,丢掉的数量在 `hints_scope` 里),`block_types` 是软偏好(命中的切片最终分乘 `KB_SEARCH_HINT_BLOCK_BOOST`,1.2);
别的键忽略并列在 `hints_ignored`,`hints_used` / `hints_applied` 说明实际生效的部分。主体、时间这类靠原文核对的不做硬过滤。
查询侧任务指令 `KB_SEARCH_QUERY_INSTRUCTION` 默认留空:2026-09-16 用七份题集消融过,加与不加互有一两题的进退、延迟相同,没有一致收益。

```bash
curl -s -H "Authorization: Bearer $KB_SEARCH_TOKEN" -X POST http://<host>:9810/search \
  -H "Content-Type: application/json" -d '{"question": "这款芯片的工作温度范围是多少", "kbs": ["kb_002"], "top_k": 8}'
```

回归评测(题集与结果都在 `runtime/eval/`,不入库):

```bash
cd app && .venv/bin/kb search eval --set ../runtime/eval/synth-products.json --out ../runtime/eval/search-eval-synth-products.json --auto
```

`eval` 打印 hit@3/5/12(切片金标,严格按命中片算)、doc_hit@3/5/12(文档金标)、MRR、expect / expect_all、min_docs、
负例(服务是否把它标成无相关内容,不等于 agent 已拒答)、平均延迟与报错数(报错的题计入分母),并与上一次同名结果比差值;
`--auto` 无视题目自带的库让服务自己路由。2026-09-16 定参后的基线(自动路由,切片 hit@5 / MRR):products 11/15 0.70、reports 4/4 1.0、
library 14/15 0.94、projects 12/15 0.80,文档 hit@5 全满,负例 8/8,健康 / 报告跨文档题集全过;图路跳数 0 / 1 / 2 质量相同、只差延迟,取 1。`kb search make-set --kb kb_00N --n 15 --out …` 从库里随机抽切片、用本库的抽取模型造题
(金标 = 切片 point_id,只在开发期跑)。现有题集:`synth-{products,reports,library,projects}.json`(49 题)、
`health-crossdoc.json`(10)、`reports-crossdoc.json`(3)、`negatives.json`(8 道库外负例)。

## 开发与测试

```bash
cd app && .venv/bin/python -m pytest -q        # pyproject 里配了 testpaths 与 -p no:cacheprovider
```

- 回归测试不依赖容器,跑完不到一分钟;需要真调 LLM 的用例只用便宜的 Flash 档模型。
- 改了解析器输出就把 `parsers/common.py` 里的档案版本号加一,存量文件会自动重排。
- 控制台文案要同时加进 `static/i18n.js` 的中英字典,`ConsoleI18nTests` 会拦;数据(目录名、文件名、实体名)不翻译。
- 只改 `static/` 不用重启服务;改了 Python 要重启 `carrel-web.service`,
  重启前先看 `journalctl --user -u carrel-web.service --since -5min`,抽标签 / 切块预览是在 web 进程里同步跑的,
  重启会把它们杀掉。
- 改了 `deployment/systemd/` 必须重跑 `./scripts/install-systemd.sh`(`--check` 只比对),否则 git 里的与实际运行的会静默漂移。
- 状态库是 `runtime/state/kb-pipeline.db`;路径写错会静默建出一个空库。

## 排障

```bash
systemctl --user list-timers "knowledge-base*"                 # 下一次触发
journalctl --user -u carrel-worker -f                  # 看解析
journalctl --user -u carrel-graph-rebuild --since -3h  # 看建图 / 并入
curl -s localhost:9800/api/health | python3 -m json.tool       # 各服务健康
app/.venv/bin/kb status                                        # 队列与库状态
app/.venv/bin/kb graph status                                  # 各库建图状态与版本
```

- **文件一直排队不解析**:确认 `KB_PARSE_ENABLED=1`;看 worker 日志有没有「让路」或锁;
  `runtime/state/mirror_sync.lock.d` 存在说明同步工具正在推送。
- **建图被拒「lock already exists」**:另一次建图还在跑。锁是 flock,持有进程一退出就自动失效;
  锁目录里的 `pid` / `started_at` 只是显示用的残留,不要手删锁目录。
- **维护单元显示 failed 且退出码 75**:连续多轮让路,说明系统一直忙;等入库高峰过去会自己恢复。
- **图片全部失败**:多模态或跨模态向量服务不健康;临时置 `VISUAL_EMBEDDING_ENABLED=0` 可只入文本。
- **Neo4j 数据损坏**:停容器,用 neo4j 镜像以 root 删 `runtime/neo4j/data`,起容器,`kb graph neo4j-import` 重投当前版本。

## 安全边界

- **控制台 9800** 默认只监听本机。要在别的设备上用,设 `KB_WEB_HOST=0.0.0.0` 并配 `KB_WEB_TOKEN`,
  之后每个接口调用都要带这个 token,页面每个浏览器问一次。浏览器的写操作另外限定在控制台自己的 origin
  (协议、主机、端口都要一致)。不要把控制台暴露到公网:它能读全部切片、能停服务。
- **检索服务 9810** 靠静态 Bearer token,不设 token 时只接受本机请求。
- **模型密钥** 存在本机状态库里,接口不回显;改模型的地址、端口或协议时必须重填 Key,存着的 Key 不会
  被误发到新地址;带 Key 的请求不跟随重定向。
- **哪些内容会出机**:正文、切片和图片会发给你配置的向量、图片描述和建图接口。用公网供应商就是把内容
  经网络交给它们;请用 HTTPS 地址和你信得过的供应商。
- 模型服务、MinerU、Qdrant、Neo4j、OpenSearch 全部绑定 loopback。密钥只存在本机:
  `config/knowledge-base.env`、`deployment/compose/.env`、状态库,都不入库。
- **镜像目录边界**:指向已开启目录之外的软链接一律跳过;顶层目录本身是软链接的默认不认,
  `KB_MIRROR_ALLOW_LINKED_DIRS=1` 才认。
- 跨网络访问建议走 Tailscale 一类的私有网络,不做端口转发。

## 许可

MIT,见 [`LICENSE`](LICENSE);第三方组件与提示词来源见 [`NOTICE.md`](NOTICE.md)。
