# 配置说明

[首页](../README.zh-CN.md) · [English](configuration.md) | 中文

## 配置文件

`deploy.sh` 会生成以下两个文件，后续运行时保留已有配置：

| 文件 | 用途 |
|---|---|
| `config/knowledge-base.env` | 应用路径、模型接口、凭据和处理参数 |
| `deployment/compose/.env` | 容器镜像、解析后端策略、资源限制和本地模型服务 |

可参考 [应用配置模板](../config/knowledge-base.env.example) 和 [Compose 配置说明](../deployment/compose/README.md)。生成的文件包含凭据，已由 Git 忽略规则排除。应用优先读取 `config/knowledge-base.env`，不存在时兼容读取 `app/.env`；也可通过 `KB_ENV_FILE` 指定其他文件。

修改环境配置后，应重启使用该配置的常驻应用服务。Linux 上，控制台和检索服务分别是 `carrel-web.service`、`carrel-search.service`；新启动的扫描和 worker 进程会读取更新后的文件。正在处理的任务应先完成，再安排重启，详见 [运维说明](operations.zh-CN.md)。检索服务自己的 `KB_SEARCH_*` 参数是例外：服务每分钟重读一次配置文件，改参数、换令牌都不用重启；监听地址和端口，以及文件里其他部分的参数（如各模型接口地址）仍需重启，服务进程环境里显式设置的参数优先于文件。

## 模型接口

默认部署启动存储服务和 MinerU。通过 `--with-local-models` 可额外启动五个向量、重排与视觉模型服务。

| 能力 | 配置 | 使用条件 |
|---|---|---|
| 文本向量 | `EMBEDDING_BASE_URL`、`EMBEDDING_MODEL_ID`、`EMBEDDING_API_KEY`、`EMBEDDING_DIM` | 入库和向量检索需要 |
| 图片描述 | `VLM_BASE_URL`、`VLM_MODEL_ID`、`VLM_API_KEY`、`KB_IMAGE_MAX_PIXELS` | 处理图片和文档插图时需要；图片送出前先缩到像素预算以内 |
| 图谱抽取与摘要 | 在控制台注册对话模型，再为知识库选择 | 启用建图时需要 |
| 文本重排 | `RERANKER_BASE_URL` | 可选，未配置时使用检索融合顺序 |
| 视觉向量 | `VISUAL_EMBEDDING_ENABLED`、`VISUAL_EMBEDDING_BASE_URL`、`VISUAL_EMBEDDING_MODEL_ID`、`VISUAL_EMBEDDING_DIM` 及其他 `VISUAL_EMBEDDING_*` | 可选，用于像素层面的检索；配套实现连接本地 vLLM |
| 视觉重排 | `VISUAL_RERANKER_BASE_URL` | 可选，用于视觉结果重排 |

文本向量使用 OpenAI 兼容的 embeddings API，图片描述和建图模型使用兼容的 chat API。重排使用 `/v1/rerank`，视觉向量使用 vLLM 的多模态 pooling 请求格式；各接口需使用对应的请求格式。

使用配套模型服务时，先按 [本地模型部署说明](../deployment/compose/README.md#local-model-servers-profile-local-models) 下载权重，再运行 `./deploy.sh --with-local-models`。建图用的对话模型仍需在控制台单独配置。

应在首次入库前选定向量模型和维度。模板中文本向量为 1024 维（`EMBEDDING_DIM`），视觉向量为 2048 维（`VISUAL_EMBEDDING_DIM`）。修改维度需要重建受影响的集合；更换模型时，即使维度相同，也需要重新生成资料向量。

## 资料目录

将每组资料放入 `runtime/mirror/<目录名>/`，也可通过 `KB_MIRROR_ROOT` 指定其他位置。在控制台启用目录后，系统为其分配稳定的 `kb_NNN` 编号，并开始处理文件。

资料可通过 rsync、Syncthing、挂载文件系统或手动复制同步到镜像目录。

- 新文件未达到 `KB_MIN_FILE_AGE_SECONDS` 时暂缓处理，模板为 180 秒。长时间上传可配合原子移动或同步锁。
- 同步程序可在复制期间持有 `runtime/state/mirror_sync.lock.d`，让扫描等待；同步完成后释放锁。
- 文件消失后会进入索引删除流程，失活数据按保留策略清理；文件重新出现时可以恢复索引状态。
- 所有已登记目录同时消失时（镜像目录被卸载或清空），扫描器拒绝下线处理；单个知识库内的文件删除会照实处理，使用 `rsync --delete` 时要留意。
- 顶层软链接目录需显式设置 `KB_MIRROR_ALLOW_LINKED_DIRS=1`。库内文件链接不能超出获准的根目录；被拒绝的根目录仍保留登记，并在控制台显示受限状态。

## 全局设置

| 类别 | 主要参数 | 说明 |
|---|---|---|
| 路径 | `KB_LOCAL_BASE_DIR`、`KB_MIRROR_ROOT`、`KB_STATE_DB`、`KB_RUNTIME_DIR`、`KB_CACHE_DIR`、`KB_LOG_DIR`、`KB_GRAPH_WORK_DIR` | 默认以项目安装位置为基准 |
| 存储 | `QDRANT_URL`、`QDRANT_API_KEY`、`OPENSEARCH_URL`、`NEO4J_URI`、`NEO4J_USER`、`NEO4J_PASSWORD` | 与实际部署的服务对应 |
| 解析 | `MINERU_SERVICE_URL`、`MINERU_BACKEND`、`MINERU_LANG`、`KB_PARSE_ENABLED`、`KB_MIN_FILE_AGE_SECONDS` | `MINERU_BACKEND=auto` 读取解析容器选定的后端 |
| 任务 | `KB_JOB_MAX_RETRIES`、`KB_JOB_RETRY_BASE_SECONDS`、`KB_JOB_RETRY_MAX_SECONDS`、`KB_PARSE_JOB_LEASE_SECONDS`、`KB_METADATA_JOB_LEASE_SECONDS` | 重试和任务租约。任务连不上所需的服务时退回队列，不计重试次数：先等基础间隔，之后每次加倍，不超过上限；这样退回达到 `KB_JOB_MAX_RETRIES` 次后记为失败 |
| 保留期 | `QDRANT_INACTIVE_RETENTION_DAYS`、`KB_CACHE_ROTATION_KEEP`、`KB_LOG_ROTATION_KEEP_MONTHS`、`KB_BACKUP_KEEP` | 删除的撤销窗口（模板为 7 天）、缓存与日志轮换的保留份数、每夜状态备份的保留份数（7）。图片描述和视觉向量缓存不按时间过期，解析缓存里已经没有对应图片时，才由每周清理删除 |
| 图谱 | `GRAPH_GC_KEEP_VERSIONS`、`GRAPH_GC_GRACE_SECONDS`、`QDRANT_GRAPH_COLLECTION_RETENTION_DAYS`、`NEO4J_GRAPH_RETENTION_DAYS`、`GRAPH_NEO4J_IMPORT_*` | 每库保留现行版加 N−1 个旧版，不看天数；最新一次暂停 / 失败的版本留给续跑，不删也不占名额（回退用 `kb graph rollback --source <key> --graph-version <旧版>`：目标必须是建成过的版本，集合点数要同建成时对得上，半截版本加 `--force` 才切）；两个保留天数键只对手工的按天清理命令生效；Neo4j 导入 |
| 建图模型 | `KB_GRAPH_LLM_CONCURRENCY`、`KB_GRAPH_LLM_TIMEOUT`、`KB_GRAPH_CIRCUIT_FAILS` | 并发、超时与熔断 |
| 控制台 | `KB_WEB_HOST`、`KB_WEB_PORT`、`KB_WEB_TOKEN`、`KB_CONSOLE_SERVICES` | 默认 `127.0.0.1:9800`，默认管理 `database,mineru` 服务组 |
| 检索 | `KB_SEARCH_HOST`、`KB_SEARCH_PORT`、`KB_SEARCH_TOKEN`、`KB_SEARCH_TOP_K`、`KB_SEARCH_RERANK`、`KB_SEARCH_CONTEXT_TOKENS`、`KB_SEARCH_REQUEST_BUDGET`、`KB_SEARCH_ROUTE_MAX_KBS`、`KB_SEARCH_ROUTE_GAP`、`KB_SEARCH_ROUTE_FLOOR` 及其他 `KB_SEARCH_*` | 监听地址与令牌、返回数量、重排、上下文预算、整次请求的时间预算（应小于调用方的超时）；自动路由会把证据分数与最高分相差不超过 GAP 的知识库一起查（最多 MAX_KBS 个），全部低于 FLOOR 时标记结果偏弱 |

每个参数在代码里都有默认值。模板列出了全部检索参数及其代码默认值，除监听地址和令牌外都是注释掉的，取消注释之前一律按标定好的默认值运行（有一条测试保证两边不再漂移）。各阈值的含义见 [`app/kb_search/config.py`](../app/kb_search/config.py)；检索服务的 `/health` 会报告重排和视觉通道当前是否启用。

## 单库设置

控制台按知识库存储以下配置：

| 类别 | 主要参数 | 用途 |
|---|---|---|
| 切块 | `max_tokens`、`overlap_tokens` | 在向量模型上下文限制内控制块大小与重叠 |
| 图谱控制 | `graph_enabled`、`graph_auto_append`、`graph_profile` | 启用图谱、允许增量更新、保存自动归纳的场景画像 |
| 类型与关系 | `graph_entity_types`、`graph_parent_types`、`graph_type_definitions`、`graph_predicates`、`graph_examples`、`graph_language` | 自动生成的类型、关系与示例 |
| 抽取 | `graph_tune_sample_size`、`graph_unit_chunks`、`graph_max_gleanings` | 类型归纳采样量、抽取单元大小与补充抽取轮数 |
| 模型 | `graph_llm.extract`、`graph_llm.summarize`、`graph_llm.tune` | 为各建图阶段选择已注册的模型 |
| 重建 | `graph_rebuild_interval`、`graph_rebuild_new_chunk_pct`、`graph_rebuild_new_chunk_count`、`graph_rebuild_operator` | 按时间与资料变化阈值触发，支持 `or` / `and` |

修改切块设置后，使用控制台的整库重新解析操作，将新配置应用到已有文件。解析器实现版本更新时，后续扫描会自动为受影响文件重新排队。

## 访问控制与数据处理

- **控制台：** 默认仅监听本机。局域网访问需设置 `KB_WEB_HOST=0.0.0.0` 和 `KB_WEB_TOKEN`；配置令牌后，API 调用必须携带令牌，浏览器写操作还需满足同源检查。控制台提供资料读取和服务管理功能。
- **检索服务：** 远程调用受保护接口需配置 `KB_SEARCH_TOKEN`；未设置时，这些接口仅接受本机请求。`/health` 无需认证，并会返回运行元数据，其可访问范围由监听地址和网络规则决定。
- **存储与模型服务：** 提供的 Compose 配置将端口绑定到 loopback，可按访问范围调整绑定地址。loopback 能挡住其他机器，挡不住同一台机器上浏览器打开的网页：Qdrant 为此关闭了 CORS，但配套的 Qdrant 没有 API Key，因此不要在运行这套服务的机器上浏览网页。
- **模型数据：** 文档、切片和图片会发送给配置的接口。远程供应商会收到这些内容；本地接口在自有基础设施内处理。
- **凭据：** 模型注册表密钥保存在本机状态库，API 不回显。更换模型端点后，复用密钥需要重新输入。环境文件与数据库包含部署凭据。
- **可选依赖：** PDF 原图检索需安装 `pdf-images`，其许可证见 [第三方声明](../NOTICE.md)。
