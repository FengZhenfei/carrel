# 运维与开发

[首页](../README.zh-CN.md) · [English](operations.md) | 中文

除非命令中明确使用 `cd`，以下操作均以仓库根目录为起点。文中的 `kb` 指 `app/.venv/bin/kb`，安装包也提供同等入口 `carrel`。

## 控制台

控制台包含三个主要视图：

| 视图 | 功能 |
|---|---|
| 配置 | 切块、图谱设置、模型选择、重建策略与处理操作 |
| 文件 | 解析状态、切块数量、错误、重试、切块预览和任务时间线 |
| 图谱 | 当前实体与关系、类型筛选和归并记录 |

顶栏提供模型注册表、服务健康与控制、中英文切换。保存配置只记录设置；修改切块参数后，应执行整库重新解析。解析器版本变化时，后续扫描可以自动为受影响文件重新排队。

关闭知识库会停用其索引并进入保留期管理，保留期内重新启用可以恢复。目录改名识别成功时保留原编号，有歧义时可在控制台执行“沿用”操作。

控制台管理 API 位于端口 9800 的 `/api` 路径。写操作必须来自控制台自身的源；设置了 `KB_WEB_TOKEN` 后，所有调用都要携带令牌，见 [配置说明](configuration.zh-CN.md#访问控制与数据处理)。请求体的字段定义见 [`kb_server/api.py`](../app/kb_server/api.py)。

| 路由 | 用途 |
|---|---|
| `GET /api/overview`、`/api/health`、`/api/limits` | 侧栏状态、服务健康、向量模型决定的切块上限 |
| `POST /api/enroll`；`POST /api/kbs/{kb_id}/adopt`、`unenroll`；`DELETE /api/kbs/{kb_id}` | 启用目录、沿用改名目录、关闭或删除知识库 |
| `GET` / `PUT /api/kbs/{kb_id}/config` | 读取和保存单库设置 |
| `POST /api/kbs/{kb_id}/parse_now`、`reparse`、`chunk_preview` | 立即排队新文件、整库重新解析、切块预览 |
| `GET /api/kbs/{kb_id}/files`、`files/{file_id}/chunks`、`jobs` | 文件状态、单个文件的切块、任务时间线 |
| `POST /api/kbs/{kb_id}/graph_schema`、`graph_build`、`graph_append`、`graph_pause`；`DELETE /api/kbs/{kb_id}/graph` | 抽取标签、建图、增量并入、暂停、删除图谱 |
| `GET /api/kbs/{kb_id}/graph_preview`、`graph_merges`、`graph_builds`、`graph_corpus` | 当前图谱、归并记录、构建历史、语料规模 |
| `GET /api/jobs/failed`、`/api/jobs/{job_id}`；`POST /api/jobs/{job_id}/cancel`、`/api/files/retry` | 失败任务、单个任务、取消、重试 |
| `GET` / `POST /api/llms`；`DELETE /api/llms/{name}` | 模型注册表（密钥只写不读） |
| `POST /api/services/{key}/restart`、`restart_all`、`stop_all` | 对 `KB_CONSOLE_SERVICES` 列出的服务行做重启与停止 |

## 部署与服务

| 命令 | 作用 |
|---|---|
| `./deploy.sh` | 检测环境并部署，保留已有配置 |
| `./deploy.sh detect` | 查看环境检测和拟采用的配置 |
| `./deploy.sh --cpu` | 使用 CPU 解析部署方式 |
| `./deploy.sh --with-local-models` | 加入可选模型服务，需先准备权重 |
| `./deploy.sh --with-pdf-images` | 安装可选的 PDF 原图支持 |
| `./deploy.sh status` | 查看容器与控制台健康状态 |
| `./deploy.sh down` | 停止并移除 Compose 容器，保留数据；应用的 systemd 服务单独管理 |
| `./deploy.sh purge` | 移除部署容器、自建解析镜像、服务单元和 Compose 配置 |

`purge --images` 还会删除拉取的镜像，`purge --data` 会删除运行数据与模型权重。使用这些删除选项前，先备份需要保留的数据。

Linux 上可用以下命令查看服务：

```bash
systemctl --user list-timers 'carrel-*'
journalctl --user -u carrel-worker -f
journalctl --user -u carrel-graph-rebuild --since -3h
journalctl --user -u carrel-web --since -5min
```

[systemd 文档](../deployment/systemd/README.md) 介绍安装、用户会话退出后继续运行、单元模板及无 systemd 环境下的调度方式。[Compose 文档](../deployment/compose/README.md) 介绍后端选择、模型权重、内存配置与容器日志。

## 定时任务

| 单元 | 启用后的调度周期 | 用途 |
|---|---|---|
| `carrel-scan.timer` | 30 秒后首次执行，之后每分钟 | 扫描变化并排队 |
| `carrel-worker.timer` | 1 分钟后首次执行，之后每 5 分钟 | 消费入库任务 |
| `carrel-graph-rebuild.timer` | 10 分钟后首次执行，之后每 2 小时 | 增量更新，或按策略整库重建 |
| `carrel-qdrant-gc.timer` | 30 分钟后首次执行，之后每 24 小时 | 运行 `kb cleanup parse-assets-gc`（随后跑 `cleanup graph-gc`，按留 N 版规则清理图谱版本）：清理保留期已过的失活索引点、解析资产、旧任务记录和已关闭的知识库。图版本由建图收尾阶段（`GRAPH_GC_KEEP_VERSIONS`）以及 `kb cleanup qdrant-graph-gc` / `neo4j-graph-gc` 清理 |
| `carrel-cache-weekly.timer` | 1 小时后首次执行，之后每 7 天 | 轮换项目缓存 |
| `carrel-logs-monthly.timer` | 2 小时后首次执行，之后每 30 天 | 轮换日志 |

控制台和检索 API 作为常驻服务运行。定时器按相对间隔执行，不使用日历时刻。入库、建图或同步繁忙时，维护任务可以延后；连续多次延后会以退出码 75 显示失败状态，策略见 `scripts/lib/kb-maint-defer.sh`。

默认维护只处理项目资产。清理用户级 uv/pip 缓存或 Docker 缓存需显式设置 `KB_HOST_HOUSEKEEPING=1`。

## 命令行参考

完整参数可通过 `app/.venv/bin/kb --help` 和各子命令的 `--help` 查看。

| 命令组 | 用途 |
|---|---|
| `config`、`status`、`health` | 生效配置、队列和服务状态 |
| `init-db` | 初始化状态库 |
| `scan [--source kb_NNN] [--requeue-failed] [--rehash]` | 扫描已启用目录并排队；可重排失败文件；可按内容重新哈希而不信任修改时间 |
| `worker --once [--max-jobs N] [--max-seconds S]` | 执行一轮有预算限制的入库任务（不要在 systemd 的 worker 旁再起一个） |
| `qdrant ensure-collections`、`ensure-graph-collections` | 创建缺失集合 |
| `fts init/status/rebuild/sync-doc/search` | 关键词索引管理 |
| `graph build/append/check-rebuild [--execute] [--force-full]` | 整库建图、增量并入、按策略检查是否该重建（定时器跑的就是它） |
| `graph adopt-current/rollback/neo4j-import/neo4j-status/neo4j-delete [--graph-version V]` | 图版本基线、回退到保留的旧版本、Neo4j 投影管理 |
| `graph query/factcheck/status` | 图谱检查与评测 |
| `search eval/make-set` | 检索评测与题集准备 |
| `cleanup status/weekly/monthly/qdrant-gc/qdrant-graph-gc/neo4j-graph-gc/graph-gc/parse-assets-gc [--dry-run]` | 保留期与维护操作（定时器跑的就是这些） |
| `reset --source kb_NNN [--all] --yes` | 清空指定知识库的状态、缓存和索引并重建其集合 |

由 systemd 管理 worker 时，通过对应服务单元触发一轮入库。以下编号需替换成实际知识库编号：

```bash
app/.venv/bin/kb scan --source kb_001
systemctl --user start --no-block carrel-worker.service
```

`reset` 不带 `--yes` 时只输出计划，确认前应核对影响范围。仅需重新解析时，优先使用控制台的对应操作；正常部署保留已有知识库数据。

## 排障

先查看 `./deploy.sh status`、`app/.venv/bin/kb status` 和对应服务日志。控制台健康接口为 `/api/health`，配置令牌后需携带控制台令牌。`app/.venv/bin/kb graph status` 可查看图版本和构建状态。

| 现象 | 检查方向 |
|---|---|
| 文件一直排队 | `KB_PARSE_ENABLED`、worker 日志、活动锁，以及模型接口是否已配置且可用 |
| 建图提示已有锁 | 确认是否存在运行中的构建；flock 随进程退出释放 |
| 维护退出码为 75 | 系统繁忙导致连续延后，检查入库、建图、同步和 Qdrant 状态 |
| 图片处理失败 | 分别检查图片描述接口，以及已启用的视觉向量接口 |
| 解析后端不符合预期 | 查看 `docker logs carrel-mineru` 和 Compose 环境中的 `MINERU_BACKEND_POLICY` |
| 检索结果较弱或仅供诊断 | 先检查 `retrieval_summary`、来源状态、路由与可用通道，再决定是否调整阈值 |
| Neo4j 投影需要恢复 | 先检查日志与备份；服务恢复可用后，可通过 `kb graph neo4j-import` 投影当前图版本 |

恢复存储服务时，先保留状态库和相关运行数据，再重建索引或投影。

## 开发与测试

安装应用和测试依赖后执行：

```bash
cd app
.venv/bin/python -m pytest -q
```

回归测试使用服务桩和临时数据，不依赖运行中的容器。缺少可选依赖或已安装的 systemd 单元时，部分检查会跳过。针对真实部署的检索评测是独立步骤，见 [检索评测](retrieval.zh-CN.md#检索评测)。

- 修改解析器输出后，应更新 `parsers/common.py` 中相应的档案或版本号，让受影响文档重新处理。
- 控制台文案需加入 `static/i18n.js` 中英字典；文件名、实体名保留原始语言。
- 静态前端变更无需重启服务；Python 变更需重启受影响的应用服务。重启前应确认活动任务，控制台内执行的预览和标签抽取可能被中断。
- 修改 systemd 模板后，先用 `./scripts/install-systemd.sh --check` 检查差异，再按需重新安装单元。
- 回归夹具使用合成数据，运行资料与凭据保存在源码目录之外。`app/tests/test_deployment.py` 包含隐私检查，机器专属的扫描规则放在仓库外。
