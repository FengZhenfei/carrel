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

删除知识库会先终止该库正在运行的解析和建图任务，再立即清除它的索引、图谱数据、状态和缓存，不经过保留期。删除完成前，控制台将其显示为正在删除；某个存储服务没能清理干净时，显示为删除未完成，此时不能重新启用或沿用，下一轮夜间维护会接着删，也可以再执行一次删除。删除期间持有建图锁：删除进行时不会开始新的建图，其他知识库正在建图时也不能删除。

控制台管理 API 位于端口 9800 的 `/api` 路径。写操作必须来自控制台自身的源；设置了 `KB_WEB_TOKEN` 后，所有调用都要携带令牌，见 [配置说明](configuration.zh-CN.md#访问控制与数据处理)。请求体的字段定义见 [`kb_server/api.py`](../app/kb_server/api.py)。

| 路由 | 用途 |
|---|---|
| `GET /api/overview`、`/api/health`、`/api/limits` | 侧栏状态、服务健康、向量模型决定的切块上限 |
| `POST /api/enroll`；`POST /api/kbs/{kb_id}/adopt`、`unenroll`；`DELETE /api/kbs/{kb_id}` | 启用目录、沿用改名目录、关闭或删除知识库 |
| `GET` / `PUT /api/kbs/{kb_id}/config` | 读取和保存单库设置 |
| `POST /api/kbs/{kb_id}/parse_now`、`reparse` | 立即排队新文件、整库重新解析 |
| `GET /api/kbs/{kb_id}/files`、`files/{file_id}/chunks` | 文件状态、单个文件的切块 |
| `POST /api/kbs/{kb_id}/graph_schema`、`graph_build`、`graph_append`、`graph_pause`；`DELETE /api/kbs/{kb_id}/graph` | 抽取标签、建图、增量并入、暂停、删除图谱 |
| `GET /api/kbs/{kb_id}/graph_preview`、`graph_merges`、`graph_corpus` | 当前图谱、归并记录、语料规模 |
| `GET /api/jobs/{job_id}`；`POST /api/jobs/{job_id}/cancel`、`/api/files/retry` | 单个任务及其时间线、取消、重试 |
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
| `carrel-qdrant-gc.timer` | 30 分钟后首次执行，之后每 24 小时 | 先运行 `kb cleanup backup`（见下文），再运行 `kb cleanup parse-assets-gc`，随后运行 `cleanup graph-gc`：清理超过保留期的失活索引点及其切块记录和解析资产、删除时间超过撤销窗口的文件、旧任务记录、超过保留期的已关闭知识库以及没删完的知识库；图版本只留现行版加 N−1 个（`GRAPH_GC_KEEP_VERSIONS`，不看天数，与建图收尾同一条规则） |
| `carrel-cache-weekly.timer` | 1 小时后首次执行，之后每 7 天 | 轮换项目缓存；解析缓存里已经没有对应图片的图片描述和视觉向量缓存随之删除 |
| `carrel-logs-monthly.timer` | 2 小时后首次执行，之后每 30 天 | 轮换日志 |

控制台和检索 API 作为常驻服务运行，内存紧张时批处理单元先于它们被终止（`OOMScoreAdjust` 分别为 200 和 100）。定时器按相对间隔执行，不使用日历时刻。入库、建图或同步繁忙时，维护任务可以延后；连续多次延后会以退出码 75 显示失败状态，策略见 `scripts/lib/kb-maint-defer.sh` 和 [systemd 文档](../deployment/systemd/README.md#busy-yield)。

整机关机、重启或断电停下的整库建图，由定时的图谱检查（`check-rebuild --execute`）接着处理。这一轮本来不需要整库重建时，会按同一版本续跑：跑完的阶段跳过，抽取结果走缓存，最多续两次。首次建图还没成功过，或者重建策略已经到期时，这一轮会另起新版本从头跑，阶段不跳过，图谱设置没变的话抽取结果仍走缓存。

建图收到停止信号时会当场记下 systemd 是否正在关机；被直接杀掉的，按「开跑早于本次开机」认出；机器一直开着期间死掉的建图，扫描会在一分钟内记为已不存在，不会被续。人停的（控制台暂停或关闭、Ctrl-C、`kill`、`systemctl stop`）、OOM、带 `--no-activate-aliases` 跑的建图、升级到这一版之前就已开始的建图、被回退顶替的版本，以及之后改过图谱设置（模型、标签、谓词、语言、单元大小、嵌入模型，或升级带来的提示词变化）的建图，都留给控制台处理：缓存还能续用时按钮是「继续建图」，否则是「立即/重新建图」，会另起新版本。没有安排定时图谱检查的主机不会自动续跑；不在 Linux 上时，停止也不会被认作关机。

不带 `--execute` 的 `kb graph check-rebuild` 会用 `would_resume`（版本和已续次数）报出下一轮会尝试续跑的建图，次数用完的报成 `resume_declined`（`attempts_exhausted`）。它不检查模型和配置指纹，所以真正执行的那一轮仍可能拒绝（`config_changed`）或跳过（`build_skipped`）。要让某个库的定时建图不再跑，可以在建图进行中点「暂停建图」，或者关掉「开启知识图谱」并保存；停掉 `carrel-graph-rebuild.service` 只会结束当前这一轮，下一轮仍由定时器照常触发。

默认维护只处理项目资产。清理用户级 uv/pip 缓存或 Docker 缓存需显式设置 `KB_HOST_HOUSEKEEPING=1`。

每夜备份只留不可再生的内容：状态库（各库配置、模型注册表、抽取与事实缓存）、管线 env、compose 的 `.env` 以及 `runtime/eval` 下的题集，原样复制到 `backups/state/<时间戳>/`，只保留最近 `KB_BACKUP_KEEP`（7）份。向量、索引、图谱和解析缓存都能从镜像重建。恢复时先停定时器和两个服务，从最新一份放回两个 env 与 `runtime/state/kb-pipeline.db`，用 `docker compose up -d` 起容器，再起服务和定时器；状态库里的缓存还在的话，整库重建不会再调用远端模型。

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
| `cleanup status/weekly/monthly/parse-assets-gc/graph-gc/backup [--dry-run]` | 维护状态、定时器自动运行的清理，以及每夜备份 |
| `cleanup qdrant-gc/qdrant-graph-gc/neo4j-graph-gc [--dry-run]` | 手动工具：`qdrant-gc` 只删过期的失活点；另外两个按天数清理图版本（对应两个图谱保留天数参数） |
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
| 文件一直排队 | `KB_PARSE_ENABLED`、worker 日志、活动锁，以及模型接口是否已配置且可用。任务连不上所需的服务（如解析服务或模型接口）时会退回队列，不计重试次数，等待时间逐次加倍；这样退回达到 `KB_JOB_MAX_RETRIES` 次后，按普通错误记为失败 |
| 删除后显示删除未完成 | 有存储服务没能清理干净：检查 Qdrant、OpenSearch 和 Neo4j，然后再删除一次，或等夜间维护接着删 |
| 建图提示已有锁 | 确认是否存在运行中的构建；flock 随进程退出释放 |
| 维护退出码为 75 | 系统繁忙导致连续延后，检查入库、建图、同步和 Qdrant 状态 |
| 图片处理失败 | 分别检查图片描述接口，以及已启用的视觉向量接口 |
| 解析后端不符合预期 | 查看 `docker logs carrel-mineru` 和 Compose 环境中的 `MINERU_BACKEND_POLICY` |
| 检索结果较弱或仅供诊断 | 先检查 `retrieval_summary`、来源状态、路由与可用通道，再决定是否调整阈值 |
| Neo4j 投影需要恢复 | 先检查日志与备份；服务恢复可用后，可通过 `kb graph neo4j-import` 投影当前图版本 |

恢复存储服务时，先保留状态库和相关运行数据，再重建索引或投影。

## 开发与测试

`deploy.sh` 按 `app/requirements.lock` 安装测试过的那套版本（含 pytest）；手动安装等价于 `app/.venv/bin/pip install -r app/requirements.lock -e app`。依赖有变动时，在测试过的虚拟环境里用 `app/.venv/bin/python scripts/kb-lock-deps.py app/pyproject.toml > app/requirements.lock` 重新生成锁文件。安装应用和测试依赖后执行：

```bash
cd app
.venv/bin/python -m pytest -q
```

回归测试使用服务桩和临时数据，不依赖运行中的容器。缺少可选依赖或已安装的 systemd 单元时，部分检查会跳过。针对真实部署的检索评测是独立步骤，见 [检索评测](retrieval.zh-CN.md#检索评测)。

- 修改解析器输出后，应更新 `parsers/common.py` 中相应的档案或版本号，让受影响文档重新处理。
- 控制台文案需加入 `static/i18n.js` 中英字典；文件名、实体名保留原始语言。
- 静态前端变更无需重启服务；Python 变更需重启受影响的应用服务。重启前应确认活动任务：控制台内执行的标签抽取会丢失，正在进行的知识库删除会中断，之后由夜间维护接着完成。
- 修改 systemd 模板后，先用 `./scripts/install-systemd.sh --check` 检查差异，再按需重新安装单元。
- 回归夹具使用合成数据，运行资料与凭据保存在源码目录之外。`app/tests/test_deployment.py` 包含隐私检查，机器专属的扫描规则放在仓库外。
