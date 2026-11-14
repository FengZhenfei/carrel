# 容器部署

[首页](../../README.zh-CN.md) · [English](README.md) | 中文

Light 和 Standard 共用一个应用镜像。应用容器运行控制台、检索 API 和后台调度；
MinerU 作为独立解析容器集成。Standard 额外启动 Qdrant、OpenSearch、Neo4j；
Light 使用持久化应用目录中的 LanceDB、DuckDB 和 SQLite。

LLM、Embedding、Reranker 继续在控制台独立配置接口，部署不包含这些模型的服务或权重。

应用镜像预装 Carrel 已支持代码语言的语法解析库，在镜像构建时准备好，
首次解析代码时无需再下载。Python 代码使用 Python 自带的 AST 解析。

## 启动

安装 Docker 和 Compose v2，在仓库根目录执行：

```bash
./deploy.sh --light --build
# 或选择 Standard：
# ./deploy.sh --standard --build
```

`deploy.sh` 是可选的 Compose 快捷入口，负责生成 `deployment/compose/.env`、
数据库密码、构建应用与解析镜像并启动所选服务。宿主机不需要安装 Python 或 systemd。
默认使用 CPU 解析；配置 NVIDIA Container Toolkit 后，可加 `--gpu`。
MinerU 沿用已有的硬件检测和解析后端选择规则。

打开 **http://127.0.0.1:9800**，在「系统设置」中配置模型接口。
将原文件目录放进 `data/runtime/mirror/`，再在控制台启用。
应用首次启动会生成 `data/config/knowledge-base.env`，其中 `KB_SEARCH_TOKEN`
供检索客户端使用。

首次启动会下载 MinerU 的解析资源，后续通过 `MINERU_MODELS_PATH` 复用。
可选的 `--with-pdf-images` 构建参数会加入 PyMuPDF，用于检索 PDF 原始分辨率图片；
许可证见[第三方声明](../../NOTICE.md)。

## 直接使用 Compose

```bash
cd deployment/compose
cp .env.example .env
# Light：修改 CARREL_EDITION=light，COMPOSE_PROFILES= 留空。
# Standard：CARREL_EDITION=standard，COMPOSE_PROFILES=standard，
# 并在启动前设置非空的 NEO4J_PASSWORD。
docker compose up -d --build
docker compose ps
docker compose logs -f app
```

`CARREL_IMAGE=carrel-app:local` 是本地构建的镜像名称。使用已发布镜像时，
将 `CARREL_IMAGE` 改为对应镜像地址，执行 `docker compose pull app`，
再执行 `docker compose up -d --no-build app`。
仓库的镜像工作流可在版本标签或手动触发时构建 amd64、arm64 镜像；
增加工作流本身不会发布镜像。

需要镜像加速时，可配置 `CARREL_PYTHON_IMAGE`、`MINERU_BASE_IMAGE` 和数据库镜像地址。
`PIP_INDEX_URL` 设置构建时的 Python 包索引。GPU 解析使用
`COMPOSE_FILE=docker-compose.yml:compose.gpu.yml`、`MINERU_FLAVOR=gpu`
及对应的 GPU 基础镜像。

## 数据与网络

| 数据 | 位置 |
|---|---|
| 应用配置、原文件、索引、图谱、状态、日志、备份 | `CARREL_DATA_PATH`，默认仓库根目录下的 `data/` |
| Standard 数据库文件 | 按 Compose 项目隔离的 Docker 卷：`qdrant-data`、`opensearch-data`、`neo4j-data` 及 Neo4j 辅助卷 |
| MinerU 模型 | `MINERU_MODELS_PATH`，默认 `models/mineru/` |
| MinerU 输出及后端信息 | `MINERU_OUTPUT_PATH`，默认 `runtime/mineru-output/`，与应用共享 |

存储版本写入数据目录后固定。部署另一个版本时，使用独立目录和 Compose 项目，
不进行跨模式迁移。复用 Standard 数据卷时，应保持 Compose 项目名不变。

模型地址由应用容器访问。同一宿主机上的服务使用 `host.docker.internal`，并确保服务
允许 Docker 网桥访问；`127.0.0.1` 指向应用容器自身。其他机器使用可达地址。
集成服务通过 `mineru`、`qdrant`、`opensearch`、`neo4j` 服务名访问。

默认端口只绑定本机。需要对外开放时调整 `APP_BIND_ADDRESS`，并先为控制台配置
`KB_WEB_TOKEN`；检索使用生成的 Bearer Token。应用不挂载 Docker socket，容器启停
和重启通过 Compose 管理。Light 控制台不显示「系统服务」。

## 调度、重启与升级

内置调度按原有频率执行：每分钟扫描、每五分钟解析、每两小时图谱维护、每日回收、
每周缓存整理、每月日志归档。控制台手动操作使用相同的管线入口。
队列和下次执行时间保存在 `data/runtime/state/automation.db`。
重复等待请求会合并，维护任务避开正在运行的任务，建图沿用全局锁。
单独重启 Web 进程不会中断由调度器启动的建图进程。

容器停止时通知正在执行的任务，并预留 60 秒退出时间。重启后，中断任务沿用已有
恢复和缓存规则；用户主动暂停的任务保持暂停。Docker 的 `unless-stopped` 策略
会在宿主机重启后启动应用，主动停止的容器除外。

```bash
./deploy.sh status
./deploy.sh down                   # 保留文件和数据库卷
./deploy.sh --build                # 源码更新后重建，复用 .env 和已有数据
cd deployment/compose
docker compose restart app        # 重载应用配置
```

HTTP 与任务日志在 `data/logs/`，调度器日志也可通过 `docker compose logs app` 查看。
容器健康检查覆盖调度器、控制台和检索进程；依赖就绪情况由控制台和 `/health`
分别展示。后台任务在依赖未就绪时等待后续执行。

源码部署仍保留 [systemd 安装入口](../systemd/README.md)。已有宿主机部署转为容器时，
需要明确映射路径和数据卷；快捷脚本不会搬移已有数据或停止宿主机服务。
