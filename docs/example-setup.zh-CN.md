# 部署实例

[English](example-setup.md) | 中文

Carrel 的组件比较多，完整的本地部署对机器也有一定要求。这里介绍作者自己在 NVIDIA DGX Spark 上日常使用的部署，方便你了解各部分如何配合，以及哪些部分可以省掉。

## 机器

**服务器：** 一台 NVIDIA DGX Spark（GB10，128 GB 统一内存，Ubuntu 24.04），所有服务都在这台机器上：控制台、检索服务、定时任务、各类存储、解析服务和全部模型服务。组件与 `./deploy.sh --with-local-models` 安装的相同。

**Mac：** 作者日常使用的电脑。资料放在 iCloud 云盘的「知识库」文件夹里，由这台 Mac 推送到服务器；作者也在这台 Mac 上通过 Codex 提问。

在手机、平板或其他 Mac 上放进同一个 iCloud 云盘文件夹的文件，也会走同样的流程。

## 服务器上的服务

共九个容器，即 [deployment/compose](../deployment/compose/README.md) 中启用本地模型 profile 后的全部服务：

| 容器 | 作用 |
|---|---|
| `carrel-mineru` | MinerU，在 GPU 上使用视觉语言模型后端 |
| `carrel-qdrant`、`carrel-opensearch`、`carrel-neo4j` | 向量、关键词、图谱 |
| `carrel-embedding` | 文本向量，Qwen3-Embedding-0.6B |
| `carrel-reranker` | 文本重排序，Qwen3-Reranker-0.6B |
| `carrel-vl-embedding` | 视觉向量，Qwen3-VL-Embedding-2B |
| `carrel-vl-reranker` | 视觉重排序，Qwen3-VL-Reranker-2B |
| `carrel-vlm` | 图片描述，Qwen3-VL-8B-Instruct-FP8 |

只有建图用的是远程模型：通过 DeepSeek 的 OpenAI 兼容接口调用 DeepSeek Flash，并在控制台的模型注册表中登记。解析、向量化和图片描述都在服务器上完成，只有建图会把文本发送给这家服务商。

定时任务按 [默认频率](../deployment/systemd/README.md#units) 运行：扫描每分钟一次，解析队列每五分钟一次，图谱检查每两小时一次，另有按天、按周、按月的清理任务。

## 资料同步

Mac 上有两个每分钟运行一次的 LaunchAgent：一个请 iCloud 把只在云端的文件下载到本机，另一个用 rsync 经 SSH 把文件夹推送到服务器的镜像目录（`runtime/mirror`）。「知识库」下的每个子文件夹对应一个知识库，比如产品资料、论文报告、项目资料等。脚本和安装步骤见 [examples/macos-icloud-sync](../examples/macos-icloud-sync/README.zh-CN.md)。

日常使用的流程是：

1. 在任意设备上保存、修改或删除「知识库」里的文件。
2. Mac 把改动推送到镜像目录。
3. 扫描任务把文件加入队列，解析队列完成处理；如果该知识库启用了图谱，下一次图谱检查会把它并入图谱。
4. 在 Mac 上向 Codex 提问，由检索 Skill 调用服务器上的检索服务，Codex 再根据返回的证据作答。

## 需要手动处理的事

- **新建知识库：** 新的子文件夹要先在控制台启用才会开始处理。作者会在这时一并设好该库的选项，比如是否建图、用哪个模型建图。
- **日常查看：** 控制台会显示处理失败的文件、服务状态和任务进度，偶尔看一眼就够了。
- **调整设置后：** 修改切块方式后，需要在控制台发起一次全量重新解析。

## 更轻量的部署

试用 Carrel 并不需要上面的全部组件。

- **没有 GPU，或显存较小：** 不加 `--with-local-models`，把模型接口指向服务商即可，解析服务也可以在 CPU 上运行。见 [模型接口](configuration.zh-CN.md#模型接口)。
- **不建图：** 图谱是每个知识库的可选项，不建图就不需要对话模型。
- **资料本来就在服务器上：** 直接放进 `runtime/mirror/<目录名>/`，或者用 `KB_MIRROR_ROOT` 指向挂载的共享目录，就不需要同步了。
- **其他同步工具：** Linux 上的 rsync、Syncthing 或网络共享都可以。见 [资料目录](configuration.zh-CN.md#资料目录)。
