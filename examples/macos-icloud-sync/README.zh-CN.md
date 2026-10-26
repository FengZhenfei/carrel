# 示例：从 Mac 同步 iCloud 云盘文件夹

[English](README.md) | 中文

这是 [作者实际部署](../../docs/example-setup.zh-CN.md) 中 Mac 这一端的做法，Carrel 本身运行在一台 NVIDIA DGX Spark 上。资料放在 iCloud 云盘的一个文件夹里，每个子文件夹对应一个知识库；Mac 上的两个 LaunchAgent 负责让服务器上的镜像目录（`runtime/mirror`）与这个文件夹保持一致：

| 代理 | 脚本 | 每分钟做的事 |
|---|---|---|
| `local.carrel.icloud-prefetch` | `icloud-prefetch.sh` | 把只存在云端、还没下载到本机的文件下载下来 |
| `local.carrel.mirror-push` | `push-mirror.sh` | 用 rsync 经 SSH 把文件夹推送到服务器的镜像目录 |

镜像目录用什么方式填充，Carrel 并不限定（见 [资料目录](../../docs/configuration.zh-CN.md#资料目录)）。iCloud 云盘之所以要多花些功夫，是因为有两个坑：一是还没下载的文件只是占位文件，rsync 会把它当成空文件推过去；二是没有权限的后台任务读到的文件夹是空的，这时 `rsync --delete` 会把服务器上的镜像也清空。推送脚本每一轮都会先排除这两种情况。

## 推送前的检查

- **占位文件：** 文件夹里还有没下载完的文件，或者下载脚本正在运行，这一轮就跳过。
- **文件夹为空或读不了：** 直接拒绝推送，避免因为缺权限把镜像清空。
- **运行方式：** 只在 launchd 下运行，因为下面授予的权限只在这种方式下生效。从终端手动运行时，必须加 `--allow-no-launchd` 才会执行。
- **服务器连不上：** 跳过这一轮，服务器上什么都不改。
- **锁：** 本机用一把锁防止两次推送重叠；复制期间，脚本在服务器上持有 `runtime/state/mirror_sync.lock.d`，Carrel 的扫描会等它释放后再进行。进程异常退出留下的锁，过期后由下一轮接管。
- **复制到一半的文件：** 借助 `--delay-updates`，新增和修改的文件在整批传完之前都处于隐藏状态：rsync 先把它们写成以点开头的临时文件，传完的暂存在 `.rsync-partial/` 文件夹里，最后一起改名就位。扫描会跳过隐藏的文件和文件夹，所以不会读到半批数据。

删除也会同步过去：从文件夹里删掉的文件，下一轮推送后会从镜像中消失，Carrel 随后把它移出索引。失效的数据会在撤销期内保留（`QDRANT_INACTIVE_RETENTION_DAYS`，模板为 7 天），文件在这段时间内放回来，索引就会恢复。

## 准备工作

- 能免交互登录服务器的 SSH 密钥：`ssh -o BatchMode=yes <主机> true` 必须执行成功。
- 服务器上装有 rsync（大多数 Linux 发行版自带）。
- 可选：Xcode 命令行工具（`xcode-select --install`）。装了以后，`install.sh` 会用 `icloud_download.swift` 编译一个小的下载工具；没装时，下载脚本改用 `brctl download`。

## 安装

在 Carrel 的代码目录里（或者从 GitHub 单独下载这个文件夹）执行：

```bash
cp -R examples/macos-icloud-sync ~/carrel-sync   # 放在 iCloud 云盘以外的任意位置
cd ~/carrel-sync
cp sync.env.example sync.env                     # 填写 SYNC_SOURCE、SYNC_HOST 和 SYNC_REMOTE_DIR
./install.sh
```

`install.sh` 会先检查 SSH 登录，再编译下载工具，把两个代理写入 `~/Library/LaunchAgents` 并加载。代理直接运行这个文件夹里的脚本，所以装好以后不要再移动它。

接下来打开 **系统设置 → 隐私与安全性 → 完全磁盘访问权限**，添加 `/bin/bash` 和 `/bin/zsh`（在文件选择框里按 Cmd+Shift+G 可以直接输入路径）。launchd 用 bash 运行推送脚本、用 zsh 运行下载脚本。macOS 要求程序在读取 iCloud 云盘、桌面或文稿之前先获得授权，而后台任务无法响应授权弹窗，所以不授权的话，每一轮都会停在上面的检查上。这项权限对 launchd 启动的其他 bash 和 zsh 脚本同样有效，授权前请先权衡。

可选下载工具可能需要单独授权。如果 `logs/icloud-prefetch.trace.log` 出现该工具的 `Code=257`，想使用它时可将安装目录里的 `bin/icloud-download` 添加到完全磁盘访问权限。工具失败后，本轮自动改用系统的 `brctl`，下一轮重新尝试工具。

下载默认同时等待最多 64 个文件，等待时每隔 5 秒统一检查，发现完成就立即补入新文件。单个文件超时后记录并留待下一轮重试。`logs/icloud-prefetch.log` 每 30 秒更新一次进度，并在本轮结束时汇总；请求被接受和文件实际落地分别统计。批量大小及超时可在 `sync.env` 调整。

查看日志，确认最初几轮的结果：

```bash
tail -f ~/carrel-sync/logs/push-mirror.log
```

| 日志 | 含义 |
|---|---|
| `pushed: … server flag set` | 有改动，已推送 |
| `no-op` | 没有改动 |
| `skip: …` | 这一轮在等某个条件（占位文件、锁、网络），下一轮会再试 |
| `abort: …` | 检查没通过，需要处理（权限、文件夹为空） |

想先手动预览一次推送，可以在有权读取该文件夹的终端里运行 `./push-mirror.sh --dry-run --allow-no-launchd`，它只列出将要发生的改动，不会改动服务器上的任何内容。

## 服务器端设置

- 每次推送产生改动后，脚本会更新服务器上的 `runtime/state/mirror_changed.flag`。设置 `KB_SCAN_REQUIRE_FLAG=1` 后，Carrel 只在这类推送之后才扫描；模板里的 `0` 表示每分钟都扫描，这个标记留着也没有影响。
- `sync.env` 里的 `SYNC_REMOTE_LOCK_STALE_SECONDS` 要和服务器上的 `KB_MIRROR_LOCK_STALE_SECONDS` 保持一致（默认都是一小时）。
- 如果用 `KB_MIRROR_ROOT` 改了镜像目录的位置，`SYNC_REMOTE_MIRROR` 也要指向同一处。

## 同步其他文件夹或网盘

推送脚本适用于 Mac 上的任意文件夹：把 `SYNC_SOURCE` 指向该文件夹，再用 `./install.sh --push-only` 安装即可。普通文件夹里不会有占位文件，这项检查也不会带来额外开销。

NAS 或其他网盘的同步客户端在本机维护的文件夹也可以这样推送，前提是在客户端中把文件设为始终保留在本机：仅存于云端的文件同样是占位文件，推送会一直等到它们下载完成。这类文件夹可能和 iCloud 云盘一样受到 macOS 的保护，如果日志中出现 `abort: source unreadable`，按前文步骤授予完全磁盘访问权限即可；桌面、文稿和下载以外的普通文件夹则不需要。

## 文件说明

| 文件 | 用途 |
|---|---|
| `push-mirror.sh` | 推送脚本，参数有 `--dry-run`、`--no-flag`（不更新服务器上的标记）和 `--allow-no-launchd` |
| `icloud-prefetch.sh` | 请求 iCloud 下载占位文件，并等到文件真正落地 |
| `icloud_download.swift` | 下载工具 `bin/icloud-download` 的源码 |
| `sync.env.example` | 配置模板，可选项连同默认值一并列出 |
| `launchd/*.plist` | 两个代理的定义，`install.sh` 会填入本文件夹的路径 |
| `install.sh` | 安装代理，另有 `--push-only` 和 `--uninstall` |

两个脚本各自在 `logs/` 里轮换日志，超过 2 MB 就换新文件，保留三份旧的。`*.launchd.log` 只记录意外的报错；`logs/icloud-prefetch.last-errors.log` 列出上一轮下载失败或超时的文件。

传输期间会定期刷新服务器上的同步锁。复制前先记录待通知状态，中断后即使下一轮没有文件需要补传，也会补发 Carrel 扫描通知。预演模式只检查服务器并列出拟议改动。

## 卸载

```bash
~/carrel-sync/install.sh --uninstall
```

这条命令会停用并删除两个代理。之后可以删掉整个文件夹；如果没有别的程序需要，也请把 `/bin/bash` 和 `/bin/zsh` 从完全磁盘访问权限里移除。
