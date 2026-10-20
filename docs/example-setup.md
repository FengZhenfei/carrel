# Example setup

English | [中文](example-setup.zh-CN.md)

Carrel has many moving parts, and the full local stack needs a capable
machine. This page describes the author's own installation, in daily use
on an NVIDIA DGX Spark, so you can see how the parts fit together and which
ones you can leave out.

## Machines

**Server.** One NVIDIA DGX Spark (GB10, 128 GB of unified memory, Ubuntu
24.04) runs everything: the console, the search service, the scheduled
tasks, the stores, the parser and all model servers. It has the components
that `./deploy.sh --with-local-models` installs.

**Mac.** The author's everyday computer. The documents live in an iCloud
Drive folder named `知识库` ("knowledge base"), and the Mac pushes that
folder to the server. The author also asks questions here, in Codex.

## Server

Nine containers, the ones in [deployment/compose](../deployment/compose/README.md)
with the local model profile:

| Container | Role |
|---|---|
| `carrel-mineru` | MinerU, with its vision-language backend on the GPU |
| `carrel-qdrant`, `carrel-opensearch`, `carrel-neo4j` | vectors, keywords, graphs |
| `carrel-embedding` | text embeddings, Qwen3-Embedding-0.6B |
| `carrel-reranker` | text reranking, Qwen3-Reranker-0.6B |
| `carrel-vl-embedding` | visual embeddings, Qwen3-VL-Embedding-2B |
| `carrel-vl-reranker` | visual reranking, Qwen3-VL-Reranker-2B |
| `carrel-vlm` | picture descriptions, Qwen3-VL-8B-Instruct-FP8 |

Graph building is the one part that uses a hosted model: DeepSeek Flash
through DeepSeek's OpenAI-compatible API, registered in the console's model
registry. Parsing, embeddings and picture descriptions all run on the
server; only graph building sends text out, to that provider.

The scheduled tasks run at the [default cadence](../deployment/systemd/README.md#units):
the scan every minute, the parse queue every five minutes, the graph check
every two hours, and the cleanups daily, weekly and monthly.

## Getting documents to the server

iCloud Drive is the single entry point for documents. Files in `知识库` can
be added, changed or removed on a Mac, an iPhone, an iPad or any other
device signed in to the same Apple Account, and iCloud brings every change
to this Mac. From there the automated pipeline takes over: pushing to the
server, parsing, indexing and graph updates need no manual step (a new
knowledge base only has to be enabled once in the console).

The Mac runs two LaunchAgents, every minute. One asks iCloud to download
files that are only in the cloud; the other copies the folder into the
server's mirror (`runtime/mirror`) with rsync over SSH. Each subfolder of
`知识库` is one knowledge base. The scripts, with installation steps, are in
[examples/macos-icloud-sync](../examples/macos-icloud-sync/README.md).

In daily use:

1. Save, change or delete a file in `知识库`, on any device.
2. The Mac pushes the change to the mirror.
3. The scan queues the file, the parse queue processes it, and the next
   graph check adds it to that knowledge base's graph, if it has one.
4. Ask Codex on the Mac. Its search skill queries the search service on the
   server, and Codex answers from the evidence.

The approach does not depend on iCloud. If your documents live on a NAS or
in another network or cloud drive, the same pattern applies: keep the
server's mirror in step with that folder, by a scheduled sync or by
mounting it, and the pipeline does the rest. Common options:

- On a Mac, push the folder that the drive's sync client keeps locally,
  with this example's push script (`./install.sh --push-only`).
- Schedule an rsync task on the NAS that pushes to the server.
- Mount the NAS share on the server and point `KB_MIRROR_ROOT` at the
  mount point.

[Document folders](configuration.md#document-folders) lists what a sync
method should take care of.

## What stays manual

- **A new knowledge base:** a new subfolder is not processed until it is
  enabled in the console. At that point the author also chooses its
  settings, such as whether it gets a graph and which model builds it.
- **Checking in:** the console shows failed files, service health and task
  progress. A look now and then is enough.
- **Changing settings:** after a chunking change, start a full re-parse
  from the console.

## Lighter setups

You do not need all of this to try Carrel.

- **No GPU, or a smaller one:** leave out `--with-local-models` and point the
  model endpoints at a provider. The parser also runs on the CPU. See
  [Model endpoints](configuration.md#model-endpoints).
- **No graphs:** graphs are optional for each knowledge base. Without them
  you need no chat model.
- **Documents already on the server:** put them in `runtime/mirror/<folder>/`
  directly, or point `KB_MIRROR_ROOT` at a mounted share. Then no sync is
  needed.
- **Another sync tool:** rsync from Linux, Syncthing or a network share all
  work. See [Document folders](configuration.md#document-folders).
