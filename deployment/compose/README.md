# Carrel infrastructure (Docker Compose)

Everything the pipeline needs that is not the pipeline itself:

| Service | Container | Port (loopback) | Image |
|---|---|---|---|
| MinerU parser | `carrel-mineru` | 8765 | built here from `mineru/Dockerfile` |
| Qdrant (vectors) | `carrel-qdrant` | 6333 | `qdrant/qdrant:v1.18.2` |
| OpenSearch (keywords) | `carrel-opensearch` | 9200 | `opensearchproject/opensearch:3.7.0` |
| Neo4j (graph projection) | `carrel-neo4j` | 7474 / 7687 | `neo4j:5.26.28-community` |
| *profile `local-models`* | | | |
| text embeddings | `carrel-embedding` | 8101 | vLLM, Qwen3-Embedding-0.6B |
| text reranker | `carrel-reranker` | 8102 | vLLM, Qwen3-Reranker-0.6B |
| visual embeddings | `carrel-vl-embedding` | 8103 | vLLM, Qwen3-VL-Embedding-2B |
| visual reranker | `carrel-vl-reranker` | 8104 | vLLM, Qwen3-VL-Reranker-2B |
| vision-language model | `carrel-vlm` | 8105 | vLLM, Qwen3-VL-8B-Instruct-FP8 |

Run `./deploy.sh` at the repo root instead of `docker compose` directly: it
detects the machine, writes `.env` next to this file and only then calls
compose. Everything below explains what it decided and how to change it.

## What `deploy.sh` writes into `.env`

| Key | Chosen from |
|---|---|
| `COMPOSE_FILE` | `docker-compose.yml:compose.gpu.yml` when an NVIDIA driver *and* the container toolkit (CDI spec or nvidia runtime) are present, otherwise the base file alone. The overlay only adds `gpus: all` to the parser. |
| `COMPOSE_PROFILES` | `local-models` with `--with-local-models`, otherwise empty. |
| `MINERU_FLAVOR`, `MINERU_BASE_IMAGE` | `gpu` (vLLM base image, both backends installed) or `cpu` (`python:3.12-slim`, pipeline backend only). |
| `QDRANT_IMAGE`, `NEO4J_IMAGE`, `OPENSEARCH_IMAGE`, `VLLM_IMAGE` | Pinned tags; prefixed with a mirror registry when Docker Hub does not answer from this host. On aarch64 `VLLM_IMAGE` is NVIDIA's DGX Spark build. |
| `PIP_INDEX_URL` | A PyPI mirror when pypi.org does not answer; used for the image build and the host venv. |
| `TORCH_CPU_INDEX_URL` | For the `cpu` flavour: PyTorch's CPU-only wheel index, or a mirror of it when the official file host does not answer; the build falls back to PyPI's torch when neither works. |
| `MINERU_MODEL_SOURCE` | `huggingface` when reachable, else `modelscope`, else `auto`. |
| `OPENSEARCH_JAVA_HEAP`, `NEO4J_*`, `QDRANT_MEM_LIMIT` | Memory tier: small (< 32 GB), medium (< 96 GB), large. |
| `NEO4J_PASSWORD` | Random; copied into `config/knowledge-base.env`. |
| `TZ` | The host's timezone. |

Delete `.env` and re-run `deploy.sh` to detect again; edit it to override.
Every other key is documented in `.env.example`, and the defaults in
`docker-compose.yml` are the same values, so a key missing from `.env` falls
back to what the example says.

## The parser container

`mineru/entrypoint.sh` runs at every start:

1. `detect.py` asks torch what GPU the container received and applies
   `MINERU_BACKEND_POLICY`:
   - `auto` (default): `vlm-engine` when compute capability ≥ `MINERU_VLM_MIN_COMPUTE`
     (7.0) and total memory ≥ `MINERU_VLM_MIN_VRAM_GB` (8), otherwise `pipeline`.
     Unified-memory machines such as DGX Spark report their whole RAM here,
     which is what vLLM can actually use.
   - `pipeline`, `vlm-engine`, `hybrid-engine`: forced; a VLM backend without a
     capable GPU aborts with a clear message.
   - vLLM's memory share is `MINERU_VLM_TARGET_VRAM_GB` (7) divided by the
     GPU's total memory, so the same setting gives ~0.06 on a 121 GB box and
     ~0.85 on an 8 GB card. `MINERU_GPU_MEMORY_UTILIZATION` overrides it.
2. Downloads the weights for that backend once (`mineru-models-download`,
   HuggingFace or ModelScope) into `models/mineru/`, together with MinerU's
   own `mineru.json`. A later switch of backend downloads the other set.
3. Writes its decision to `runtime/mineru-output/carrel-mineru.json`. The
   pipeline reads it when `MINERU_BACKEND=auto`, and the console shows the
   backend next to the parser row.
4. Starts `mineru-api` with the vLLM engine arguments (VLM) or without (pipeline).

The image flavours differ only in the base image and the extras installed:

```bash
# rebuild after changing MINERU_VERSION or the Dockerfile
cd deployment/compose && docker compose build mineru
```

## Local model servers (profile `local-models`)

Only for hosts with an NVIDIA GPU. Put the weights under `models/` first:

```bash
# from HuggingFace (or the ModelScope equivalents)
pip install -U huggingface_hub
for m in Qwen/Qwen3-Embedding-0.6B Qwen/Qwen3-Reranker-0.6B Qwen/Qwen3-VL-Embedding-2B \
         Qwen/Qwen3-VL-Reranker-2B Qwen/Qwen3-VL-8B-Instruct-FP8; do
  huggingface-cli download "$m" --local-dir "models/$(basename "$m")"
done
./deploy.sh --with-local-models
```

`deploy.sh --with-local-models` also lists the model rows in
`KB_CONSOLE_SERVICES` and enables the visual vectors in
`config/knowledge-base.env`. The `*_GPU_MEMORY_UTILIZATION` fractions in
`.env.example` were tuned for a 121 GB unified-memory machine; scale them up
on a card with less memory. On that machine the five model servers take 0.65
of the memory, about 0.71 together with the parser's share; with unified
memory this is system RAM, and it cannot be swapped out.

The five model servers start in a queue: each one waits for the previous
server's `/health` before it loads its model (embedding, reranker,
vl-embedding, vl-reranker, vlm), and with the profile on `deploy.sh` points the
parser at the last of them through `MINERU_WAIT_FOR_URL`. Servers that profile
the shared memory at the same time fail their memory check and restart until
Docker's backoff happens to separate them; the queue avoids that both under
`docker compose up` and when Docker restores the containers at boot. A full
start takes about the sum of the load times.
A container still waiting shows `health: starting`; if the first server never
comes up, the ones behind it keep waiting, so look at that one first.

## Data locations

All bind mounts are relative to this directory, so they land in the checkout:

| Container path | Host path |
|---|---|
| Qdrant storage | `runtime/qdrant_data/` |
| OpenSearch data | `runtime/opensearch/data/` |
| Neo4j data / logs / plugins / import | `runtime/neo4j/` |
| MinerU weights and config | `models/mineru/` |
| MinerU output, decision file | `runtime/mineru-output/` |

Some of these are created root-owned by the containers; `deploy.sh purge
--data` removes them through a throwaway container when plain `rm` cannot.

Pictures reach the vision models inside the request (base64), so those
containers mount no media directory, only their weights (and the reranker's
chat template). Two store settings keep the data small:

- Neo4j keeps a single transaction log file. A community-edition node has no
  online backup, so the logs only serve crash recovery, and the image default
  of two days / 2 GB fills up after large graph-version deletions.
- OpenSearch's query insights are off. Left on, they keep the text of search
  requests in `top_queries-*` indices for seven days, and deleting a knowledge
  base does not clear them.

## Network

Every port binds `127.0.0.1`. The console (9800) and the search API (9810)
are started by the app, not by compose. Changing a `*_BIND_ADDRESS` to
`0.0.0.0` exposes an unauthenticated store or free GPU inference to the LAN.

Loopback keeps other machines out, not a browser running on this host: a web
page opened here can send requests to `127.0.0.1`. Qdrant's CORS is disabled
(`QDRANT__SERVICE__ENABLE_CORS=false`) so that such a page cannot read the
corpus or delete collections across origins. Qdrant still has no API key and
does not check the `Host` header, so do not browse the web on the host that
runs the stack.

## Health and logs

```bash
./deploy.sh status
docker logs -f carrel-mineru          # first start: weight download, backend decision
docker inspect -f '{{.State.Health.Status}}' carrel-opensearch
curl -s localhost:8765/health         # parser
curl -s localhost:6333/healthz        # qdrant
curl -s 'localhost:9200/_cluster/health?pretty'
```

Every container logs through Docker's `json-file` driver capped at three 50 MB
files, so `docker logs` holds the recent part only. Changes to the compose
settings take effect when a container is recreated; re-running `./deploy.sh`
does that for the services whose settings changed.

## Updating and removing

```bash
./deploy.sh                 # re-run: rebuilds only what changed, keeps .env
./deploy.sh down            # stop and remove containers, keep data
./deploy.sh purge           # + built images, systemd units, .env
./deploy.sh purge --images --data   # + pulled images, runtime/ and models/
```
