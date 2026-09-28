#!/usr/bin/env bash
# Carrel one-command deployment.
#
#   ./deploy.sh [up]        detect the machine, write the two config files,
#                           pull/build and start the containers, install the
#                           app into app/.venv, initialise the state DB and
#                           (Linux) install + enable the systemd user units
#   ./deploy.sh detect      print what `up` would choose; change nothing
#   ./deploy.sh status      containers and console health
#   ./deploy.sh down        stop and remove the containers (data stays)
#   ./deploy.sh purge       down, then remove the images this stack built,
#                           the systemd units and the compose .env
#                           --images  also remove the images it pulled
#                           --data    also delete runtime/ and models/
#
# Options for `up`:
#   --with-local-models     also run the five vLLM model servers (needs an
#                           NVIDIA GPU and the weights under models/)
#   --with-pdf-images       also install the optional PyMuPDF extra (AGPL-3.0)
#                           so search can serve original-resolution PDF pictures
#   --cpu                   build the CPU flavour of the parser even if a GPU
#                           is present
#   --no-systemd            do not install or enable the user units
#   --no-app                containers only: skip venv, config, init-db, units
#   --yes                   never prompt (assumed when stdin is not a tty)
#
# Re-running `up` is safe: existing .env and config/knowledge-base.env are
# kept, images are rebuilt only when their inputs changed, running containers
# are recreated only when their definition changed.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_DIR="$ROOT/deployment/compose"
COMPOSE_ENV="$COMPOSE_DIR/.env"
APP_ENV="$ROOT/config/knowledge-base.env"
STORE_CONTAINERS=(carrel-qdrant carrel-opensearch carrel-neo4j)
MODEL_CONTAINERS=(carrel-embedding carrel-reranker carrel-vl-embedding carrel-vl-reranker carrel-vlm)
DOCKER_HUB_MIRRORS=(docker.m.daocloud.io docker.1ms.run)
PYPI_MIRRORS=(https://mirrors.aliyun.com/pypi/simple/ https://pypi.tuna.tsinghua.edu.cn/simple/)
# PEP 503 indexes of PyTorch's CPU-only wheels (the official one serves files
# from download-r2.pytorch.org, which some networks cannot reach).
TORCH_CPU_INDEXES=(https://download.pytorch.org/whl/cpu/ https://mirror.sjtu.edu.cn/pytorch-wheels/cpu/)

CMD="up"
WITH_MODELS=0
WITH_PDF_IMAGES=0
FORCE_CPU=0
NO_SYSTEMD=0
NO_APP=0
YES=0
PURGE_IMAGES=0
PURGE_DATA=0
[[ -t 0 ]] || YES=1

for arg in "$@"; do
  case "$arg" in
    up|detect|status|down|purge) CMD="$arg" ;;
    --with-local-models) WITH_MODELS=1 ;;
    --with-pdf-images) WITH_PDF_IMAGES=1 ;;
    --cpu) FORCE_CPU=1 ;;
    --no-systemd) NO_SYSTEMD=1 ;;
    --no-app) NO_APP=1 ;;
    --yes|-y) YES=1 ;;
    --images) PURGE_IMAGES=1 ;;
    --data) PURGE_DATA=1 ;;
    -h|--help) sed -n '2,34p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $arg (try --help)" >&2; exit 2 ;;
  esac
done

log()  { printf '\033[1;34m[carrel]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[carrel]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[carrel]\033[0m %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# set_kv FILE KEY VALUE: replace the KEY= line or append it.
set_kv() {
  local file="$1" key="$2" val="$3" tmp
  tmp="$(mktemp)"
  awk -v k="$key" -v v="$val" '
    BEGIN { done = 0 }
    index($0, k "=") == 1 { print k "=" v; done = 1; next }
    { print }
    END { if (!done) print k "=" v }' "$file" > "$tmp"
  cat "$tmp" > "$file"
  rm -f "$tmp"
}

get_kv() {  # get_kv FILE KEY -> value or empty
  [[ -f "$1" ]] || return 0
  awk -v k="$2" 'index($0, k "=") == 1 { sub("^" k "=", ""); print; exit }' "$1"
}

reachable() {  # reachable URL [seconds]: HTTP answered at all (any status)
  local code
  code="$(curl -sS -o /dev/null -m "${2:-6}" -w '%{http_code}' "$1" 2>/dev/null || true)"
  [[ -n "$code" && "$code" != "000" ]]
}

# ── detection ──────────────────────────────────────────────────────────────
ARCH="$(uname -m)"
OS="$(uname -s)"
GPU_DRIVER=0; GPU_DOCKER=0; GPU_NAME=""
MEM_GB=0; MEM_TIER="small"
REGISTRY_PREFIX=""; HUB_STATE="direct"
PIP_INDEX=""; PIP_STATE="direct"
TORCH_CPU_INDEX=""
MODEL_SOURCE="auto"
HOST_TZ="UTC"

detect() {
  if have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then
    GPU_DRIVER=1
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  fi
  if docker info --format '{{range $k, $v := .Runtimes}}{{$k}} {{end}}' 2>/dev/null | grep -q nvidia \
     || [[ -f /etc/cdi/nvidia.yaml || -f /var/run/cdi/nvidia.yaml ]]; then
    GPU_DOCKER=1
  fi
  if [[ "$OS" == "Darwin" ]]; then
    MEM_GB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024 ))
  else
    MEM_GB=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo 0) / 1024 / 1024 ))
  fi
  if   (( MEM_GB >= 96 )); then MEM_TIER="large"
  elif (( MEM_GB >= 32 )); then MEM_TIER="medium"
  else MEM_TIER="small"; fi

  if reachable https://registry-1.docker.io/v2/; then
    HUB_STATE="direct"
  else
    HUB_STATE="unreachable"
    for m in "${DOCKER_HUB_MIRRORS[@]}"; do
      if reachable "https://$m/v2/"; then REGISTRY_PREFIX="$m/"; HUB_STATE="mirror $m"; break; fi
    done
  fi
  if reachable https://pypi.org/simple/pip/; then
    PIP_STATE="direct"
  else
    PIP_STATE="unreachable"
    for m in "${PYPI_MIRRORS[@]}"; do
      if reachable "${m}pip/"; then PIP_INDEX="$m"; PIP_STATE="mirror $m"; break; fi
    done
  fi
  # The official CPU index is only usable when its file host answers too.
  if reachable https://download-r2.pytorch.org/whl/cpu/; then
    TORCH_CPU_INDEX="${TORCH_CPU_INDEXES[0]}"
  else
    for m in "${TORCH_CPU_INDEXES[@]:1}"; do
      if reachable "${m}torch/"; then TORCH_CPU_INDEX="$m"; break; fi
    done
  fi
  if reachable https://huggingface.co/; then MODEL_SOURCE="huggingface"
  elif reachable https://www.modelscope.cn/; then MODEL_SOURCE="modelscope"
  else MODEL_SOURCE="auto"; fi

  if [[ -n "${TZ:-}" ]]; then HOST_TZ="$TZ"
  elif [[ -f /etc/timezone ]]; then HOST_TZ="$(cat /etc/timezone)"
  elif have timedatectl; then HOST_TZ="$(timedatectl show -p Timezone --value 2>/dev/null || echo UTC)"
  elif [[ -L /etc/localtime ]]; then HOST_TZ="$(readlink /etc/localtime | awk -F/ '{print $(NF-1)"/"$NF}')"
  fi
  [[ -n "$HOST_TZ" ]] || HOST_TZ="UTC"
}

gpu_usable() { (( GPU_DRIVER == 1 && GPU_DOCKER == 1 && FORCE_CPU == 0 )); }

hub_image() {  # hub_image REPO[:TAG] -> with mirror prefix; official images get library/
  local ref="$1"
  if [[ -n "$REGISTRY_PREFIX" ]]; then
    [[ "$ref" == */* ]] || ref="library/$ref"
    echo "${REGISTRY_PREFIX}${ref}"
  else
    echo "$ref"
  fi
}

print_detection() {
  log "host: $OS $ARCH, ${MEM_GB} GB RAM (memory tier: $MEM_TIER), timezone $HOST_TZ"
  if gpu_usable; then log "gpu: $GPU_NAME (driver + container toolkit present) -> parser flavour gpu"
  elif (( GPU_DRIVER == 1 && FORCE_CPU == 1 )); then log "gpu: $GPU_NAME present, --cpu given -> CPU parser"
  elif (( GPU_DRIVER == 1 )); then warn "gpu: $GPU_NAME found but Docker cannot use it (no NVIDIA container toolkit / CDI spec) -> CPU parser"
  else log "gpu: none -> CPU parser (pipeline backend)"; fi
  log "docker hub: $HUB_STATE; pypi: $PIP_STATE; model weights: $MODEL_SOURCE; cpu torch wheels: ${TORCH_CPU_INDEX:-pypi fallback}"
}

# ── compose .env ───────────────────────────────────────────────────────────
write_compose_env() {
  local flavour compose_files profiles base_image
  if gpu_usable; then flavour=gpu; compose_files="docker-compose.yml:compose.gpu.yml"
  else flavour=cpu; compose_files="docker-compose.yml"; fi
  if [[ "$flavour" == gpu ]]; then base_image="$(hub_image vllm/vllm-openai:v0.21.0)"
  else base_image="$(hub_image python:3.12-slim)"; fi
  profiles=""; (( WITH_MODELS == 1 )) && profiles="local-models"

  if [[ -f "$COMPOSE_ENV" ]]; then
    log "keeping existing $COMPOSE_ENV (delete it to re-detect)"
    # Only the profile follows the command line on re-runs.
    set_kv "$COMPOSE_ENV" COMPOSE_PROFILES "$profiles"
    return
  fi
  cp "$COMPOSE_DIR/.env.example" "$COMPOSE_ENV"
  chmod 600 "$COMPOSE_ENV"
  set_kv "$COMPOSE_ENV" COMPOSE_FILE "$compose_files"
  set_kv "$COMPOSE_ENV" COMPOSE_PROFILES "$profiles"
  set_kv "$COMPOSE_ENV" MINERU_FLAVOR "$flavour"
  set_kv "$COMPOSE_ENV" MINERU_BASE_IMAGE "$base_image"
  set_kv "$COMPOSE_ENV" MINERU_MODEL_SOURCE "$MODEL_SOURCE"
  set_kv "$COMPOSE_ENV" QDRANT_IMAGE "$(hub_image qdrant/qdrant:v1.18.2)"
  set_kv "$COMPOSE_ENV" NEO4J_IMAGE "$(hub_image neo4j:5.26.28-community)"
  set_kv "$COMPOSE_ENV" OPENSEARCH_IMAGE "$(hub_image opensearchproject/opensearch:3.7.0)"
  if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    set_kv "$COMPOSE_ENV" VLLM_IMAGE "nvcr.io/nvidia/vllm:26.07-py3"
  else
    set_kv "$COMPOSE_ENV" VLLM_IMAGE "$(hub_image vllm/vllm-openai:v0.21.0)"
  fi
  [[ -n "$PIP_INDEX" ]] && set_kv "$COMPOSE_ENV" PIP_INDEX_URL "$PIP_INDEX"
  [[ -n "$TORCH_CPU_INDEX" ]] && set_kv "$COMPOSE_ENV" TORCH_CPU_INDEX_URL "$TORCH_CPU_INDEX"
  set_kv "$COMPOSE_ENV" TZ "$HOST_TZ"
  case "$MEM_TIER" in
    large)  set_kv "$COMPOSE_ENV" OPENSEARCH_JAVA_HEAP 4g; set_kv "$COMPOSE_ENV" OPENSEARCH_MEM_LIMIT 8g
            set_kv "$COMPOSE_ENV" NEO4J_HEAP_MAX_SIZE 2G;  set_kv "$COMPOSE_ENV" NEO4J_PAGECACHE_SIZE 2G
            set_kv "$COMPOSE_ENV" NEO4J_MEM_LIMIT 6g;      set_kv "$COMPOSE_ENV" QDRANT_MEM_LIMIT 4g ;;
    medium) set_kv "$COMPOSE_ENV" OPENSEARCH_JAVA_HEAP 2g; set_kv "$COMPOSE_ENV" OPENSEARCH_MEM_LIMIT 4g
            set_kv "$COMPOSE_ENV" NEO4J_HEAP_MAX_SIZE 1G;  set_kv "$COMPOSE_ENV" NEO4J_PAGECACHE_SIZE 1G
            set_kv "$COMPOSE_ENV" NEO4J_MEM_LIMIT 4g;      set_kv "$COMPOSE_ENV" QDRANT_MEM_LIMIT 2g ;;
    *)      set_kv "$COMPOSE_ENV" OPENSEARCH_JAVA_HEAP 1g; set_kv "$COMPOSE_ENV" OPENSEARCH_MEM_LIMIT 2g
            set_kv "$COMPOSE_ENV" NEO4J_HEAP_MAX_SIZE 1G;  set_kv "$COMPOSE_ENV" NEO4J_PAGECACHE_SIZE 512m
            set_kv "$COMPOSE_ENV" NEO4J_MEM_LIMIT 2g;      set_kv "$COMPOSE_ENV" QDRANT_MEM_LIMIT 1g ;;
  esac
  local password
  password="$(LC_ALL=C tr -dc 'A-Za-z0-9' </dev/urandom 2>/dev/null | head -c 24 || true)"
  [[ ${#password} -ge 16 ]] || password="$(date +%s%N | sha256sum | head -c 24)"
  set_kv "$COMPOSE_ENV" NEO4J_PASSWORD "$password"
  log "wrote $COMPOSE_ENV (flavour $flavour, compose files $compose_files)"
}

# ── app config ─────────────────────────────────────────────────────────────
write_app_config() {
  local password
  password="$(get_kv "$COMPOSE_ENV" NEO4J_PASSWORD)"
  if [[ -f "$APP_ENV" ]]; then
    log "keeping existing $APP_ENV"
    if [[ "$(get_kv "$APP_ENV" NEO4J_PASSWORD)" != "$password" ]]; then
      warn "NEO4J_PASSWORD in $APP_ENV differs from the compose .env; fix one of them if Neo4j refuses connections"
    fi
    return
  fi
  cp "$ROOT/config/knowledge-base.env.example" "$APP_ENV"
  chmod 600 "$APP_ENV"
  set_kv "$APP_ENV" NEO4J_PASSWORD "$password"
  set_kv "$APP_ENV" MINERU_BACKEND auto
  set_kv "$APP_ENV" KB_PARSE_ENABLED 1
  if (( WITH_MODELS == 1 )); then
    set_kv "$APP_ENV" KB_CONSOLE_SERVICES "database,mineru,embedding,visual_embedding,vlm,reranker"
    set_kv "$APP_ENV" VISUAL_EMBEDDING_ENABLED 1
    set_kv "$APP_ENV" KB_SEARCH_VISUAL 1
    set_kv "$APP_ENV" RERANKER_BASE_URL "http://127.0.0.1:8102/v1"
    set_kv "$APP_ENV" VISUAL_RERANKER_BASE_URL "http://127.0.0.1:8104/v1"
  else
    set_kv "$APP_ENV" KB_CONSOLE_SERVICES "database,mineru"
  fi
  log "wrote $APP_ENV"
  (( WITH_MODELS == 1 )) || warn "point EMBEDDING_*, VLM_* (and optionally RERANKER_BASE_URL) in $APP_ENV at your model endpoints before ingesting"
}

# ── containers ─────────────────────────────────────────────────────────────
compose() { (cd "$COMPOSE_DIR" && docker compose "$@"); }

verify_gpu_in_container() {
  # The image is built now; ask torch inside it whether the GPU really came
  # through. If not, drop the GPU overlay so the parser falls back to CPU
  # instead of crashing on a missing device.
  local image out
  image="carrel-mineru:$(get_kv "$COMPOSE_ENV" MINERU_VERSION)-gpu"
  out="$(docker run --rm --gpus all --entrypoint python3 "$image" -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || true)"
  if [[ "$out" != "True" ]]; then
    warn "the parser image cannot see the GPU (docker run --gpus all failed); starting the parser on CPU"
    set_kv "$COMPOSE_ENV" COMPOSE_FILE "docker-compose.yml"
  fi
}

wait_healthy() {  # wait_healthy SECONDS NAME...
  local deadline=$(( $(date +%s) + $1 )); shift
  local pending name status last=""
  while :; do
    pending=()
    for name in "$@"; do
      status="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$name" 2>/dev/null || echo missing)"
      [[ "$status" == "healthy" || "$status" == "running" ]] || pending+=("$name:$status")
    done
    (( ${#pending[@]} == 0 )) && return 0
    if (( $(date +%s) > deadline )); then
      warn "still not healthy: ${pending[*]}"
      return 1
    fi
    if [[ " ${pending[*]} " == *" carrel-mineru:"* ]]; then
      last="$(docker logs --tail 1 carrel-mineru 2>&1 | tail -c 160 || true)"
    fi
    log "waiting for ${pending[*]}${last:+ | $last}"
    sleep 15
  done
}

prepare_dirs() {
  # Create the bind-mount targets as the current user; Docker would otherwise
  # create them root-owned. OpenSearch runs as uid 1000 inside its container
  # and cannot chown, so its data dir is made world-writable when it is new.
  local d
  for d in runtime/mirror runtime/media runtime/mineru-output runtime/qdrant_data runtime/opensearch/data \
           runtime/neo4j/data runtime/neo4j/logs runtime/neo4j/plugins runtime/neo4j/import models/mineru logs; do
    mkdir -p "$ROOT/$d"
  done
  if [[ -z "$(ls -A "$ROOT/runtime/opensearch/data" 2>/dev/null)" ]]; then
    chmod 0777 "$ROOT/runtime/opensearch/data"
  fi
}

containers_up() {
  local flavour
  flavour="$(get_kv "$COMPOSE_ENV" MINERU_FLAVOR)"
  prepare_dirs
  log "building the parser image (flavour $flavour; the first build downloads a large base image)"
  compose build mineru
  [[ "$flavour" == gpu ]] && verify_gpu_in_container
  log "starting containers"
  compose up -d --remove-orphans
  local names=("${STORE_CONTAINERS[@]}" carrel-mineru)
  (( WITH_MODELS == 1 )) && names+=("${MODEL_CONTAINERS[@]}")
  log "waiting for the stores (a few minutes at most) and the parser (downloads its weights on first start)"
  wait_healthy 3600 "${names[@]}" || warn "check 'docker logs <name>' for the containers above"
}

# ── app ────────────────────────────────────────────────────────────────────
pick_python() {
  local cand
  for cand in "${PYTHON:-}" python3.12 python3.13 python3; do
    [[ -n "$cand" ]] || continue
    if have "$cand" && "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
      echo "$cand"; return 0
    fi
  done
  return 1
}

app_install() {
  local py
  py="$(pick_python)" || die "Python 3.12+ is required for the app (set PYTHON=/path/to/python3.12), or re-run with --no-app"
  if [[ ! -x "$ROOT/app/.venv/bin/python" ]]; then
    log "creating app/.venv with $py"
    "$py" -m venv "$ROOT/app/.venv"
  fi
  log "installing the app into app/.venv"
  if [[ -n "$PIP_INDEX" ]]; then export PIP_INDEX_URL="$PIP_INDEX"; fi
  "$ROOT/app/.venv/bin/python" -m pip install -q --upgrade pip
  local extras=""
  if [[ "$WITH_PDF_IMAGES" -eq 1 ]]; then
    extras="[pdf-images]"
    log "installing the optional pdf-images extra (PyMuPDF, AGPL-3.0; see NOTICE.md)"
  fi
  "$ROOT/app/.venv/bin/python" -m pip install -q -e "$ROOT/app$extras" pytest
  mkdir -p "$ROOT/runtime/mirror" "$ROOT/runtime/media" "$ROOT/logs"
  log "initialising the state database"
  (cd "$ROOT/app" && KB_ENV_FILE="$APP_ENV" KB_LOCAL_BASE_DIR="$ROOT" ./.venv/bin/kb init-db)
}

systemd_available() {
  [[ "$OS" == "Linux" ]] && have systemctl && systemctl --user show-environment >/dev/null 2>&1
}

app_services() {
  if (( NO_SYSTEMD == 1 )) || ! systemd_available; then
    log "systemd user units not installed; start the services by hand:"
    echo "    cd $ROOT/app && KB_ENV_FILE=$APP_ENV KB_LOCAL_BASE_DIR=$ROOT .venv/bin/python -m kb_server   # console"
    echo "    cd $ROOT/app && KB_ENV_FILE=$APP_ENV KB_LOCAL_BASE_DIR=$ROOT .venv/bin/python -m kb_search   # search API"
    echo "    KB_ENV_FILE=$APP_ENV $ROOT/scripts/kb-pipeline-scan.sh && $ROOT/scripts/kb-pipeline-worker-once.sh   # one ingest round"
    return
  fi
  log "installing and enabling the systemd user units"
  CARREL_HOME="$ROOT" "$ROOT/scripts/install-systemd.sh" --enable
  if have loginctl && [[ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]]; then
    warn "timers stop when you log out; run once:  sudo loginctl enable-linger $USER"
  fi
}

console_url() {
  local port host bind
  port="$(get_kv "$APP_ENV" KB_WEB_PORT)"; port="${port:-9800}"
  bind="$(get_kv "$APP_ENV" KB_WEB_HOST)"; bind="${bind:-127.0.0.1}"
  if [[ "$bind" == "127.0.0.1" || "$bind" == "localhost" || "$bind" == "::1" ]]; then
    echo "http://127.0.0.1:$port"
    return
  fi
  host="$(hostname -I 2>/dev/null | awk '{print $1}')"
  [[ -n "$host" ]] || host="$(ipconfig getifaddr en0 2>/dev/null || echo 127.0.0.1)"
  echo "http://$host:$port"
}

# ── commands ───────────────────────────────────────────────────────────────
cmd_up() {
  have docker || die "docker is required (https://docs.docker.com/engine/install/)"
  docker compose version >/dev/null 2>&1 || die "docker compose v2 is required"
  docker info >/dev/null 2>&1 || die "docker daemon not reachable (is it running, is your user in the docker group?)"
  have curl || die "curl is required"
  detect
  print_detection
  write_compose_env
  containers_up
  if (( NO_APP == 1 )); then
    log "containers are up (--no-app: venv, config and units skipped)"
    return
  fi
  write_app_config
  app_install
  app_services
  echo
  log "done. Console: $(console_url)"
  log "put documents under $ROOT/runtime/mirror/<folder>/ and enable the folder in the console."
  log "to reach the console from other devices set KB_WEB_HOST=0.0.0.0 and KB_WEB_TOKEN in $APP_ENV, then restart carrel-web.service"
}

cmd_detect() {
  detect
  print_detection
  local flavour="cpu"; gpu_usable && flavour="gpu"
  log "would write: MINERU_FLAVOR=$flavour, QDRANT_IMAGE=$(hub_image qdrant/qdrant:v1.18.2), model source $MODEL_SOURCE${PIP_INDEX:+, pip index $PIP_INDEX}"
}

cmd_status() {
  if [[ -f "$COMPOSE_ENV" ]]; then compose ps; else docker ps --filter name=carrel- ; fi
  local url; url="$(console_url)"
  if curl -sS -m 5 "$url/api/health" >/dev/null 2>&1; then
    log "console health ($url/api/health):"; curl -sS -m 5 "$url/api/health"; echo
  else
    warn "console not reachable at $url"
  fi
}

cmd_down() {
  if [[ -f "$COMPOSE_ENV" ]]; then
    # --profile so the model servers go too, whatever COMPOSE_PROFILES says now
    compose --profile local-models down --remove-orphans
  else
    docker rm -f "${STORE_CONTAINERS[@]}" carrel-mineru "${MODEL_CONTAINERS[@]}" >/dev/null 2>&1 || true
    docker network rm carrel >/dev/null 2>&1 || true
  fi
  log "containers stopped and removed (data under runtime/ kept)"
}

cmd_purge() {
  cmd_down
  if systemd_available; then CARREL_HOME="$ROOT" "$ROOT/scripts/install-systemd.sh" --uninstall || true; fi
  local img
  for img in $(docker images --format '{{.Repository}}:{{.Tag}}' | grep '^carrel-mineru:' || true); do
    docker image rm "$img" >/dev/null 2>&1 && log "removed image $img" || true
  done
  if (( PURGE_IMAGES == 1 )) && [[ -f "$COMPOSE_ENV" ]]; then
    for key in QDRANT_IMAGE NEO4J_IMAGE OPENSEARCH_IMAGE MINERU_BASE_IMAGE VLLM_IMAGE; do
      img="$(get_kv "$COMPOSE_ENV" "$key")"
      [[ -n "$img" ]] || continue
      docker image rm "$img" >/dev/null 2>&1 && log "removed image $img" || true
    done
  fi
  if (( PURGE_DATA == 1 )); then
    if (( YES == 0 )); then
      read -r -p "delete $ROOT/runtime and $ROOT/models (all indexed data and downloaded weights)? [y/N] " ans
      [[ "$ans" == y || "$ans" == Y ]] || die "aborted"
    fi
    # Containers may have created root-owned files; try plain rm first, then
    # a throwaway container with the same mounts.
    rm -rf "$ROOT/runtime" "$ROOT/models" 2>/dev/null || \
      docker run --rm -v "$ROOT:/work" alpine sh -c 'rm -rf /work/runtime /work/models' >/dev/null 2>&1 || \
      warn "could not delete runtime/ or models/; remove them with sudo"
    log "runtime/ and models/ deleted"
  fi
  rm -f "$COMPOSE_ENV"
  log "purged (config/knowledge-base.env kept)"
}

case "$CMD" in
  up)     cmd_up ;;
  detect) cmd_detect ;;
  status) cmd_status ;;
  down)   cmd_down ;;
  purge)  cmd_purge ;;
esac
