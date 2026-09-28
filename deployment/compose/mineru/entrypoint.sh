#!/usr/bin/env bash
# Carrel parser entrypoint.
#
#   1. detect the GPU and pick the backend (detect.py)
#   2. download the weights for that backend into /models on first start
#   3. publish the decision to /data/output/carrel-mineru.json for the
#      pipeline (MINERU_BACKEND=auto) and the console
#   4. exec mineru-api with the matching arguments
set -euo pipefail

MODELS_DIR="${MINERU_MODELS_DIR:-/models}"
SHARED_DIR="${CARREL_SHARED_DIR:-/data/output}"
MINERU_VERSION="${MINERU_VERSION:-unknown}"
mkdir -p "$MODELS_DIR" "$SHARED_DIR"

# Weights and MinerU's own config live on the bind mount, so a recreated
# container never downloads twice.
export HF_HOME="$MODELS_DIR/huggingface"
export MODELSCOPE_CACHE="$MODELS_DIR/modelscope"
export MINERU_TOOLS_CONFIG_JSON="$MODELS_DIR/mineru.json"

# 1. decide
eval "$(python3 /usr/local/bin/carrel-mineru-detect.py --shell)"
echo "[carrel-mineru] backend=$CARREL_BACKEND device=$CARREL_DEVICE gpu='${CARREL_GPU_NAME:-none}' total=${CARREL_GPU_TOTAL_GB:-?}GB util=$CARREL_GPU_UTIL ($CARREL_REASON)"
case "$CARREL_BACKEND" in
  vlm-engine|hybrid-engine)
    if [[ "$CARREL_CAPABLE" != "1" ]]; then
      echo "[carrel-mineru] $CARREL_BACKEND was forced but no capable GPU is visible; set MINERU_BACKEND_POLICY=pipeline or fix the GPU passthrough" >&2
      exit 1
    fi
    model_type=vlm ;;
  pipeline)
    model_type=pipeline ;;
  *)
    echo "[carrel-mineru] unknown backend '$CARREL_BACKEND'" >&2; exit 1 ;;
esac

# 2. weights (once per backend and MinerU version)
marker="$MODELS_DIR/.carrel-models-${model_type}-${MINERU_VERSION}"
if [[ ! -f "$marker" ]]; then
  source="${MINERU_MODEL_SOURCE:-auto}"
  echo "[carrel-mineru] downloading $model_type weights (source=$source) into $MODELS_DIR ..."
  mineru-models-download -s "$source" -m "$model_type"
  touch "$marker"
  echo "[carrel-mineru] weights ready"
fi
if [[ ! -f "$MINERU_TOOLS_CONFIG_JSON" ]]; then
  echo "[carrel-mineru] $MINERU_TOOLS_CONFIG_JSON missing after download" >&2
  exit 1
fi
# From here on MinerU must read the local copies, never the network.
export MINERU_MODEL_SOURCE=local
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MINERU_DEVICE_MODE="$CARREL_DEVICE"

# 3. publish
python3 /usr/local/bin/carrel-mineru-detect.py --json \
  | python3 -c 'import json,sys,time; d=json.load(sys.stdin); d["started_at"]=int(time.time()); d["port"]=8000; print(json.dumps(d, ensure_ascii=False, indent=2))' \
  > "$SHARED_DIR/carrel-mineru.json.tmp"
mv -f "$SHARED_DIR/carrel-mineru.json.tmp" "$SHARED_DIR/carrel-mineru.json"

# 4. serve
args=(--host 0.0.0.0 --port 8000)
if [[ "$model_type" == "vlm" ]]; then
  args+=(
    --enable-vlm-preload true
    --gpu-memory-utilization "$CARREL_GPU_UTIL"
    --max-model-len "${MINERU_MAX_MODEL_LEN:-8192}"
    --max-num-seqs "${MINERU_MAX_NUM_SEQS:-16}"
    --max-num-batched-tokens "${MINERU_MAX_NUM_BATCHED_TOKENS:-16384}"
    --mm-processor-cache-gb "${MINERU_MM_PROCESSOR_CACHE_GB:-1}"
    --dtype "${MINERU_DTYPE:-bfloat16}"
    --enforce-eager
  )
else
  args+=(--enable-vlm-preload false)
fi
echo "[carrel-mineru] exec mineru-api ${args[*]}"
exec mineru-api "${args[@]}"
