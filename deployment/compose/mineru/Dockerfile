# Carrel parser image: MinerU's API server with both backends installed.
#
#   FLAVOR=gpu  (default) starts from the vLLM OpenAI image so vlm-engine works
#               out of the box; the pipeline backend is installed too, so the
#               entrypoint can fall back to it on a small GPU.
#   FLAVOR=cpu  starts from a slim Python image with CPU-only torch and the
#               pipeline backend only. Pass BASE_IMAGE=python:3.12-slim with it.
#
# The entrypoint decides which backend to run at container start; see
# entrypoint.sh and detect.py.
ARG BASE_IMAGE=vllm/vllm-openai:v0.21.0
FROM ${BASE_IMAGE}

ARG FLAVOR=gpu
ARG MINERU_VERSION=3.4.4
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MINERU_VERSION=${MINERU_VERSION} \
    CARREL_MINERU_FLAVOR=${FLAVOR}

RUN set -eux; \
    if [ "$FLAVOR" = "cpu" ]; then \
        apt-get update; \
        apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 curl; \
        rm -rf /var/lib/apt/lists/*; \
        # CPU-only torch keeps the image small; if the wheel index is not
        # reachable from this network, fall back to PyPI's build (works, larger).
        if ! python3 -m pip install --index-url "${TORCH_CPU_INDEX_URL}" torch torchvision; then \
            echo "carrel: ${TORCH_CPU_INDEX_URL} unreachable, installing torch from PyPI instead"; \
            python3 -m pip install torch torchvision; \
        fi; \
        # six is imported by the pipeline backend but not declared; the vLLM
        # base image happens to ship it, the slim image does not.
        python3 -m pip install "mineru[pipeline]==${MINERU_VERSION}" six; \
    else \
        # The vlm and vllm extras keep Transformers on the MinerU-compatible
        # line and vLLM below 0.22; lmcache from the base image conflicts.
        python3 -m pip install "mineru[pipeline,vlm,vllm]==${MINERU_VERSION}"; \
        python3 -m pip uninstall -y lmcache || true; \
    fi

COPY entrypoint.sh /usr/local/bin/carrel-mineru-entrypoint
COPY detect.py /usr/local/bin/carrel-mineru-detect.py
RUN chmod +x /usr/local/bin/carrel-mineru-entrypoint

ENTRYPOINT ["/usr/local/bin/carrel-mineru-entrypoint"]
