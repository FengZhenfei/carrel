# One application image for both storage editions. Model servers stay external.
ARG PYTHON_IMAGE=python:3.12-slim
FROM ${PYTHON_IMAGE}
ARG WITH_PDF_IMAGES=0
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG DEBIAN_MIRROR=http://deb.debian.org
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH KB_LOCAL_BASE_DIR=/data \
    KB_ENV_FILE=/data/config/knowledge-base.env KB_RUNTIME=container \
    KB_APP_DIR=/opt/carrel/app KB_PYTHON=/opt/venv/bin/python \
    KB_SCRIPTS_DIR=/opt/carrel/scripts KB_CONFIG_TEMPLATE=/opt/carrel/config/knowledge-base.env.example \
    TREE_SITTER_LANGUAGE_PACK_CACHE_DIR=/opt/carrel/code-parsers
RUN sed -i "s|http://deb.debian.org|${DEBIAN_MIRROR}|g" /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::http::Timeout=30 -o Acquire::Retries=2 update \
    && apt-get -o Acquire::http::Timeout=30 -o Acquire::Retries=2 install -y --no-install-recommends bash curl procps ca-certificates \
    && rm -rf /var/lib/apt/lists/* && python -m venv /opt/venv
WORKDIR /opt/carrel
COPY app/requirements*.lock ./app/
RUN pip install --index-url "$PIP_INDEX_URL" -r app/requirements.lock -r app/requirements-light.lock \
    && if [ "$WITH_PDF_IMAGES" = 1 ]; then pip install --index-url "$PIP_INDEX_URL" -r app/requirements-pdf-images.lock; fi
COPY app/pyproject.toml ./app/
COPY app/kb_pipeline ./app/kb_pipeline
COPY app/kb_search ./app/kb_search
COPY app/kb_server ./app/kb_server
COPY app/kb_runtime ./app/kb_runtime
RUN pip install --no-deps ./app
COPY LICENSE NOTICE.md ./
COPY THIRD_PARTY_LICENSES ./THIRD_PARTY_LICENSES
COPY scripts ./scripts
RUN --mount=type=bind,source=deployment/compose/app,target=/build-input \
    if [ -d /build-input/code-parsers ]; then \
        mkdir -p "$TREE_SITTER_LANGUAGE_PACK_CACHE_DIR" && \
        cp -a /build-input/code-parsers/. "$TREE_SITTER_LANGUAGE_PACK_CACHE_DIR/"; \
    fi && python scripts/prepare-code-parsers.py
COPY config/knowledge-base.env.example ./config/knowledge-base.env.example
RUN mkdir -p /data
EXPOSE 9800 9810
ENTRYPOINT ["python", "-m", "kb_runtime"]
