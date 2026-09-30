# Third-party notices

Carrel is released under the MIT License (see `LICENSE`). It builds on the
following projects, which keep their own licenses.

## Code and prompts included in this repository

- **GraphRAG** (Microsoft, MIT). The record grammar, few-shot examples and
  gleaning loop of the entity-extraction and description-summary prompts in
  `app/kb_pipeline/graph/prompts.py` derive from GraphRAG 3.1.2
  (`graphrag/prompts/index/extract_graph.py`, `summarize_descriptions.py`).
  Copyright (c) Microsoft Corporation. The upstream license text is in
  `THIRD_PARTY_LICENSES/GraphRAG-MIT.txt`; the prompts were rewritten and
  extended here (type menus, quantity limits, measurement facts).
- **markdown-it** 15.0.2 (MIT). Copyright (c) 2014 Vitaly Puzrin, Alex
  Kocharin. `app/kb_server/static/vendor/markdown-it.umd.min.js` is the
  unmodified browser build; the console's chunk preview renders chunk text
  with it. The license text is in `THIRD_PARTY_LICENSES/markdown-it-MIT.txt`.
- **KaTeX** 0.18.10 (MIT). Copyright (c) 2013-2020 Khan Academy and other
  contributors. `app/kb_server/static/vendor/katex.min.js`, `katex.min.css`
  and `fonts/*.woff2` are unmodified files of the release; the same preview
  renders formulas with them. The license text is in
  `THIRD_PARTY_LICENSES/KaTeX-MIT.txt`.
- **Qwen3 reranker chat templates** (Alibaba Cloud, Apache-2.0).
  `deployment/compose/assets/qwen3_reranker.jinja` and
  `qwen3_vl_reranker.jinja` are the templates published with the
  Qwen3-Reranker and Qwen3-VL-Reranker model cards, unmodified. The license
  text is in `THIRD_PARTY_LICENSES/Apache-2.0.txt`.

## Services run as separate containers

These are not part of this repository. `deployment/compose` pulls or builds
them; each is used unmodified over its network API.

- **MinerU** (OpenDataLab, AGPL-3.0) — document parser. The image built from
  `deployment/compose/mineru/Dockerfile` installs the `mineru` package and its
  models; Carrel talks to it over HTTP only.
- **vLLM** (Apache-2.0) — base image of the parser and of the optional local
  model servers.
- **Qdrant** (Apache-2.0) — vector store.
- **OpenSearch** (Apache-2.0) — keyword index.
- **Neo4j Community Edition** (GPL-3.0) — graph projection, used through the
  Bolt protocol with the official `neo4j` Python driver (Apache-2.0).

## Model weights

Weights are downloaded by the user or by the parser container at first start
and are never committed. Check each model's license before use:
MinerU2.5 (OpenDataLab, AGPL-3.0), Qwen3-Embedding, Qwen3-Reranker,
Qwen3-VL-Embedding, Qwen3-VL-Reranker and Qwen3-VL-8B-Instruct (Alibaba
Cloud, Apache-2.0).

## Python dependencies

See `app/pyproject.toml`. The required dependencies are available under
permissive licenses (MIT, BSD, Apache-2.0). One optional extra is not:

- **PyMuPDF** (`carrel[pdf-images]`, AGPL-3.0 or a commercial license from
  Artifex). It is imported only by `app/kb_search/images.py` to render
  original-resolution pictures and crops from PDFs; without it the search
  service serves the parser's cached images and reports the fallback in
  `/health`. `deploy.sh` installs it only when run with `--with-pdf-images`;
  leave it out if the AGPL terms do not suit your deployment.

## Model behaviour notes

- The optional local `vl-reranker` server starts vLLM with
  `--trust-remote-code`, which executes the model repository's own Python
  code; it applies only to the weights you place under `models/`.
- `deploy.sh` may select public mirror registries and package indexes when
  the primary hosts are unreachable from your network; it prints what it
  chose and the values are editable in `deployment/compose/.env`.
