# API and configuration

This file describes the calling conventions of the Carrel search service (`app/kb_search`); the server code is the authority on the API.

## Connection

Requires Python 3.9+ and the ability to reach port 9810 at the chosen address from this machine. The client uses only the standard library; it does not call SSH, does not scan configuration directories for keys, does not install dependencies and does not modify the server.

The default address is `http://127.0.0.1:9810`. When the service runs on another machine, replace the address with a host name that can reach it, or its name on a private network (Tailscale, for example), and confirm that port 9810 is reachable; the client does not switch or merge addresses on its own.

Environment variables can be used directly: `CARREL_SEARCH_BASE_URL` sets the address and `CARREL_SEARCH_TOKEN` the token; without the latter, `KB_SEARCH_TOKEN` is read. A real token must never be written into the skill, a question, command arguments or result files.

An optional JSON config is shown in [config.example.json](config.example.json). Point to it with `--config /absolute/path/config.json` or set `CARREL_SEARCH_CONFIG`; when neither is given, `~/.config/carrel-search/config.json` is tried. The example contains no real token. The config may add `token_file`, pointing to a file that holds the token separately; environment variables take precedence. The script never generates a config or reads keys on the server.

Check the connection with `python3 "$SKILL_DIR/scripts/carrel_search.py" health` (optionally with `--base-url`). Authentication still uses the separately stored token.

The command-level `--base-url` takes precedence over the environment and the config; `--timeout` overrides the config, defaulting to 60 seconds with no automatic retries. Global arguments go before the subcommand. 401/403 means the authentication must be fixed and a connection error means reachability must be checked; neither should be read as the knowledge base having no content.

Below, `$SKILL_DIR` is the actual skill directory and `$WORK_DIR` the existing working directory of the current task. These variables must be set by the caller; IDs/versions in JSON requests come from actual API responses and are never copied from placeholders.

## Commands and requests

| Command | HTTP | Purpose |
|---|---|---|
| `health` | GET `/health` | Status of the service and its listed dependencies; does not prove every branch is healthy |
| `catalog` | GET `/catalog` | Current KB names, IDs and capabilities; not required before every query |
| `search` | POST `/search` | First retrieval, or a new sub-question |
| `context` | POST `/context` | Read back a range of original text from a given version of the same document |
| `image` | GET `/image/{kb_id}/{point_id}` | Original image bytes; save first, then view |
| `crop` | POST `/crop` | Deterministic crop of an already confirmed original image |
| `neighbors` | POST `/graph/neighbors` | One-hop relations of an entity in the current graph: predicate, direction, weight, far-end entity, evidence chunks; the agent decides the next hop itself |

### Search

```bash
python3 "$SKILL_DIR/scripts/carrel_search.py" search --question "<the user's actual question>"
python3 "$SKILL_DIR/scripts/carrel_search.py" search --request "$WORK_DIR/request.json" --output "$WORK_DIR/result.json"
```

Request sketch; omit the fields that are not actually needed:

```json
{
  "question": "<the user's actual question>",
  "kbs": ["<kb id from the catalog>"],
  "top_k": 12,
  "context": true,
  "explain": false,
  "hints": {
    "doc_ids": ["<confirmed document id>"],
    "rel_paths": ["<full rel_path as returned by the API>"],
    "content_version": "<content_version from the original response>",
    "block_types": ["table"]
  }
}
```

`question` is required; without `kbs` the service picks the knowledge bases. `top_k` is 1–50 and defaults to the service setting. `context` is a boolean for whether neighbouring chunks are fetched, not a conversation history. `explain` only adds retrieval explanations and generates no answer.

`hints.doc_ids/rel_paths/content_version` are hard filters and several fields intersect; do not guess full paths from file names. `block_types` is a soft preference. Check `hints_used/hints_ignored/hints_scope`; subject and date are not supported hard-filter keys.

For image queries use `search --question "the actual image query" --image /absolute/path/query.png`; the script handles the base64 encoding. The raw image file is capped at 9,000,000 bytes, matching the API's base64 limit. Image-to-image results still need their textual constraints checked; do not assume that vector similarity already satisfies every combined image-and-text condition.

### Fetching context

```bash
python3 "$SKILL_DIR/scripts/carrel_search.py" context --request "$WORK_DIR/context.json" --output "$WORK_DIR/context-result.json"
```

```json
{
  "kb_id": "<from the hit>",
  "doc_id": "<from the hit>",
  "content_version": "<from the hit>",
  "chunk_from": 5,
  "chunk_to": 7
}
```

The range is taken from the actual hit's chunk index plus the neighbouring chunks needed; a page number is not a chunk_index. Start and end are non-negative and ascending, and the service requires `chunk_to - chunk_from <= 60`; usually start with a smaller range. This endpoint does not apply the 6,000-token text budget of `/search`, so avoid fetching an overly large range at once. When an old version yields no material, search again; do not mix old and new text.

### Original images and crops

```bash
python3 "$SKILL_DIR/scripts/carrel_search.py" image --kb "<kb id>" --point-id "<point_id of the image hit>" --output "$WORK_DIR/source-image.png"
python3 "$SKILL_DIR/scripts/carrel_search.py" crop --request "$WORK_DIR/crop.json" --output "$WORK_DIR/detail.png"
```

```json
{
  "kb_id": "<from the image hit>",
  "point_id": "<from the image hit>",
  "bbox": [0.1, 0.2, 0.8, 0.9],
  "pad": 16
}
```

`bbox` is given as 0–1 fractions or 0–1000 per-mille of the original image, not pixels; `pad` is 0–200 pixels. Look at the original image first, then locate the region. The original is saved with the service's raw bytes and the extension does not change the encoding; the returned `mime_type` is authoritative. Crops are output as PNG. The client returns the absolute path, content hash, source, dimensions and crop box; it does not hand the image to the model merely as base64 text.

### Graph neighbourhood

```bash
python3 "$SKILL_DIR/scripts/carrel_search.py" neighbors --kb "<kb id>" --entity "<entity title or alias>" --limit 20
python3 "$SKILL_DIR/scripts/carrel_search.py" neighbors --kb "<kb id>" --entity-id "<entity_id from the previous response>" --type has_stage --direction out
```

JSON can also be passed with `--request`:

```json
{
  "kb_id": "<kb id>",
  "entity_id": "<entity_id from the previous response>",
  "limit": 20,
  "types": ["<predicate>"],
  "direction": "both"
}
```

`entity_id` can be replaced by `entity` (title or alias); `types` may be omitted, and `direction` is one of `both/out/in`, defaulting to `both`. Names are matched exactly, case-insensitively, against titles or aliases.
When several entities share a name, the service picks the one with the larger `degree` among the matching candidates as `entity` and lists the others in `matches`; the caller must still check type, scope and sources, and the connection count is no substitute for entity disambiguation.
When nothing is found, `found=false` and `candidates` offers a few vector-similar entities (with IDs) to check and choose from; this is not a confirmed hit. A knowledge base without a current graph returns 404.

Each entry in the response's `neighbors[]` contains:

- `type` (the predicate), `directed` and `direction`: only for relations with `directed=true` does `out` mean the current entity is the subject and `in` that the far end is. `directed=false` is an undirected association whose `in/out` only reflects the storage direction in the graph database, not a semantic subject/object or causality. Query undirected relations with `both` so that no association is missed because of the storage direction.
- `weight/npmi/cooccur`, `description` (a model summary, not original text), `type_violation` (a flag that an endpoint type is out of bounds).
- `other`: the far-end entity's ID, title, type, scope and so on.
- `evidence[]`: at most 3 chunks per relation, with `point_id/doc_id/chunk_index/content_version/rel_path/page_idx/active`. Take the top-level `kb_id` of the response, then build the range parameters of `/context` from the evidence's `doc_id/content_version` and `chunk_index`; do not pass the whole evidence object as the request. `active=false` means the evidence point has been deactivated and cannot serve as current grounds.

**Coverage and truncation:** `limit` defaults to 20 with a range of 1–100; relations are returned in descending weight order. `count` is only the number returned this time; the current API has no `total/has_more/truncated` or pagination parameters.
When the count reaches `limit`, more relations may remain; narrow with `types` according to the question or raise `limit`. Repeating the same request does not yield a next page. `entity.degree` is no substitute for the total number of relations under the current filters either.
`entity.docs` lists at most 12 document paths as source leads, not a complete document list. The graph also filters out some relations, so neighbourhood results cannot prove "all results" in the material; completeness must be checked against the original directory or ledger.
Multi-hop traversal is done by the caller step by step according to the question, checking the relation and the original evidence at every step; the time per hop varies with the entity and the running load.

## Reading responses

JSON commands: `{"call_id":"<id of this call>","operation":"search","result":{<the service's original response>}}`. With `--output` the complete JSON is saved to a new file and the terminal shows only `response_file`; the file must then be read, and a file pointer must not be treated as evidence already read. For image commands, `result.path` points to the actual file, which needs an image tool to view.

Descriptive names such as `Sources` correspond to the actual lower-case JSON keys `sources/entities/relationships/specs/pages`. Keep `retrieval_summary`, the complete sources and the truncation state; the client does not rerank, does not delete evidence a second time and does not widen the knowledge base selection on its own. `call_id` only distinguishes multiple responses and is not a server-side document identity.

On failure, a JSON error goes to stderr with a non-zero exit code; empty results are never returned silently. All output files are written without overwriting, and the output directory must already exist.
