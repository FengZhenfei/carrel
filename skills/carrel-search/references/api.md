# API and configuration

This file describes the calling conventions of the Carrel search service (`app/kb_search`) and of the bundled client; the server code is the authority on the API.

## Connection

Requires Python 3.9+ and the ability to reach port 9810 at the chosen address from this machine. The client uses only the standard library; it does not call SSH, does not scan configuration directories for keys, does not install dependencies and does not modify the server.

The default address is `http://127.0.0.1:9810`. When the service runs on another machine, replace the address with a host name that can reach it, or its name on a private network (Tailscale, for example), and confirm that port 9810 is reachable; the client does not switch or merge addresses on its own.

Environment variables can be used directly: `CARREL_SEARCH_BASE_URL` sets the address and `CARREL_SEARCH_TOKEN` the token; without the latter, `KB_SEARCH_TOKEN` is read. A real token must never be written into the skill, a question, command arguments or result files.

An optional JSON config is shown in [config.example.json](config.example.json). Point to it with `--config /absolute/path/config.json` or set `CARREL_SEARCH_CONFIG`; when neither is given, `~/.config/carrel-search/config.json` is tried. The example contains no real token. The config may add `token_file`, pointing to a file that holds the token separately; environment variables take precedence. The script never generates a config or reads keys on the server.

`work_dir` in the config (or `CARREL_SEARCH_WORK_DIR`) is where complete responses and fetched pictures are kept; without it the client uses `carrel-search/` under the system temporary directory, created for the current user only. Files older than a day are removed on the next call.

Check the connection with `python3 "$SKILL_DIR/scripts/carrel_search.py" health` (optionally with `--base-url`). Authentication still uses the separately stored token.

The command-level `--base-url` takes precedence over the environment and the config; `--timeout` overrides the config, defaulting to 60 seconds with no automatic retries. Global arguments go before the subcommand. 401/403 means the authentication must be fixed and a connection error means reachability must be checked; 503 means the service could not reach its own retrieval back ends (or its state database was busy) and the request can be retried later. None of these should be read as the knowledge base having no content.

The service bounds each search by its request budget (`KB_SEARCH_REQUEST_BUDGET`, 45 seconds by default), below the client's default timeout; keep `--timeout` above the budget. When time runs short the service still answers, skipping the later stages (widening, reranking, neighbouring context, subject neighbourhoods) and listing them under "gaps in this retrieval".

Below, `$SKILL_DIR` is the actual skill directory.

## Commands

| Command | HTTP | Purpose and arguments |
|---|---|---|
| `health` | GET `/health` | Status of the service and its listed dependencies, no token needed; does not prove every branch is healthy |
| `catalog` | GET `/catalog` | Current knowledge base names, IDs, domains, subject types, entity types, document samples and whether each has a graph; not required before every query |
| `search` | POST `/search` | Retrieval. `--question` (required, at most 2000 characters), `--kb` (repeatable; without it the service routes), `--top-k 1..50`, `--block-type` (repeatable, a soft preference), `--in-doc <ref>` (repeatable; search only inside the document of that source), `--rel-path` (repeatable; a path inside the knowledge base, a hard filter), `--image file` (search by picture), `--no-context` (no neighbouring chunks), `--explain` |
| `context` | POST `/context` | Read back a range of original text from the same version of a document. `--ref <ref>` (S, F or N) with `--before` / `--after` (1 chunk each by default; a stitched hit is read over its whole range) |
| `image` | GET `/image/{kb_id}/{point_id}` | The original picture, saved to the work directory; the path is printed. `--ref <ref>` |
| `crop` | POST `/crop` | Deterministic crop of the original picture. `--ref <ref> --bbox x0,y0,x1,y1`, optionally `--pad 0..200` |
| `neighbors` | POST `/graph/neighbors` | One-hop relations of an entity. `--ref <ref>`, or `--kb` with `--entity name` / `--entity-id`; `--type` (repeatable, only these predicates), `--direction both/out/in`, `--limit 1..100` (20 by default) |
| `entities` | POST `/graph/entities` | Entities by type / upper class / name. `--kb`, `--type` (repeatable), `--parent-type`, `--name`, `--limit 1..200` (50 by default), `--offset` |
| `facts` | POST `/graph/facts` | Qualified facts by subject / property. `--ref <ref>`, or `--kb` with `--subject` / `--subject-id`; `--property`, `--match auto/exact/contains`, `--limit`, `--offset`; at least one of subject and property |
| `show` | — | `show <call id>:<label>` (several at once are fine): the full text of a source or page, every field (JSON) of any other entry, and of sources and pages too with `--json`; `show <call id>` prints the compact view of that call again |

Every command also takes `--json` (print the complete response instead of the compact view) and `--output new-file` (save another copy; for pictures, save there instead). `--request file` still passes a JSON request body directly and is rarely needed.

## Compact view and labels

Each JSON command prints a compact view. Its first line reads `call <call id> · <command> · knowledge base · status…` and its last line gives the location of the complete response. The complete response (the service's original response plus the citations and numbers the client adds) is kept, with nothing dropped, in `<call id>.json` in the work directory; it normally need not be read, and `show` prints what the view leaves out.

The compact view of `search` is kept within what an agent's tool shows in one piece: the best hits are always printed in full, later ones by their opening words when space runs short, with their labels listed at the end; the chunks before and after a hit are listed by label and place only; the entity and keyword lines that picture chunks carry for indexing are not printed; source pages that hold only a file name and entity names are not printed; entities and relations that read the same are listed once, twelve of each at most. `show` prints any of these in full.

Labels:

| Prefix | What it is | Appears in |
|---|---|---|
| `S` | A source chunk | `search`, `context` |
| `F` | A fact | `search`, `facts` |
| `P` | A compiled page | `search` |
| `E` | An entity | `search`, `entities` |
| `R` | A relation matching the question | `search` |
| `H` | The one-hop neighbourhood of a subject; `H1.2` is the second relation of the first subject | `search` |
| `N` | A relation returned by a neighbourhood lookup | `neighbors` |

What `--ref` takes: `context` / `image` / `crop` need a chunk, so `S` stands for itself and `N` and `F` for their first evidence chunk (for a fact, the chunk holding its value comes first when that can be told); `neighbors` / `facts` need an entity, so `E` and `H` stand for themselves and `H1.2` and `N` for the far end of the relation. Pointing at the wrong kind is reported as an error. Labels are independent per call, which is why a reference carries the call id.

A citation is printed in full the first time it appears in a view; later entries with the same citation say "same citation as S3", meaning the citation printed on that entry.

Left out of the compact view: IDs, scores, metadata of unselected candidates, the model-written descriptions of entities and relations, and the structured extraction fields of pictures. All of them are in the complete response.

## Endpoint details

### Search

The response (visible with `--json` or `show`) has the top-level keys `question/kbs/kb_names/retrieval_summary/sources/specs/pages/entities/relationships/neighborhoods/doc_aggs`.

- `sources[]`: `role=hit` is a hit and carries `accepted` (true only when its rerank score passed the threshold); `role=neighbor` is a chunk next to a hit, and `of` says which hit it belongs to. Each carries `n/kb_id/doc_id/rel_path/chunk_index/content_version/point_id/block_type/place/text/token_count`; picture chunks add `visual` (description, text in the picture, extracted facts, confidence, number of conflicts) and stitched ones `stitched` (`chunk_from/chunk_to/pieces[]`).
- `specs[]`: qualified facts. The compact view prints each as "subject · property = value unit @ conditions · time"; the time is `when` / `valid_from`, which is the document's date for a fact whose material states no time of its own. Also `series_text`, `conflict`, `verified/sources_active`, `kinds`, `comparable`, `unit_canonical`, `ref_min/ref_max` and `flag`. `evidence[]` are the chunks the fact comes from (3 at most, with `doc_id/chunk_index/content_version/place/active`): a fact comes from an extraction unit that may span several chunks, and when the value's wording appears in exactly one of them that chunk comes first with `located=true` and the citation takes its place; otherwise the citation stops at the document.
- `pages[]`: compiled pages, `kind` being `subject` / `timeline` / `source`; `summary` is the overview, and timeline pages and subject pages with series add `text`.
- `entities[]` / `relationships[]`: graph items matching the question (at most 24 each); `description` is a model summary. `entities[].id` shares its namespace with the `entity_id` of the graph endpoints.
- `neighborhoods[]`: the one-hop neighbourhood of the subjects, 4 at most by default. Entities the question names come first (a title or alias appearing whole in the question, case and spaces ignored; a name of two or three characters without letters or digits counts only when the entity is also a graph route match), then the graph route's best matches fill up. Each carries `id/title/type/named/relations (total)/facts (count)/predicates (counts per kind)/neighbors`; `neighbors[]` are its eight strongest relations (`type/direction/directed/weight/other{id,title,type,degree}`), one line per far end. These are leads without evidence. A search scoped to documents (`--in-doc`, `--rel-path`) does not carry the block.
- `retrieval_summary`: `evidence_state` (`accepted` / `diagnostic` / `unranked`), `no_relevant_content`, `degraded`, `routing` (`chosen/weak/widened`), `buckets` and `selection.quota_filled`, `timings_ms`.

`--in-doc` and `--rel-path` are hard filters; `--block-type` is a soft preference. Subjects and dates are not supported filters: write a time range into the question (it is a boost, not a filter, so the dates in the results still need checking).

Image queries use `search --question "<the actual image retrieval question>" --image /absolute/path/query.png`; the script does the base64 encoding. The original image file is limited to 9,000,000 bytes. A file the service cannot read as an image is answered with 422 (send PNG or JPEG). When the image vector cannot be computed, "gaps in this retrieval" lists `visual_query` and the results come from the question text alone. Image-to-image results still need their textual constraints checked; vector similarity must not be assumed to satisfy every combined image-and-text condition.

### Reading back the original

`context --ref` takes the chunk's `doc_id/content_version` and index and reads from `chunk_index - before` to `chunk_index + after` (the service allows at most 60 chunks per call). The returned chunks have the row shape of search hits, without stitching or budget truncation; picture chunks are marked the same way and `image --ref` works on them. When the requested old version can no longer be fetched, search again; never mix old and new text.

### Original picture and crop

`image --ref` fetches the original of a picture chunk: the bitmap embedded in the PDF first, then a high-resolution render of its position on the page, and only then the picture in the parser cache; the output gives the saved path, type, size and source. A crop is a part of the same original and has no more pixels. For a chunk that is not a picture the service answers 404. The `--bbox` of `crop` is a 0–1 fraction or 0–1000 per-mille range over the original picture, not pixels: values above 1000 are rejected, and anything up to 1000 is read as per-mille, so pixel coordinates crop somewhere else without an error. Look at the original picture before choosing the region. A crop is a PNG.

### Graph neighbourhood

A name is matched against titles and aliases, ignoring case and spaces. Of several entities with the same name the one with the most relations is taken and the others are listed under "other entities with this name"; prefer `--ref` whenever a label is available. When nothing matches, a few vector-similar candidates (with IDs) are offered; check them and look one up again with `--entity-id`. A knowledge base without a current graph answers 404.

The response has the top-level keys `kb_id/kb_name/graph_version/found/entity/matches/candidates/direction/types/count/total/has_more/predicates/neighbors`. `entity` carries `id/title/type/parent_type/aliases/scope/degree/pagerank/description/docs/facts`. Each entry of `neighbors[]`:

- `type` (the predicate), `directed` and `direction`: only for a relation with `directed=true` does `out` mean the current entity is the subject and `in` that the far end is; `directed=false` is an undirected association that says nothing about subject, object or causality.
- `weight/npmi/cooccur`, `description` (a model summary, not original text), `type_violation` (the end types do not fit this kind of relation: a check at extraction, not proof that the relation is wrong), `relation_id`.
- `other`: the far-end entity's ID, title, type, scope and its own relation count `degree`.
- `evidence[]`: at most 3 chunks with `point_id/doc_id/chunk_index/content_version/rel_path/place/active`. The first one carries `excerpt` (at most 200 characters, the sentences of the evidence that name the far end) and `excerpt_match`: `both` when both ends are named in the chunk, `other` when only the far end is, `center` / `none` when the far end was not located (the compact view says so). When several relations share an excerpt the compact view prints it under the first and writes "same excerpt as N5" under the others. The excerpt is located by name in the original text, without a model; a text chunk that names the far end is preferred over a picture chunk, and an excerpt taken from a picture chunk has `visual=true` (that text is the vision model's description). Evidence with `active=false` has been deactivated and is not current grounds.

`predicates` is the number of relations of each kind (predicate) the entity has, independent of `limit` and `--type`; `total` is the number of relations under the same filter. There is no paging parameter: narrow with `--type` or raise `--limit`, since repeating the same request does not return a next page. `entity.docs` lists at most 12 document paths and is not a complete list. The graph also filters some relations, so a neighbourhood cannot prove that it holds "everything" in the documents.

### Listing entities and facts

Use these only when the question asks for a complete set. The `entities` response carries `total/offset/limit/count/has_more`, and the first page (`offset=0`) adds `types`: the number of entities per type in this knowledge base (the line at the top of the compact view, with the upper class in brackets). Each entity carries `id/title/type/parent_type/scope/degree/pagerank/aliases`, a `description` cut to 200 characters, `docs` (the 3 documents that mention it most) and `doc_count`, in descending `pagerank` order; entities that only appear on boilerplate pages and reference-number entities are not listed.

`facts`: `--match auto` tries an exact match first and containment when nothing matched. Of same-named subjects the one with the most relations is chosen and the others are listed; when the name does not match, candidates are offered. `facts[]` has the fields of the facts of `search` plus `evidence[]` (at most 3 chunks); `series_text` only covers the facts returned on this page. With a subject only, the first page adds `properties`: the properties under that subject with counts, 50 at most. When "gaps in this retrieval" lists `spec_payload_missing`, the affected rows only have basic fields (the graph is switching versions); query again later.

The text of a fact row is a statement the pipeline composed from extracted fields, not the document's own words; read the evidence back with `context --ref` when the original is needed. Both endpoints list what the graph registered (the result of model extraction), not everything in the documents; a knowledge base without a current graph answers 404.

## Errors

On failure one line of JSON is written to stderr and the exit code is non-zero; an empty result is never returned silently. A mistyped label, a reference of the wrong kind, or a call whose stored response has been removed is reported as an error without sending a request.
