# Retrieval and API

[Home](../README.md) · English | [中文](retrieval.zh-CN.md)

The search service retrieves evidence from indexed documents and graphs.
The calling agent uses that evidence to answer, requests more context when
needed, and cites its sources.

## Connect an agent

The bundled [carrel-search skill](../skills/carrel-search/SKILL.md) includes a
Python standard-library client and instructions for searching, checking
evidence, following relationships, and citing sources. Install the skill using
your agent's supported skill mechanism.

The client defaults to `http://127.0.0.1:9810`. For another host, set
`CARREL_SEARCH_BASE_URL` and `CARREL_SEARCH_TOKEN` in the client environment.
The client also supports `~/.config/carrel-search/config.json` and a separate
token file. See the [client API reference](../skills/carrel-search/references/api.md)
for precedence, timeouts, image inputs, and command syntax.

From the repository root, a local query can be made with:

```bash
python3 skills/carrel-search/scripts/carrel_search.py search \
  --question "What are the delivery and acceptance requirements?"
```

Each command prints a compact view of the response: retrieval status, the text
of the sources with ready-made citations, facts, page summaries, and graph
leads. The complete response is kept in a work directory, and every entry has
a label, so a later command points at it with `--ref <call id>:<label>`
instead of retyping IDs: `context --ref 3fa2c1:S3` reads the original around
a source, `neighbors --ref 3fa2c1:H1` walks the graph from a subject, and
`show 3fa2c1:F2` prints an entry in full. `--json` prints the complete
response instead.

Store tokens in the client environment or a separate token file. The service
uses `KB_SEARCH_TOKEN`; client configuration uses the `CARREL_` variables above.
With no server token, protected endpoints accept loopback callers only.
`/health` is unauthenticated.

The running service rereads the `KB_SEARCH_*` keys of
`config/knowledge-base.env` once a minute, so a new token or parameter takes
effect within a minute without a restart, and a replaced token stops working.
The listen address and port still require a restart, and keys set explicitly
in the service's own environment take precedence over the file.

## HTTP endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service health, authentication mode, and knowledge-base metadata |
| `GET /catalog` | Knowledge-base names, domains, size, file samples, and graph availability; `?refresh=1` refreshes the catalog |
| `POST /search` | Retrieve sources and available entities, relationships, facts, compiled pages, and the one-hop neighbourhood of the subjects |
| `POST /context` | Fetch a document's chunks by index range |
| `GET /image/{kb_id}/{point_id}` | Fetch an image associated with a retrieved chunk |
| `POST /crop` | Crop a selected region of that image |
| `POST /graph/neighbors` | Fetch evidence-backed relationships around an entity, each with an excerpt of the original text |
| `POST /graph/entities` | List entities by type, upper class, or name, with a total and paging |
| `POST /graph/facts` | List the structured facts of a subject, or of a property across subjects, with a total and paging |

The console has a separate `/api` namespace on port 9800 for administration.
Agents normally use the search service on port 9810.

The catalog is cached for `KB_SEARCH_CATALOG_TTL` seconds (60 by default) and
rebuilt at once when a knowledge base is enrolled or closed. `has_graph` is
true only while the knowledge base also has active documents. If the vector
store cannot be read for a knowledge base while the catalog is built (a
collection that does not exist yet counts as empty), its entry reports `chunks`
and `has_graph` as `null` with a `degraded` note, searches still try its graph,
and the catalog is rebuilt after about 15 seconds.

### Search

```bash
curl -sS http://127.0.0.1:9810/search \
  -H "Authorization: Bearer $KB_SEARCH_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"question": "What are the delivery and acceptance requirements?", "top_k": 8}'
```

Set the shell variable to the configured server token.
For a token-free loopback deployment, omit the Authorization header.

`question` is required. Optional `kbs` restricts the knowledge bases; omit it
for automatic routing and use `/catalog` to discover real IDs. `top_k` is
1–50. `context` is a boolean controlling neighboring context;
`explain` requests retrieval diagnostics. Image queries can include `image_b64`
or use the client's `--image` option; an upload the service cannot read as an
image is rejected with HTTP 422 (send PNG or JPEG).

Each request runs within a time budget, `KB_SEARCH_REQUEST_BUDGET` seconds (45
by default, `0` disables it); keep the caller's timeout above it. Every stage
gets the smaller of its own timeout and the remaining budget. When the budget
runs short, the service does not widen to other knowledge bases; once it is
spent, it also skips reranking (keeping fusion order) and neighboring context.
Each of these is recorded in `retrieval_summary.degraded`. If nothing can be
returned because the vector and keyword channels failed on every knowledge
base, or because candidate text could not be fetched, `/search` answers HTTP
503 instead of an empty result.

The `hints` object supports four keys:

| Key | Behavior |
|---|---|
| `doc_ids` | Hard document filter |
| `rel_paths` | Hard filter using paths returned by the API |
| `content_version` | Hard version filter |
| `block_types` | Soft preference, such as favoring tables |

Hard filters intersect. They also constrain attached derived evidence: pages
with out-of-scope sources are omitted, and relationships are not attached under
hard scope filtering. Inspect `hints_used`, `hints_ignored`, and `hints_scope`
to see which hints took effect.

### More text and images

Use `kb_id`, `doc_id`, and `content_version` from a result with `/context`:

```json
{
  "kb_id": "<from the search result>",
  "doc_id": "<from the search result>",
  "content_version": "<from the search result>",
  "chunk_from": 5,
  "chunk_to": 7
}
```

The range uses chunk indexes. Pass the returned content version to read the
same document revision as the original hit.

`/image` uses the mirrored PDF only when its version matches the retrieved
chunk and the required image support is available; otherwise it can use the
parser cache. `X-Image-Source` identifies the selected source. `/crop` accepts
`kb_id`, `point_id`, `bbox`, and optional `pad`. Both endpoints take the UUID
`point_id` returned in `sources`; any other value is rejected with HTTP 422.
`bbox` values are 0–1 fractions or 0–1000 per-mille, not pixels: values above
1000 are rejected and any value up to 1000 is read as per-mille. The crop
actually used is returned in `X-Crop-Box`. Inspect the image before
choosing the region. See the [image reference](../skills/carrel-search/references/api.md).

### Graph relationships

`POST /graph/neighbors` accepts `kb_id`, an `entity` title or `entity_id`, and
optional `types`, `direction`, and `limit`. Prefer a returned entity ID when
names are ambiguous. Multi-hop investigation consists of separate requests,
checking the source evidence at each step.

- Directed relationships use `in` and `out` semantically. For undirected
  relationships, use `direction=both` to retrieve either storage direction.
- The default limit is 20, with a maximum of 100. Results are ranked by weight;
  `count` records the number of relationships in this response, `total` the
  number under the same filter, and `has_more` whether some were left out.
- The endpoint returns a single bounded neighborhood. Narrow with `types`
  or adjust `limit` when more focused results are needed.
- `predicates` lists how many relationships of each kind the entity has,
  independent of `limit` and `types`: it shows which other kinds can be
  requested by predicate. Each far end carries `degree`, its own relationship
  count, and the entity carries `facts`, the number of facts under it.
- Each relationship carries an excerpt of the original text: the first
  evidence chunk has `excerpt` (at most 200 characters, the sentences that
  name the far end) and `excerpt_match` (`both` when both ends are named in
  the chunk, `other` when only the far end is, `center` or `none` when the far
  end was not located). The excerpt is located by name, without a model. A
  text chunk that names the far end is preferred over a picture chunk; an
  excerpt taken from a picture chunk is marked `visual`, because that text is
  the vision model's description.
- Relationship descriptions summarize linked evidence. Their evidence positions
  can be passed to `/context` to retrieve the original passages.

### Subject neighbourhoods in search results

`/search` returns `neighborhoods`: for the entities the question is about, a
preview of where the graph leads, so an agent can decide whether to call
`/graph/neighbors` without first looking each subject up.

- Subjects are first the entities the question names: a title or alias that
  appears whole in the question, ignoring case and spaces. A name of two or
  three characters without letters or digits counts only when the entity is
  also among the graph route's matches. The remaining slots are filled with
  the graph route's best matches. `KB_SEARCH_NEIGHBORHOODS` sets the total
  (4 by default, `0` disables the block).
- Each subject carries `named`, `relations` (its relationship total), `facts`,
  `predicates` (counts per kind), and `neighbors`: its eight strongest
  relationships with predicate, direction, and the far end, one line per far
  end.
- The block carries leads only, without evidence. It is omitted for scoped
  queries (hard `hints` filters), and a failure to build it is recorded in
  `retrieval_summary.degraded` without affecting the search.
- The names of all entities of a knowledge base are kept in the search process
  per graph version and read once after a restart or a new graph version.

### Enumerate entities and facts

Use these two endpoints when a question asks for everything of a kind ("all
products", "every parameter of this device"). Ordinary questions are answered
by `/search` alone.

`POST /graph/entities` accepts `kb_id` and optional `types`, `parent_types`
(the upper classes `entity`, `part`, `property`, `process`, `standard`,
`document`), `name` (text contained in the title or an alias), `limit`
(50 by default, at most 200), and `offset`. Entities are ordered by
importance, and each carries `docs` (the documents that mention it most) and
`doc_count`. The first page also returns `types`: the number of entities per
type in this knowledge base.

`POST /graph/facts` accepts `kb_id`, a `subject` title or `subject_id`, an
optional `property` (property name, symbol, or concept name), `match`
(`auto`, `exact`, or `contains`), `limit`, and `offset`. At least one of
subject and property is required. Rows have the shape of `specs` in
`/search` and carry an `evidence` list whose positions can be passed to
`/context`.

- A matched property brings every spelling of the same normalized concept
  with it. `match=auto` tries an exact match first and containment only when
  nothing matched; `property.matched` reports which one applied.
- With a subject only, the first page also returns `properties`: the
  properties this subject has, with counts.
- `total`, `offset`, and `has_more` describe the whole result; request the
  next page with a larger `offset`.
- Entity names ignore case and spaces. When a name does not match,
  `found=false` and `candidates` lists similar entities to choose from by ID.
- Both endpoints list what the graph registered during extraction. They do not
  prove that the documents contain nothing else.

Responses carry the knowledge base's folder name (`kb_names` in `/search`,
`kb_name` elsewhere). A document is cited as `<folder name>/<rel_path>`,
followed by `place`: the short locator that sources and evidence chunks carry
(page, slide, or sheet rows; the deepest heading for documents without pages).

## Interpret the evidence

| Result field | How to use it |
|---|---|
| `sources` | Original passages, provenance, acceptance flags, and available visual descriptions |
| `entities`, `relationships` | Graph candidates and connections; verify relevant links against sources |
| `neighborhoods` | The strongest relationships of the entities the question names; leads for walking the graph |
| `specs` | Structured facts with units, conditions, time, source references, and conflict indicators; `evidence` lists the chunks a fact rests on, with positions for `/context`, the one whose text holds the value first (`located`) when that can be told |
| `pages` | Compiled subject, timeline, or source views; `compiled=true` distinguishes derived content |
| `doc_aggs` | Documents represented in this result set |
| `retrieval_summary` | Routing, selection, evidence status, and degradation diagnostics |

Check `retrieval_summary.evidence_state`: `accepted`, `diagnostic`, or
`unranked`. With `no_relevant_content=true`, returned candidates are not
sufficient grounds for an answer. An unavailable reranker can produce
`unranked` evidence. Service errors and unavailable channels are reported
separately in the diagnostics.

For facts, pages, entities, and relationships, inspect `verified` and
`sources_active`; entities and relationships are checked through one
representative source point per document. After a document is deleted or
re-parsed, the graph catches up only with its next version. Until then,
`verified=false` marks items whose sources are all inactive: they are not
current grounds, such facts carry no hint, and such pages carry no body text.
Items whose sources could not be checked are left unmarked. Conflict indicators
identify differing records. Compare measurements using their subject, unit,
date, and conditions. Series group the facts included in the current result.

`text_truncated` means more source text may be needed. `stitched.pieces`
identifies the positions used in joined passages. Source text is bounded by
`KB_SEARCH_CONTEXT_TOKENS`; compiled page text has a separate
`KB_SEARCH_PAGE_TEXT_TOKENS` budget. Both budgets apply independently of `top_k`.
Use `/context` to retrieve omitted conditions, table headers, or surrounding
passages needed for an answer.

Unavailable stages are reported through `retrieval_summary.degraded`.
Depending on the failure, retrieval can continue with fusion order, keyword
candidates, or the remaining channels. The response records the available
channels and source positions for citation. Entries include:

| Entry | Meaning |
|---|---|
| `visual_query: …` | An image query's image vector could not be computed; results come from the question text alone |
| `catalog: …` | The catalog could not read the vector store; chunk counts and graph availability are unknown |
| `lexical_profile: …` | Knowledge bases were chosen without lexical evidence |
| `<kb>:graph`, `<kb>:visual`, `rerank: TimeoutError` | The stage failed, timed out, or had no budget left; reranking falls back to fusion order |
| `budget: widen skipped` | Too little budget remained to widen to the other knowledge bases |
| `budget: context skipped`, `context: …` | Neighboring chunks or table heads were not fetched |
| `budget: neighborhoods skipped`, `neighborhoods: …` | The subject neighbourhoods were not built |
| `budget: spec evidence skipped`, `<kb>:spec_evidence: …` | Facts carry no source chunks, or only chunk ids for that knowledge base |
| `<kb>:backfill: …` | Candidate text could not be fetched; candidates without text were dropped |
| `widen: rerank: …` | Reranking after widening failed; the first round's below-the-floor verdict is kept |

## Evaluate retrieval

Keep question sets and outputs under `runtime/eval/`, outside Git. With a
question set you have created:

```bash
cd app
.venv/bin/kb search eval --set ../runtime/eval/my-set.json \
  --out ../runtime/eval/result.json --auto
```

`--auto` evaluates automatic routing instead of using each question's supplied
knowledge-base selection. The evaluator reports chunk and document hit rates,
MRR, expected-answer checks, document coverage, negative-query behavior,
latency, errors, and changes from the previous result. A re-parse changes
chunk point IDs, so chunk-level gold is checked before the run: questions none
of whose gold chunks remain are counted as `stale_gold` and left out of hit@k
and MRR, while document gold and expected answers are still judged. Run
`make-set` again to restore chunk-level scoring.

`kb search make-set` can draft a set using a knowledge base's extraction model.
Review the generated questions and expected evidence, then run the set against
your chosen documents and model configuration.
