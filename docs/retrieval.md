# Retrieval and API

[Home](../README.md) · English | [中文](retrieval.zh-CN.md)

The search service retrieves evidence from indexed documents and graphs.
The calling agent uses that evidence to answer, requests more context when
needed, and names the files it relied on.

## Connect an agent

The bundled [carrel-search skill](../skills/carrel-search/SKILL.md) includes a
Python standard-library client and instructions for searching, checking
evidence, answering, listing the reference files, and offering follow-ups along
the graph. Install the skill using your agent's supported skill mechanism.

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
of the sources with the file and place each comes from, facts, page summaries,
graph leads, and the reference files. IDs, scores, and debug fields are left
out; document text is not cut. A view that does not fit one tool output is
paged, and `show <call id> --page 2` prints the next page. The complete
response is kept in a work directory, and every entry has a label, so a later
command points at it with `--ref <call id>:<label>` instead of retyping IDs:
`context --ref 3fa2c1:S3` reads the original around a source,
`context --ref 3fa2c1:S3 --whole` reads its whole document,
`neighbors --ref 3fa2c1:H1` walks the graph from a subject, and
`show 3fa2c1:F2` prints an entry in full. `--json` prints the complete
response instead.

The skill starts with one search. Simple questions use the returned evidence;
comparisons and synthesis check the requested subjects, facts and conditions,
then fill specific gaps through focused searches, source context or graph
lookups. A relevant intermediate entity can lead to a second hop when a
necessary connection is missing. Each step preserves the question's scope,
and retrieval stops when the evidence suffices or no useful lead remains.

Original images can accompany an answer after inspection, with their document
and figure/page in the caption. Low-confidence readings, version differences
and conflicting facts are checked against the source. Related excerpts retain
their original character range and can be read in full through `context`.
Whole-document reads serve explicit, complete item-by-item requests. Answers
include the files used and worthwhile follow-up directions.

How long an answer takes depends mostly on the agent's reasoning effort, not on
the service: retrieval returns within seconds and the rest is the agent
reading and thinking, and the highest levels take several times longer than a
middle-to-high one. Claude Code lets a skill pin its own level with an
`effort:` line in the frontmatter of `SKILL.md` (for example `effort: high`),
which applies to the turn that invokes the skill; in other agents, choose the
level for the session.

Store tokens in the client environment or a separate token file. The service
uses `KB_SEARCH_TOKEN`; client configuration uses the `CARREL_` variables above.
With no server token, protected endpoints accept loopback callers only.
`/health` is unauthenticated.

The running service rereads the `KB_SEARCH_*` keys of
`config/knowledge-base.env` once a minute, so a new token or parameter takes
effect within a minute without a restart, and a replaced token stops working.
The listen address and port still require a restart, and keys set explicitly
in the service's own environment take precedence over the file.

## Time and schema queries

`/search`, `/graph/neighbors`, `/graph/facts`, and `/graph/pages` default to
current evidence. Use `history: true` to include superseded or expired records,
or `as_of: "2025-06-30"` for a date. A year or month means its last day. Results
carry time scope and status so historical statements remain distinguishable.

`POST /graph/schema` takes `kb_id`, an optional `version`, and `format`
(`json` or `yaml`). `POST /graph/stats` takes `kb_id` and returns graph coverage
and type statistics. `/catalog` also summarizes types, predicates, subjects,
and time coverage. The bundled client exposes `schema`, `stats`, `--history`,
and `--as-of`.

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
| `POST /graph/facts` | List the structured facts of a subject, of a property across subjects, or of a document, or only those in cross-document conflict groups, with a total and paging |
| `POST /graph/pages` | List compiled pages (subject, timeline, source, and index pages) by kind, title, entity, or document, with their full text |
| `POST /docs` | List every document of a knowledge base with its type, index state, chunk count, and latest unresolved failure |
| `POST /grep` | Count the chunks and documents in which literal phrases appear, with the first hits and a literal check |

The console has a separate `/api` namespace on port 9800 for administration.
Agents normally use the search service on port 9810.

The catalog is cached for `KB_SEARCH_CATALOG_TTL` seconds (60 by default) and
rebuilt at once when a knowledge base is enrolled or closed. `has_graph` is
true only while the knowledge base also has active documents. If the vector
store cannot be read for a knowledge base while the catalog is built (a
collection that does not exist yet counts as empty), its entry reports `chunks`
and `has_graph` as `null` with a `degraded` note, searches still try its graph,
and the catalog is rebuilt after about 15 seconds.

### Text and image reranking

Configuring `VISUAL_RERANKER_BASE_URL` adds image reranking to the normal search
flow. `KB_SEARCH_RERANK` controls both branches; no separate enable switch is
needed. With no image endpoint or image candidates, the text path remains in use.

- Both branches share 64 candidates by default (`KB_SEARCH_RERANK_N`), reserving
  up to 8 image positions (`KB_SEARCH_VISUAL_RERANK_N`) before truncation.
  Each knowledge-base/subject group receives a minimum share; remaining positions
  go to candidates with higher recall RRF scores.
  Document status and explicit scope are checked before reading cached images.
- Text and image scoring run in parallel. The image model sees cached pixels,
  bounded source text and, for an image query, the query image. Requests use a
  multimodal `/v1/rerank` endpoint, in batches of at most four images. The image
  stage has a 15-second budget within the whole request's deadline.
- Each branch decides relevance independently. Images use their own threshold
  (0.5 by default); thresholds should be checked when changing models. An image
  scored successfully belongs only to the image branch. Each accepted branch
  rank contributes `1 / (KB_SEARCH_RRF_K + rank)`. For text queries, a branch's
  distance above its own relevance threshold reduces the weight of borderline
  matches. Confident matches retain their rank weight. Queries containing an
  image use rank fusion, with images preferred at equal ranks.
  Fused text-query results reserve accepted evidence across selected knowledge
  bases and subject/document groups, then fill by relevance. Other paths retain
  MMR when enabled. Deduplication and context budgets continue to apply.
- Missing images, timeouts or invalid model scores fall back to text and report
  degradation. Low image relevance is a rejection, not a reason to restore a
  high vector score. Block-type preferences apply inside each branch before fusion.
  Text scoring also treats missing, duplicate or invalid document scores as a
  service failure, so an incomplete response cannot masquerade as low relevance.

`retrieval_summary.ranking` reports branch status and counts; its `candidates`
field records the available, admitted and scoped pool sizes. Each source's
`ranking` reports its branch, rank and `margin_weight`. Fusion scores are rank scores, not
confidence probabilities. These retrieval changes reuse existing parsed content,
indexes and graphs. For image retrieval and display, see the image endpoints below.

When a text passage exceeds the context budget, search selects one continuous,
query-relevant excerpt and keeps a preceding sentence when it fits. The source
retains `text_truncated` and an `excerpt` character range within the original
chunk; `/context` retrieves the full passage. Tables and stitched passages keep
their existing truncation and provenance handling.

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

`POST /graph/entities` accepts `kb_id` and optional `types`, `uppers`
(the upper classes `agent`, `entity`, `part`, `property`, `process`, `event`,
`concept`, `document`), `kind` (`all`, `instance`, or `class`), and `name`
(text contained in the title, an alias, or an external identifier), `limit`
(50 by default, at most 200), and `offset`. Entities are ordered by
importance, and each carries `docs` (the documents that mention it most) and
`doc_count`. The first page also returns `types`: the number of entities per
type in this knowledge base.

`POST /graph/facts` accepts `kb_id`, a `subject` title or `subject_id`, an
optional `property` (property name, symbol, or concept name), `match`
(`auto`, `exact`, or `contains`), a document as `doc_id` or `rel_path`,
`conflict_only`, `limit`, and `offset`. At least one of subject, property,
document, and `conflict_only` is required, and the conditions intersect. Rows
have the shape of `specs` in `/search` and carry an `evidence` list whose
positions can be passed to `/context`. When known, each row also carries
`subject_id` (the entity the fact belongs to), `doc_id` (the document it came
from), and `concept_key`; a field without a value is left out (a fact that
hangs under no entity has no `subject_id`, one with no recorded document has no
`doc_id`).

- A matched property brings every spelling of the same normalized concept
  with it. `match=auto` tries an exact match first and containment only when
  nothing matched; `property.matched` reports which one applied.
- With a subject or a document, the first page also returns `properties`: the
  properties in that scope, with counts.
- A document named by `rel_path` is resolved to its `doc_id` through the state
  database; when one path has held several files, the one not deleted wins. An
  unknown path answers HTTP 404, and a `doc_id` and `rel_path` that name
  different documents answer HTTP 422. With a document, the property match
  also runs inside that document.
- `conflict_only=true` keeps only facts in a cross-document conflict group and
  sorts the facts of each group next to each other. Document and conflict
  filters are echoed in `filters`.
- `total`, `offset`, and `has_more` describe the whole result; request the
  next page with a larger `offset`.
- Entity names ignore case and spaces. When a name does not match,
  `found=false` and `candidates` lists similar entities to choose from by ID.
- These endpoints list what the graph registered during extraction. They do not
  prove that the documents contain nothing else.

### Compiled pages

`POST /graph/pages` reads compiled pages whole instead of waiting for a search
to hit them. It accepts `kb_id` and optional `kind` (`subject`, `timeline`,
`source`, or `index`), `title` (text contained in the title, ignoring case and
spaces), `id` (a page ID), `entity_id`, a document as `doc_id` or `rel_path`,
`with_text` (true by default), `limit` (20 by default, at most 100), and
`offset`.

- The response carries `kb_name`, `graph_version`, `kinds`, `pages`, and the
  paging fields `total`, `offset`, `limit`, `count`, and `has_more`, as in
  `/graph/entities`.
- Pages are sorted by kind in the order above, then by title. `kinds` counts
  the pages of each kind in the whole knowledge base, independent of the
  filters.
- Each page carries `n` (its position in the whole result), `id`, `kind`,
  `title`, `summary`, `text` (left out with `with_text=false`; `text_chars`
  always gives its length), `series`, `docs` (paths, `null` where a document
  can no longer be resolved) with the matching `doc_ids`, `entity_ids` (usable
  with `/graph/neighbors` and `/graph/facts`), `concept_keys`, and `path` (the
  page's file path in the compiled view, such as `index.md` or a path under
  `subjects/`).
- Page text is stored cut to 8,000 characters, and a page records at most 32
  documents and 32 entities, so filtering by document or entity can miss pages
  that involve more.
- A knowledge base without a page collection answers HTTP 404; an unreachable
  vector store answers HTTP 503.

### Documents and phrase counts

Use these two endpoints when a question needs the complete list of documents
or how often a phrase occurs, rather than the most relevant passages.

`POST /docs` accepts `kb_id` and optional `dir` (documents under this
directory, subdirectories included), `name` (text contained in `rel_path`,
ignoring case), `include_deleted`, `limit` (500 by default, at most 2000), and
`offset`. Documents are read from the state database and sorted by
`rel_path`. The response carries `kb_name`, `totals`, `docs`, and the paging
fields `total`, `offset`, `limit`, `count`, and `has_more`. Each document
carries `n` (its position in the whole result), `doc_id`, `rel_path`,
`filename`, `dir`, `doc_type`, `mime_type`, `size`, `mtime`,
`content_version`, `status`, `indexed`, `chunk_total`, `parser_profile`, and
`first_seen_at`, plus `diag` (a summary of the chunking check) and
`last_error` (the latest unresolved failure) when present.

- `chunk_total` counts only active chunks of the current content version, the
  ones search can see; `indexed` is true when the parsed version is the
  current one and produced chunks. `totals` summarises every document that
  matches the filters, independent of paging: `files`, `indexed`,
  `not_indexed`, and `chunks`.
- Failure messages are cut to 300 characters, without the stack trace and
  with server paths removed. Responses carry only paths relative to the
  knowledge base.
- `POST /docs` shares its path with the interactive API page (`GET /docs`);
  the two methods do not interfere.

`POST /grep` accepts `kb_id`, `phrases` (1–8, each 1–200 characters after
trimming), optional `rel_paths` and `doc_ids` (up to 500 each) to restrict the
scope, `fields` (any of `body`, `title`, and `visual`; all three by default),
and `limit` (hits returned per phrase, 30 by default, at most 200; `0` returns
counts only). `history` and `as_of` select the same document-status scope as
search; by default, superseded, archived and out-of-validity documents are excluded
before counting. Each hit includes its `content_version` for readback, and `time`
reports the selected scope and excluded-document count.

The bundled client exposes these routes as `docs --kb ID --name "filename"`
and `grep --kb ID --phrase "term"`. D references can constrain `search --in-doc`;
G references open the surrounding original text through `context --ref`.

- The response carries `kb_name`, `fields` (the fields searched), `scope` (how
  many `rel_paths` and `doc_ids` restrict it), `note` (how to read the
  counts), and `results`, one entry per phrase.
- For each phrase the result gives `total_chunks`, `docs` (each matching
  document with its chunk count, most chunks first), and `hits` sorted by path
  and chunk index. A hit carries its position, the `field` it was found in,
  and a short snippet around the phrase rather than the whole chunk.
- Counts are phrase matches of the keyword index's analyzer. CJK text is
  indexed as character bigrams, so a match can be looser than the literal
  phrase. Each returned hit is therefore checked for a literal occurrence
  after Unicode (NFKC) normalisation and whitespace removal: `literal` per hit,
  `literal_checked` and `literal_true` per phrase. Totals are not checked one
  by one.
- Only chunks of current content versions are counted. For a document whose
  keyword-index sync has not caught up yet, counts can differ from the current
  state. `docs_truncated` marks a phrase found in more than 1,000 documents.
- A keyword index that does not exist yet counts as zero hits with a
  `degraded` entry; an unreachable one answers HTTP 503.

Responses carry the knowledge base's folder name (`kb_names` in `/search`,
`kb_name` elsewhere). A document is cited as `<folder name>/<rel_path>`,
followed by `place`: the short locator that sources and evidence chunks carry
(page, slide, sheet rows, notebook cell/output, or source lines; the deepest
heading when available).

For formats with structured locations, `source_locator` records notebook
`cell` and optional `output`, a JSONL or XML `line`, or a text range's `start`
and `end`. These locations accompany search sources and `/grep` hits, alongside
the document and chunk identifiers used by `/context`.

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
date, and conditions. Series include recorded time positions matching the requested time scope,
including positions outside the ranked fact hits.

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

The [0.3.0 retrieval evaluation](evaluations/retrieval-evaluations-0.3.0.md) reports
hit@5, MRR, cross-document evidence coverage, and latency across graph and
reranking configurations, with the models and test conditions used.

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
latency, errors, and changes from the previous result. The previous result is
the file named by `--compare`, or else the existing file at `--out`, which the
run then replaces. A re-parse changes chunk point IDs, so chunk-level gold is
checked before the run: questions none of whose gold chunks remain are counted
as `stale_gold` and left out of hit@k and MRR, while document gold and expected
answers are still judged. Run `make-set` again to restore chunk-level scoring.

`kb search make-set` can draft a set using a knowledge base's extraction model.
Review the generated questions and expected evidence, then run the set against
your chosen documents and model configuration.

### Question sets

A set is a JSON list of questions, or an object whose `items` holds that list.
Only `q` (or `question`) is required; every other field adds a check:

```json
{"items": [
  {"q": "What is the operating temperature range of the ZK200?", "kbs": ["kb_001"],
   "gold_docs": ["datasheets/zk200.pdf"], "expect": ["-40", "85"]},
  {"q": "How did the delivery terms change from the 2024 contract to the 2025 one?",
   "expect": ["2024", "2025"], "expect_all": true, "min_docs": 2},
  {"q": "What is the boiling point of mercury?", "negative": true, "tag": "negative"}
]}
```

| Field | Check |
|---|---|
| `kbs` | Knowledge bases to search, as `kb_NNN`; ignored with `--auto` |
| `gold_point_ids` | Chunks the answer comes from, as `point_id`; ranked for hit@3/5/12 and MRR |
| `gold_docs` | Documents the answer comes from, as `rel_path` or file name; ranked for doc_hit@k, and for hit@k and MRR when there is no chunk gold |
| `expect` | Passes when at least one entry occurs in the returned evidence (case and spaces ignored) |
| `expect_all` | A list whose entries must all occur, or `true` to require every entry of `expect` |
| `min_docs` | The least number of distinct documents among the hits |
| `negative` | `true` for a question the knowledge bases cannot answer; passes when the service reports `no_relevant_content` |
| `tag` | A label printed next to the question |

`make-set` fills `q`, `expect`, `gold_point_ids`, `gold_docs`, `kbs` and `tag`.
For hand-written questions, take `point_id` and `rel_path` from the `sources`
of a search response.
