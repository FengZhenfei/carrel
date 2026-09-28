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

Store tokens in the client environment or a separate token file. The service
uses `KB_SEARCH_TOKEN`; client configuration uses the `CARREL_` variables above.
With no server token, protected endpoints accept loopback callers only.
`/health` is unauthenticated.

## HTTP endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service health, authentication mode, and knowledge-base metadata |
| `GET /catalog` | Knowledge-base names, domains, size, file samples, and graph availability; `?refresh=1` refreshes the catalog |
| `POST /search` | Retrieve sources and available entities, relationships, facts, and compiled pages |
| `POST /context` | Fetch a document's chunks by index range |
| `GET /image/{kb_id}/{point_id}` | Fetch an image associated with a retrieved chunk |
| `POST /crop` | Crop a selected region of that image |
| `POST /graph/neighbors` | Fetch evidence-backed relationships around an entity |

The console has a separate `/api` namespace on port 9800 for administration.
Agents normally use the search service on port 9810.

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
or use the client's `--image` option.

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
`kb_id`, `point_id`, `bbox`, and optional `pad`. Coordinates are normalized
fractions or per-mille values. Inspect the image before
choosing the region. See the [image reference](../skills/carrel-search/references/api.md).

### Graph relationships

`POST /graph/neighbors` accepts `kb_id`, an `entity` title or `entity_id`, and
optional `types`, `direction`, and `limit`. Prefer a returned entity ID when
names are ambiguous. Multi-hop investigation consists of separate requests,
checking the source evidence at each step.

- Directed relationships use `in` and `out` semantically. For undirected
  relationships, use `direction=both` to retrieve either storage direction.
- The default limit is 20, with a maximum of 100. Results are ranked by weight;
  `count` records the number of relationships in this response.
- The endpoint returns a single bounded neighborhood. Narrow with `types`
  or adjust `limit` when more focused results are needed.
- Relationship descriptions summarize linked evidence. Their evidence positions
  can be passed to `/context` to retrieve the original passages.

## Interpret the evidence

| Result field | How to use it |
|---|---|
| `sources` | Original passages, provenance, acceptance flags, and available visual descriptions |
| `entities`, `relationships` | Graph candidates and connections; verify relevant links against sources |
| `specs` | Structured facts with units, conditions, time, source references, and conflict indicators |
| `pages` | Compiled subject, timeline, or source views; `compiled=true` distinguishes derived content |
| `doc_aggs` | Documents represented in this result set |
| `retrieval_summary` | Routing, selection, evidence status, and degradation diagnostics |

Check `retrieval_summary.evidence_state`: `accepted`, `diagnostic`, or
`unranked`. With `no_relevant_content=true`, returned candidates are not
sufficient grounds for an answer. An unavailable reranker can produce
`unranked` evidence. Service errors and unavailable channels are reported
separately in the diagnostics.

For facts and pages, inspect `verified` and `sources_active`. Conflict indicators
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
channels and source positions for citation.

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
latency, errors, and changes from the previous result.

`kb search make-set` can draft a set using a knowledge base's extraction model.
Review the generated questions and expected evidence, then run the set against
your chosen documents and model configuration.
