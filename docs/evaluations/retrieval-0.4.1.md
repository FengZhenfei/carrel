# Carrel 0.4.1 general retrieval benchmarks and codebase capabilities

English | [中文](retrieval-0.4.1.zh-CN.md)

## General retrieval and document question answering

We compared 0.4.1 with 0.4.0 on the same 275 text queries, 16 page images and 55 document questions. Across 38 cross-document queries, a single response covered **76.75%** of the required evidence groups on average, up from **68.73%** in 0.4.0. The final table in this section gives the separate service-side replay of the 0.3.0 suite.

### How a query works in 0.4.1

- **Text queries.** The 0.4.0 full pipeline ranked graph and visual candidates alongside text. With the optional `KB_SEARCH_TEXT_DISCOVERY=true` profile tested here, a text reranker orders text-vector and keyword hits. The service then follows graph references, verifies source versions and appends up to six passages. Hit and MRR use direct hits. Full-response coverage also counts adjacent and graph-sourced passages.
- **Queries with an image.** Pixels drive visual retrieval and reranking. Accompanying text also contributes candidates to the same top ten. With graph exploration enabled, queries without a document filter also return one-hop leads.
- **Answers with the Skill.** The Skill already supported targeted follow-up. It now lists graph-sourced passages separately with the same usable references as direct hits. When comparing two versions of a clause, the agent can read both from the first response or look up the missing version. Follow-ups inherit the time scope and check source or graph versions. Suggested questions need a supporting lead and an unanswered point.

### Text retrieval and image location

The replay used the original 275 text queries, five knowledge bases and 454 evidence groups. Both versions used Qwen3-Embedding-0.6B and Qwen3-Reranker-4B and returned ten direct hits. An evidence group counts when all passages in one acceptable combination are present. Coverage is averaged by question. Both runs retained one invalid chunk ID in the original annotations.

| Metric | 0.4.0 full retrieval | 0.4.1 text discovery |
|---|---:|---:|
| Evidence Hit@5 | 92.36% | 93.45% |
| Evidence MRR@10 | 0.7979 | 0.8141 |
| Target-document Hit@5 | 98.91% | 98.18% |
| Evidence-group recall in the first 10 hits | 90.01% | 89.84% |
| Evidence-group coverage in returned context | 92.13% | 94.33% |
| Returned-context coverage on cross-document questions (38) | 68.73% | 76.75% |

The number of queries with an annotated passage in the first five hits rose from **254 to 257**. Paired 95% bootstrap intervals were −0.73 to +2.91 percentage points for the Hit@5 difference and −0.0026 to +0.0364 for MRR@10. A separate 0.4.0 variant retaining visual retrieval with text-only reranking achieved an MRR@10 of 0.8182.

Within the same 0.4.1 responses, direct hits and adjacent passages supplied all required evidence for **16 of 38** cross-document questions. Including verified passages from the graph raised that count to **23**. Median text-query latency including graph readback was 2.87 s, with a p95 of 4.09 s.

A separate test used the same 16 page images to locate each source document and page.

| Metric (16 original page images) | 0.4.0 | 0.4.1 |
|---|---:|---:|
| Correct document in the first five hits | 9/16 | 12/16 |
| Correct document and page in the first five hits | 4/16 | 6/16 |
| Document MRR@10 | 0.4792 | 0.7250 |

### How Haiku answers document questions with the Skill

The local Codex SDK ran Claude Haiku 5.5 twice per question in independent sessions. **Single search** could read every page of the first response, including graph-sourced passages. **With follow-up** could also request missing evidence. A complete answer covered every required point correctly with no critical error.

| Complete answers | 0.4.0 single search | 0.4.0 with follow-up | 0.4.1 single search | 0.4.1 with follow-up |
|---|---:|---:|---:|---:|
| All document questions | 44/55 | 48/55 | 46/55 | 49/55 |
| Cross-document | 15/25 | 19/25 | 19/25 | 19/25 |
| Single-document | 29/30 | 29/30 | 27/30 | 30/30 |

In 0.4.1 the Skill issued additional requests for 30/55 questions. There was a net gain of three complete answers on single-document questions. Both arms answered 19/25 cross-document questions completely. Tool output showed all annotated evidence for 44/55 questions in each arm. Answer grading also accepted equivalent source passages. Mean answer time rose from 56 s to 65 s and mean token use from 98.2k to 130.3k, including cached input. The paired 95% interval for the difference in answer completeness was −1.8 to +14.5 percentage points.

All 110 sessions for these 55 document questions completed. Six image lookups found no image at the cited location, and one keyword lookup used a missing reference.

### The 0.3.0 retrieval suite

All 99 original queries ran twice at a cutoff of 12, with the original library scope and with automatic routing. The 78 passage-annotated queries contribute to Hit and MRR. The published 0.3.0 run used a 0.6B text reranker. This replay used 4B and the frozen passage mapping. [Original 0.3.0 evaluation](https://github.com/FengZhenfei/carrel/blob/f466d91eb04e67f90d0aa8b53eda41f0e406b4a1/docs/evaluations/retrieval-evaluations-0.3.0.md)

| Library scope | Version | Evidence Hit@5 | MRR@12 | All required files returned | Negative controls rejected |
|---|---|---:|---:|---:|---:|
| Original scope | 0.3.0 | 69/78 | 0.7327 | 6/10 | 7/8 |
| Original scope | 0.4.1 | 73/78 | 0.8654 | 9/10 | 7/8 |
| Automatic | 0.3.0 | 69/78 | 0.7263 | 6/10 | 7/8 |
| Automatic | 0.4.1 | 73/78 | 0.8654 | 9/10 | 7/8 |

Under the original library scope, all required files appeared for 9 of 10 questions, up from 6. On those same ten questions, the count with every prescribed term in the response fell from 10 to 8.

## Codebase capabilities in 0.4.1

A separate set of 24 questions tested how well 0.4.1 could locate source code and explain its behavior. With the service and model held fixed, Haiku answered **13/24** completely from the first response and **23/24** when it could use the Skill to follow up.

The Skill script library held **3,400 indexed files and 76,431 chunks**. Each of the 24 questions searched the entire library. The reference answers drew on 18 source files from 10 projects in Python, R, Bash and JavaScript. For each question, we marked the relevant lines and the behavior an answer needed to explain before running Haiku through the local Codex SDK in the single-search and follow-up conditions.

**Did the search find the implementation?** The first five hits contained a relevant source file for 18/24 questions and at least one marked line of code for 17/24. With adjacent code and passages returned through the graph, the first response contained all required lines for 13/24 questions. When the query included one known relevant filename, that rose to 18/24. On one question, adding the filename displaced code from a function called in another file.

| Example code question (4 per category) | 0.4.1 complete from first response | 0.4.1 complete with follow-up |
|---|---:|---:|
| Which of two functions with the same name implements the requested behavior? | 2/4 | 4/4 |
| At what step does the script exit on invalid input? | 3/4 | 4/4 |
| Which takes precedence, the command-line argument or the environment variable? | 1/4 | 3/4 |
| Which file receives the arguments passed by the entry script? | 1/4 | 4/4 |
| Which results change when a threshold changes? | 3/4 | 4/4 |
| Does the client retry a failed request automatically? | 3/4 | 4/4 |

**Following a call across files.** One question asked which request field a search command used to set the result count. The first search found a Python script in another project that sent `n`. Haiku then traced the target project's shell entry point to the JavaScript code that sent the request. Its field was `limit`. Reading both files showed which implementation the command actually invoked.

The Skill made additional requests on 16/24 questions. Mean answer time went from 50.9 s to 52.7 s and mean token use from 94.7k to 127.7k, including cached input. One answer still omitted which Python interpreter launched a subprocess, although it correctly described how the API key was selected and passed.

Reading by file and line number reproduced all 141 sampled lines, including indentation. The graph showed direct links for three of four sampled imports between Python files. For two other calls where one script launched another, neither link appeared as a direct graph relation. Verifying those calls required reading the source.

[Full 0.4.0 evaluation](capability-evaluations-0.4.0.md) · [0.4.1 release notes](../releases/0.4.1.md)
