# Carrel 0.3.0: Bringing Source Text, Images, and Related Knowledge Together

[Home](../../README.md) · [Evaluation guide](../retrieval.md#evaluate-retrieval) · English | [中文](retrieval-evaluations-0.3.0.zh-CN.md)

## Key findings

Carrel's distinctive value is in bringing **source passage retrieval, image understanding, and related knowledge** into a single retrieval interface with traceable sources. Agents can follow the evidence to investigate, verify, and compose answers. This evaluation shows two concrete benefits:

- **Image evidence ranks higher.** With the graph enabled, dual reranking increased image-question hit@5 from 50.00% to 87.50% and MRR from 0.5208 to 0.8750.
- **The graph adds evidence across documents.** Among 10 cross-document questions, the number with all required documents represented in the returned evidence increased from 5 to 8 with text reranking, and from 5 to 6 with dual reranking.
- **Direct passage ranking involves trade-offs.** The current full pipeline achieved hit@5 of 88.46% and MRR of 0.7327, compared with 89.74% and 0.7942 for the internal graph-off, text-reranking pipeline. Multimodal and graph benefits have not translated into better overall target rankings.
- **Parallel execution preserved target rankings in this test.** Hit scores and MRR were identical across 24 paired runs, with an observed reduction of approximately 4.85% in median latency. More runs would be needed to establish a stable speedup.

The strengths demonstrated here are **broader ways to acquire evidence and connect information across documents**. Evaluating Carrel means considering where target evidence ranks, whether useful related knowledge is found, and how complete the evidence delivered to the agent is.

## 1. Carrel lets agents continue working from the evidence

Passage-based RAG primarily addresses finding document passages similar to a question. Carrel builds on that path by incorporating visual evidence and ontology relationships: images can serve directly as evidence, entities and facts can connect documents, and agents can retrieve surrounding context, open original images, or investigate specific relationships.

This combination suits complex material. Some information depends on layout and images; some is distributed across documents; and some questions require following related entities. These capabilities appear both in the returned evidence and in the interfaces available for further investigation.

### What the combination of capabilities provides

| Capability | What it adds for agents | Evidence or implementation status |
| --- | --- | --- |
| Separate text and image retrieval paths | Scores image and text relevance separately, then combines the evidence. | With the graph enabled, 8 image questions saw hit@5 rise from 4/8 to 7/8. |
| Ontology relationships and facts | Connects entities, relationships, and historical information across documents, providing leads for further investigation. | Complete source coverage for 10 cross-document questions rose from 5/10 to 8/10 with text reranking. |
| Original-image and context interfaces | Allows agents to retrieve surrounding context, inspect original images, and investigate specific relationships after an initial search. | Implemented interfaces; original images were used to verify image targets and alternative evidence in this evaluation. |
| Types, time, and provenance | Organizes knowledge through type schemas while preserving fact sources, temporal information, and conflict markers. | Implemented in 0.3.0, providing a basis for checking relationships and using historical evidence. |
| Continuously maintained knowledge infrastructure | File updates drive index and graph maintenance; parsing, indexing, and graph construction are managed separately. | Implemented; this retrieval evaluation reused the existing data and graphs. |

## 2. Models and test conditions

The evaluation used two embedding models and two rerankers, all running locally on a single NVIDIA DGX Spark. The text models were Qwen3 0.6B models, and the visual models were Qwen3-VL 2B models. All four retrieval models ran in BF16.

Questions were submitted sequentially, while retrieval channels and reranking branches ran concurrently within each question. Before recording the results, 8 warm-up calls were made. Models remained loaded and existing caches were retained, so the results reflect retrieval performance in a warmed-up deployment.

The table below lists the actual served model names and inference capacity settings. `max-model-len` is the model service's input capacity; `max-num-seqs` is its concurrent sequence setting, which differs from the number of simultaneous user requests in the evaluation.

### Embedding models and rerankers used

| Role | Served model name | Parameters | Precision | max-model-len | max-num-seqs | Batch invariant |
| --- | --- | --- | --- | ---: | ---: | --- |
| Text vector retrieval | qwen3-embedding-0.6b | 0.6B | bfloat16 | 8192 | 32 | Enabled |
| Text reranking | qwen3-reranker-0.6b | 0.6B | bfloat16 | 12288 | 16 | Enabled |
| Visual vector retrieval | qwen3-vl-embedding-2b | 2B | bfloat16 | 8192 | 6 | Enabled |
| Visual reranking | qwen3-vl-reranker-2b | 2B | bfloat16 | 12288 | 6 | Unset |

### Hardware, execution, and timing conditions

| Condition | Setting | Details |
| --- | --- | --- |
| Compute | 1 NVIDIA DGX Spark · NVIDIA GB10 · 20 logical CPUs | All four retrieval-model endpoints used local services on the same machine. |
| System memory | 121.7 GiB visible to the OS | Recorded from the operating system's MemTotal value. |
| Software | Ubuntu 24.04.5 LTS · aarch64 · Python 3.12.3 | NVIDIA driver 580.178.04; vLLM 0.24.0+092c4842.dev. |
| Inference image | nvcr.io/nvidia/vllm:26.07-py3 | Models remained loaded and shared the device with existing services. |
| Request execution | Sequential questions; concurrent paths within each question | An independent test process called the retrieval logic using the actual indexes and model APIs; thread pool size 32. |
| Warm-up and caching | 8 warm-up calls; existing caches retained | Two paths were warmed up per knowledge base; model, index, and system caches were retained. |
| Experiment ordering | Random question order and random strategy order within each question | Random seed 20261026; reduces the effect of a fixed execution order. |
| Timing scope | Wall-clock time around the retrieval logic | Includes retrieval and model calls; excludes final answer generation and client network round trips. |
| Load conditions | Existing service environment | No concurrent stress load was injected; background load on the machine was not recorded as a separate time series. |
| Retrieval budget | top_k=12; 64 text candidates; 8 image candidates | Context budget of 6000 tokens; text/image relevance thresholds of 0.1/0.5. |

## 3. Four configurations reveal capability gains and ranking trade-offs

All four strategies used the same material, question set, models, and retrieval budgets, with randomized execution order within each question. The primary comparison used a fixed knowledge-base scope. Each strategy ran 99 questions, including 78 with passage-level annotations.

“Text reranking” uses the existing text-scoring path. Visual vector retrieval remains enabled when visual reranking is disabled. “Dual reranking” scores text and image relevance separately, then combines their rankings.

### Fixed knowledge-base scope: direct evidence retrieval

| Strategy | hit@3 | hit@5 | hit@5 rate | hit@12 | Document@5 | MRR@12 |
| --- | --- | --- | ---: | --- | --- | ---: |
| Graph off · Text reranking | 66/78 | 70/78 | 89.74% | 72/78 | 78/78 | 0.7942 |
| Graph on · Text reranking | 68/78 | 71/78 | 91.03% | 73/78 | 77/78 | 0.7868 |
| Graph off · Dual reranking | 66/78 | 70/78 | 89.74% | 72/78 | 78/78 | 0.7202 |
| Graph on · Dual reranking | 65/78 | 69/78 | 88.46% | 72/78 | 78/78 | 0.7327 |

Graph-on text reranking had the highest hit@5 at 71/78; graph-off text reranking had the highest MRR at 0.7942. The full pipeline achieved 69/78 passage hits, and all 78/78 questions had at least one target document represented in the top five results.

This shows that most questions already reached the correct document scope. The 100% document hit rate describes document-level localization; the exact-passage hit rate was 88.46%.

### Incremental effects with a fixed knowledge-base scope

| Comparison | hit@5 change (percentage points) | MRR change | Gained hits | Lost hits |
| --- | ---: | ---: | ---: | ---: |
| Dual reranking fixed: enable graph | -1.28 | +0.0125 | 0 | 1 |
| Graph on: enable dual reranking | -2.56 | -0.0540 | 3 | 5 |
| Internal baseline → Current full pipeline | -1.28 | -0.0615 | 4 | 5 |

## 4. The clearest benefit of dual reranking is image evidence

Breaking the results down by target evidence type reveals differences behind the aggregate score: image questions improved, while exact target rankings for text and table questions declined. Images account for only 8 of the 78 ranking questions, so text and tables have greater influence on the overall score.

### Results by evidence type with the graph enabled

| Evidence type | Reranking | Questions | hit@5 | Hit rate | MRR@12 |
| --- | --- | ---: | --- | ---: | ---: |
| Text facts and explanations | Text reranking | 47 | 45/47 | 95.74% | 0.8216 |
| Text facts and explanations | Dual reranking | 47 | 41/47 | 87.23% | 0.7011 |
| Table lookup and numerical values | Text reranking | 23 | 22/23 | 95.65% | 0.8080 |
| Table lookup and numerical values | Dual reranking | 23 | 21/23 | 91.30% | 0.7478 |
| Images, interfaces, and charts | Text reranking | 8 | 4/8 | 50.00% | 0.5208 |
| Images, interfaces, and charts | Dual reranking | 8 | 7/8 | 87.50% | 0.8750 |

Image questions gained 3 top-five hits, all of which ranked first. Text questions lost 4 hits, and table questions lost 1. Across all three categories, the total changed from 71/78 to 69/78.

Direct image scoring improved image evidence retrieval. The combined rankings also show different outcomes for text, tables, and images.

### The 5 strict hit regressions have different causes

All 5 hit@5 regressions were checked individually against the original text or images. In 2 cases, the system returned equivalent evidence; in 1, it returned evidence that covered only part of the requirement. In the remaining 2, the target evidence fell outside the top five but remained within the top twelve.

Annotations that identify a single correct passage count some valid alternatives as failures. Actual top-five ranking regressions also occurred. This report retains the original scores from the frozen question set and uses the source review to explain them. The review was performed by the assistant using the original sources.

### Source-review classification of the 5 strict regressions

| Category | Count | Interpretation |
| --- | ---: | --- |
| Equivalent evidence substituted | 2 | Another passage or modality could answer the question, but the uniquely annotated target passage was not hit. |
| Partial evidence substituted | 1 | The new evidence covered only part of the requirement and could not be counted directly as an equivalent hit. |
| Target evidence fell outside the top five | 2 | The target remained in the top 12, but the top five did not provide a complete substitute. |

## 5. Graph value spans target retrieval, related knowledge, and evidence coverage

hit@5 and MRR describe where the annotated direct evidence ranks. The graph also serves another purpose: retrieving relevant entities, relationships, dependencies, and historical facts to give the agent a fuller knowledge context.

With dual reranking held fixed, enabling the graph changed hit@5 from 70/78 to 69/78 and increased MRR from 0.7202 to 0.7327. Direct target retrieval changed only modestly; cross-document coverage offers another perspective.

### More cross-document evidence: string coverage and source coverage measure different things

Among 10 cross-document questions with explicit source requirements, enabling the graph increased the number containing every target string from 4 to 10. Checking the actual returned passages and valid fact evidence showed that complete coverage of the specified documents increased from 5 to 8 with text reranking, and from 5 to 6 with dual reranking.

The difference comes from what each check examines. String matching includes historical sequence text in the graph; document coverage verifies whether the corresponding sources appear in the returned evidence. **The graph increases the coverage of related information, while source coverage measures how much of that information is represented by traceable evidence.**

### 10 cross-document questions: information presence and source coverage

| Strategy | All target strings | All documents@5 | All documents@12 | All documents in returned evidence |
| --- | --- | --- | --- | --- |
| Graph off · Text reranking | 4/10 | 2/10 | 5/10 | 5/10 |
| Graph on · Text reranking | 10/10 | 2/10 | 6/10 | 8/10 |
| Graph off · Dual reranking | 4/10 | 2/10 | 5/10 | 5/10 |
| Graph on · Dual reranking | 10/10 | 3/10 | 4/10 | 6/10 |

### Related-knowledge retrieval is a distinct evaluation dimension

A useful association needs to be relevant to the question, supported by sources, and able to contribute to the answer. Returning more nodes or relationships shows how far retrieval has expanded; the quality of that related knowledge depends on whether it is accurate and useful.

The cross-document results demonstrate part of this additional value through increased coverage of the required sources. Related-fact quality and final answer quality were not quantified in this experiment.

### Four dimensions of graph value

| Evaluation dimension | Focus | Results measured in this evaluation |
| --- | --- | --- |
| Direct evidence retrieval | Whether target passages are found and ranked highly. | hit@5 and MRR@12 were measured; overall changes were modest. |
| Additional related knowledge | Whether newly retrieved entities, relationships, and facts are accurate and useful. | Cross-document results provide partial evidence; related-fact precision and recall were not measured. |
| Cross-document evidence completeness | Whether all sources needed for a combined answer are present and facts can be traced to them. | Document-set coverage was measured on 10 questions; complete fact-level citations were not measured. |
| Benefit to agent answers | Whether related evidence helps explanation, comparison, synthesis, and further verification. | Answer correctness, completeness, and citation quality were not measured. |

## 6. Automatic knowledge-base selection shows the same direction as the primary comparison

In typical use, the system selects the knowledge base automatically. The same 99 questions were therefore tested with automatic selection, followed by another run of the current full strategy through the native `search eval --auto` command. Quality metrics matched between the two automatic-selection runs.

### Automatic selection: graph comparison with dual reranking

| Strategy | hit@5 | Document@5 | MRR@12 | Median latency (s) | P95 (s) |
| --- | --- | --- | ---: | ---: | ---: |
| Graph off · Dual reranking | 70/78 | 78/78 | 0.7181 | 1.030 | 2.196 |
| Graph on · Dual reranking | 69/78 | 78/78 | 0.7263 | 1.234 | 2.312 |

With automatic selection, enabling the graph added approximately +0.0082 MRR, while hit@5 remained 69/78. Fixed and automatic knowledge-base selection are separate experimental conditions, so the full pipeline's MRR is reported as 0.7327 and 0.7263, respectively.

The native CLI verification averaged 1.601 seconds. The table uses the outer wall-clock measurements from the paired experiment to maintain a consistent timing basis within that batch.

## 7. Dual scoring adds work; parallel execution recovers some of the time

Compared with the internal baseline, median latency in the primary comparison increased from 0.225 seconds to 1.175 seconds for the full pipeline, approximately 5.21 times as long. The pipelines perform different work: the full pipeline includes both the graph and visual reranking, so the latency difference reflects the cost of those additional capabilities.

A separate comparison held the dual-reranking algorithm and graph fixed and changed only serial versus parallel execution. This separates the cost of added visual scoring from the benefit of parallel scheduling.

### Primary comparison latency: 99 questions per strategy

| Strategy | Median latency (s) | P95 (s) | Service-reported mean (s) |
| --- | ---: | ---: | ---: |
| Graph off · Text reranking | 0.225 | 0.570 | 0.308 |
| Graph on · Text reranking | 0.423 | 0.932 | 0.492 |
| Graph off · Dual reranking | 1.025 | 1.923 | 1.078 |
| Graph on · Dual reranking | 1.175 | 2.487 | 1.332 |

### Scheduling-only comparison: identical target quality, variable speedup

Each of 8 image questions was repeated 3 times, producing 24 pairs. Serial and parallel execution both achieved hit@5 of 21/24 and MRR of 0.8750. Target evidence rankings and evaluation scores matched within every pair.

Median latency decreased from 1.108 seconds to 1.054 seconds, a reduction of 4.85%. The average paired saving was approximately 80.47 milliseconds. A question-clustered 95% bootstrap interval ranged from −151.69 to 350.38 milliseconds, indicating substantial variation in the observed speed difference.

The complete hit-list order was identical in 18 of the 24 pairs. The remaining 6 showed changes among non-target candidates. The stability observed here concerns the target evaluation scores.

### Fixed dual-reranking algorithm: serial versus parallel

| Scheduling | Runs | hit@5 | MRR@12 | Median latency (s) |
| --- | ---: | --- | ---: | ---: |
| Serial | 24 | 21/24 | 0.8750 | 1.108 |
| Parallel | 24 | 21/24 | 0.8750 | 1.054 |

## 8. Benefits vary with question types and source distribution

Historical regression questions and newly authored source-based questions showed different patterns. The 24 new questions were balanced across text, tables, and images, with 8 in each category, and improved overall with dual reranking. Historical questions were mainly text and table questions and showed larger target-ranking declines.

The composition of the question set therefore affects the results. As image lookup and cross-document reasoning become more prominent in a workload, these capabilities matter more alongside the aggregate scores.

### Comparison by question-set origin with the graph enabled

| Question set | Questions | Text hit@5 | Dual hit@5 | Text MRR | Dual MRR |
| --- | ---: | --- | --- | ---: | ---: |
| Historical regression questions | 54 | 51/54 | 47/54 | 0.8093 | 0.6998 |
| New source-based questions | 24 | 20/24 | 22/24 | 0.7361 | 0.8069 |

The four anonymized knowledge bases contained 176 documents and approximately 15.6k active chunks in total. Knowledge-base sizes and question counts differed. The table below compares text and dual reranking with the graph enabled.

### Knowledge-base size and ranking performance with the graph enabled

| Knowledge base | Documents | Active chunks | Ranking questions | Text hit@5 | Dual hit@5 | Text MRR | Dual MRR |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| Knowledge base A | 10 | 584 | 19 | 18/19 | 17/19 | 0.7632 | 0.6772 |
| Knowledge base B | 6 | 3013 | 10 | 9/10 | 9/10 | 0.7667 | 0.6778 |
| Knowledge base C | 38 | 1405 | 25 | 24/25 | 21/25 | 0.8613 | 0.7257 |
| Knowledge base D | 122 | 10580 | 24 | 20/24 | 22/24 | 0.7361 | 0.8069 |

## 9. How the capabilities extend passage-based RAG

A workflow centered on passage similarity is effective at quickly locating text. Carrel also gives agents images, structured relationships, temporal information, and source-tracing interfaces, allowing them to continue gathering evidence after the initial search.

The strongest relative benefits observed here are **improved direct image retrieval and additional cross-document evidence from the graph**. For workloads focused on precise text lookup, the internal baseline ranked targets better and ran faster on this question set. For workloads that often involve images, multiple documents, or relationships, Carrel's combination of capabilities is a closer fit.

This experiment compares mechanisms within Carrel. Its relative position against other knowledge-base products can be quantified through comparisons using the same material, models, and questions. The absolute hit@5 of 88.46% should be interpreted alongside task difficulty and the evidence annotation method.

## 10. Test design and question types

There were 99 unique questions and 749 requests in total, including the strategy comparisons, repeated runs, warm-up calls, and native CLI verification.

Of these questions, 54 came from historical development regression tests and 24 were newly authored from the original sources before the experimental results were inspected. There were also 13 cross-document questions and 8 no-evidence tests. All 8 targets for the new image questions were checked against the original images before testing. The question set and annotations were frozen before the comparisons; questions and parameters were not adjusted in response to the results.

Evaluation took place at the retrieval layer. The native scorer computed passage metrics against the frozen annotations, without generated answers or an LLM judge. The assistant prepared the questions and reviewed the sources using the original material. Existing graphs were held fixed across strategies.

### Types of the 99 unique questions

| Question type | Count | Evaluation method |
| --- | ---: | --- |
| Text facts, definitions, and explanations | 47 | Exact-passage hits, MRR, and document hits |
| Table lookup and numerical extraction | 23 | Exact-passage hits, MRR, and document hits |
| Image, interface, and chart lookup | 8 | Exact-passage hits and MRR; original images verified beforehand |
| Cross-document comparison and synthesis | 13 | Content strings and document counts; required-document coverage checked for 10 questions |
| Out-of-corpus information and no-evidence tests | 8 | Negative-test outcome; includes 1 historical question with an identified annotation error |

### Comparison design and key parameters

- The primary comparison used a 2×2 design: graph on/off × text/dual reranking, with randomized strategy order within each question.
- In graph-off runs, graph probing was replaced with an empty result in an independent test process. The production configuration retained its original values.
- Disabling visual reranking retained visual vector retrieval and the other shared retrieval and result-organization strategies.
- The final return limit was `top_k=12`, with at most 64 text reranking candidates, 8 visual reranking candidates, and a context budget of 6,000 tokens.
- Recorded retrieval limits were: vector 50, keyword 30, graph 60, and visual 20. Models and budgets were identical across strategies.
- Relevance thresholds were 0.1 for text reranking and 0.5 for visual reranking. The text reranking window was 480 tokens, with 32 tokens of overlap.

The experiment compared retrieval mechanisms using the current production corpus. Syntax-structure comparison confirmed that the tested deployment and the 0.3.0 release code had matching core retrieval, dual-scoring, grouping, and fusion functions. The difference in the text-scoring call concerned optional public-endpoint model-name and API-key parameters.

### Metric definitions

| Metric | Definition | Interpretation |
| --- | --- | --- |
| hit@5 | Across 78 questions, the fraction with at least one frozen correct passage among the top 5 direct hits. | Scored by exact chunk ID; equivalent evidence is explained in the source review. |
| MRR@12 | Across 78 questions, the mean reciprocal rank of the first correct passage; if none appears in the top 12, the score is 0. | Measures target localization; related knowledge is a separate dimension. |
| doc_hit@5 | The fraction of questions with at least one of the top 5 direct hits belonging to a target document. | Measures document-level localization; coverage of all required documents is calculated separately. |
| expect / expect_all | Whether returned evidence and graph information contain a specified string / all specified strings. | Includes historical sequence text; measures string coverage. |
| Required-document coverage | For 10 questions, whether returned sources and valid fact evidence cover all specified documents. | Measures document-level source completeness; fact-level citations are a separate dimension. |
| Median latency / P95 | Outer wall-clock time for each strategy call; P95 is the sample's 95th percentile. | Per-question retrieval timing in a warmed-up environment; generation and throughput are separate measurements. |

### Interpreting changes in a small sample

Among 78 ranking questions, a change of 1 hit corresponds to approximately 1.28 percentage points. The table below uses 10,000 bootstrap resamples of paired differences for the same questions to show the range of variation within this question set.

All MRR intervals cross 0, so the overall ranking differences retain sampling uncertainty. The 3 additional image hits and increased cross-document source coverage are concrete observations from this experiment. Historical regression questions and questions authored from source material also differ from a random sample of real user questions.

### 95% bootstrap intervals for paired differences

| Comparison | hit@5 difference (percentage points) | 95% interval (percentage points) | MRR difference | 95% interval |
| --- | ---: | --- | ---: | --- |
| Dual reranking fixed: enable graph | -1.28 | [-3.85, 0.00] | +0.0125 | [-0.0150, 0.0427] |
| Graph on: enable dual reranking | -2.56 | [-10.26, 3.85] | -0.0540 | [-0.1148, 0.0100] |
| Internal baseline → Current full pipeline | -1.28 | [-8.97, 6.41] | -0.0615 | [-0.1272, 0.0069] |
| Automatic selection: enable graph | -1.28 | [-3.85, 0.00] | +0.0082 | [-0.0227, 0.0406] |

### Review findings and measurement scope

Single-passage annotations omitted some equivalent evidence, as explained in the regression classifications. The original result for the 8 negative tests was 7/8. The failed case actually had an answer in the corpus, making it an error in a historical question's annotation. This report retains the original score from the frozen question set.

Of the 13 cross-document questions, 10 specified a clear set of required documents and supported strict coverage checks. The older wording of the other 3 did not clearly define the target-document scope. The document-count check passed on 12/13, but meeting a document-count threshold and covering every required document are different measures.

This evaluation measured retrieval results, source coverage, and latency. Final answer correctness, related-knowledge precision and recall, citation completeness, and costs under high concurrency were outside the measurements.

### Execution and consistency checks

| Check | Result | Meaning |
| --- | --- | --- |
| Recorded calls | 650 | 396 primary comparisons + 198 automatic-selection calls + 48 scheduling comparisons + 8 warm-up calls |
| Native CLI verification | 99 | Automatic selection, current full strategy; quality metrics matched the script |
| Total requests | 749 | Includes repeated execution, rather than 749 independent questions |
| Errors / Degradations | 0 / 0 | Checked across 650 records; the CLI separately reported 0 errors |
| Independently recomputed ranking records | 524 | Records with ranking metrics; recomputed values matched |
| Data and service state | Unchanged before and after | Chunk IDs and content-version hashes, graph versions, code commits, and service processes remained consistent |

## References

- [Carrel design and features](https://github.com/FengZhenfei/carrel/blob/v0.3.0/README.md)
- [0.3.0 release notes](https://github.com/FengZhenfei/carrel/blob/v0.3.0/docs/releases/0.3.0.md)
