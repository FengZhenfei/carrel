"""All prompts used by the graph build, maintained locally.

The record syntax, few-shot examples and gleaning prompts of the extraction and summary steps follow the
heavily field-tested form of GraphRAG 3.1.2 (`graphrag/prompts/index/extract_graph.py`,
`summarize_descriptions.py`, MIT licence). On 2026-09-07 several points were tightened after comparing
GraphRAG / LightRAG / Semantica / OG-RAG (see section 3.6 of the "Knowledge graph optimisation directions
and plan" document on the desktop):
· the type list is given as a "type: definition (parent)" menu; the model may not invent types;
· the Document / Section lines are background only; extracting entities from the heading itself is
  forbidden (LightRAG);
· output is capped per unit kind and the most important relations come first (LightRAG);
· unit classification gains a conclusion kind (the document's own conclusion / summary sections), and the
  boilerplate examples cover the layouts of Chinese reports;
· the few-shot examples can be replaced by examples generated from this KB's corpus (the {examples} slot,
  GraphRAG prompt tune);
· the summary prompt receives descriptions as JSON lines; statements from different sources / times must be
  attributed side by side, never flattened (LightRAG);
· the facts prompt gains four field groups: validity period, original period text, abnormality flag and
  reference range (Semantica / GraphRAG claims).
The schema sampling prompts (domain / language / persona / type list) likewise come from the upstream
prompt_tune, plus the local parent type / predicate / type definition / scenario profile steps.

These texts enter the LLM cache keys and the graph build configuration fingerprint
(build.graph_cache_fingerprint takes this file's hash): changing a single character invalidates "resume
build" once for every KB, so wording changes deserve care.
"""
from __future__ import annotations

RECORD_DELIMITER = "##"
TUPLE_DELIMITER = "<|>"
COMPLETION_DELIMITER = "<|COMPLETE|>"
DEFAULT_PREDICATE = "related_to"

# Generic few-shot examples (the three upstream examples, plus unit records, predicates and the Document line);
# they are replaced once the schema stage has generated examples from this KB's corpus (schema.generate_examples)
DEFAULT_EXAMPLES = """Example 1:
Entity_types: ORGANIZATION: a company, institution or committee (parent: entity); PERSON: a named human being (parent: entity)
Predicates: chairs, member_of, related_to
Document: verdantis-policy-brief.txt
Section: Monetary policy > Meeting schedule
Text:
The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where Central Institution Chair Martin Smith will take questions. Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.
######################
Output:
("unit"<|>body<|>news paragraph about a policy meeting)
##
("entity"<|>Central Institution<|>ORGANIZATION<|>The Central Institution is the Federal Reserve of Verdantis, which is setting interest rates on Monday and Thursday)
##
("entity"<|>Martin Smith<|>PERSON<|>Martin Smith is the chair of the Central Institution)
##
("entity"<|>Market Strategy Committee<|>ORGANIZATION<|>The Central Institution committee makes key decisions about interest rates and the growth of Verdantis's money supply)
##
("relationship"<|>Martin Smith<|>Central Institution<|>chairs<|>Martin Smith is the Chair of the Central Institution and will answer questions at a press conference<|>9)
##
("relationship"<|>Market Strategy Committee<|>Central Institution<|>member_of<|>The Market Strategy Committee is a committee of the Central Institution that sets the benchmark interest rate<|>7)
<|COMPLETE|>

######################
Example 2:
Entity_types: ORGANIZATION: a company, institution or committee (parent: entity)
Predicates: owned_by, listed_on, related_to
Document: markets-ipo-report.txt
Section: Markets > IPO
Text:
TechGlobal's (TG) stock skyrocketed in its opening day on the Global Exchange Thursday. But IPO experts warn that the semiconductor corporation's debut on the public markets isn't indicative of how other newly listed companies may perform.

TechGlobal, a formerly public company, was taken private by Vision Holdings in 2014. The well-established chip designer says it powers 85% of premium smartphones.
######################
Output:
("unit"<|>body<|>news paragraph about an IPO)
##
("entity"<|>TechGlobal<|>ORGANIZATION<|>TechGlobal is a stock now listed on the Global Exchange which powers 85% of premium smartphones)
##
("entity"<|>Vision Holdings<|>ORGANIZATION<|>Vision Holdings is a firm that previously owned TechGlobal)
##
("relationship"<|>TechGlobal<|>Vision Holdings<|>owned_by<|>Vision Holdings formerly owned TechGlobal from 2014 until present<|>5)
<|COMPLETE|>

######################
Example 3:
Entity_types: ORGANIZATION: a company, institution or committee (parent: entity); GEO: a country, city or place (parent: entity); PERSON: a named human being (parent: entity)
Predicates: located_in, held_in, negotiated_with, related_to
Document: world-news-hostages.txt
Section: World > Hostage exchange
Text:
Five Aurelians jailed for 8 years in Firuzabad and widely regarded as hostages are on their way home to Aurelia.

The swap orchestrated by Quintara was finalized when $8bn of Firuzi funds were transferred to financial institutions in Krohaara, the capital of Quintara.

The exchange initiated in Firuzabad's capital, Tiruzia, led to the four men and one woman, who are also Firuzi nationals, boarding a chartered flight to Krohaara.

They were welcomed by senior Aurelian officials and are now on their way to Aurelia's capital, Cashion.

The Aurelians include 39-year-old businessman Samuel Namara, who has been held in Tiruzia's Alhamia Prison, as well as journalist Durke Bataglani, 59, and environmentalist Meggie Tazbah, 53, who also holds Bratinas nationality.
######################
Output:
("unit"<|>body<|>news report about a hostage exchange)
##
("entity"<|>Firuzabad<|>GEO<|>Firuzabad held Aurelians as hostages)
##
("entity"<|>Aurelia<|>GEO<|>Country seeking to release hostages)
##
("entity"<|>Quintara<|>GEO<|>Country that negotiated a swap of money in exchange for hostages)
##
("entity"<|>Tiruzia<|>GEO<|>Capital of Firuzabad where the Aurelians were being held)
##
("entity"<|>Krohaara<|>GEO<|>Capital city in Quintara)
##
("entity"<|>Cashion<|>GEO<|>Capital city in Aurelia)
##
("entity"<|>Samuel Namara<|>PERSON<|>Aurelian who spent time in Tiruzia's Alhamia Prison)
##
("entity"<|>Alhamia Prison<|>GEO<|>Prison in Tiruzia)
##
("entity"<|>Durke Bataglani<|>PERSON<|>Aurelian journalist who was held hostage)
##
("entity"<|>Meggie Tazbah<|>PERSON<|>Bratinas national and environmentalist who was held hostage)
##
("relationship"<|>Firuzabad<|>Aurelia<|>negotiated_with<|>Firuzabad negotiated a hostage exchange with Aurelia<|>2)
##
("relationship"<|>Quintara<|>Aurelia<|>negotiated_with<|>Quintara brokered the hostage exchange between Firuzabad and Aurelia<|>2)
##
("relationship"<|>Quintara<|>Firuzabad<|>negotiated_with<|>Quintara brokered the hostage exchange between Firuzabad and Aurelia<|>2)
##
("relationship"<|>Tiruzia<|>Firuzabad<|>located_in<|>Tiruzia is the capital of Firuzabad<|>8)
##
("relationship"<|>Krohaara<|>Quintara<|>located_in<|>Krohaara is the capital of Quintara<|>8)
##
("relationship"<|>Cashion<|>Aurelia<|>located_in<|>Cashion is the capital of Aurelia<|>8)
##
("relationship"<|>Samuel Namara<|>Alhamia Prison<|>held_in<|>Samuel Namara was a prisoner at Alhamia prison<|>8)
##
("relationship"<|>Samuel Namara<|>Meggie Tazbah<|>related_to<|>Samuel Namara and Meggie Tazbah were exchanged in the same hostage release<|>2)
##
("relationship"<|>Samuel Namara<|>Durke Bataglani<|>related_to<|>Samuel Namara and Durke Bataglani were exchanged in the same hostage release<|>2)
##
("relationship"<|>Meggie Tazbah<|>Durke Bataglani<|>related_to<|>Meggie Tazbah and Durke Bataglani were exchanged in the same hostage release<|>2)
##
("relationship"<|>Samuel Namara<|>Firuzabad<|>held_in<|>Samuel Namara was a hostage in Firuzabad<|>2)
##
("relationship"<|>Meggie Tazbah<|>Firuzabad<|>held_in<|>Meggie Tazbah was a hostage in Firuzabad<|>2)
##
("relationship"<|>Durke Bataglani<|>Firuzabad<|>held_in<|>Durke Bataglani was a hostage in Firuzabad<|>2)
<|COMPLETE|>"""

GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text unit taken from a document, a menu of entity types and a menu of relationship predicates, identify all entities of those types in the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, exactly as it is written in the text (keep original language, case and spelling; do not translate)
- entity_type: One of the listed types. The menu below gives each type as "type: definition (parent: category)". If no listed type fits, use the closest listed type; never invent a new type.
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description>)

An entity must be a THING that can recur across documents. Do NOT emit entities for identifiers or literal values (part numbers used as codes, dates, URLs, file names, page numbers, plain numeric values); mention those inside descriptions instead.

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_predicate: one of the following predicates: [{predicates}]. Use related_to when none of them fits.
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score from 1 to 10 indicating strength of the relationship between source entity and target entity
Format each relationship as ("relationship"<|><source_entity><|><target_entity><|><relationship_predicate><|><relationship_description><|><relationship_strength>)

3. Return output as a single list of all the entities and relationships identified in steps 1 and 2. Use **##** as the list delimiter. Write all descriptions in {language}.

4. Before the entities, output exactly one record that classifies the text unit: ("unit"<|><unit_kind><|><one short reason>), where <unit_kind> is one of:
- body: narrative or specification content, including any table whose cells describe or define something (specification, characteristic, result, comparison or configuration tables)
- conclusion: the document's own summary of its results or findings: abnormal-result lists, key findings, conclusions, verdicts, recommendations, executive summaries
- listing: a bare enumeration with no descriptions at all, such as a code list, a parts or item list, a grid of positions or an index
- boilerplate: table of contents (目录 / CONTENTS with page numbers), reading instructions and disclaimers (阅读说明 / 免责声明), cover pages and greeting letters, revision or change history, legal notice, copyright or trademark statements, sales and contact pages, app or QR-code promotion pages
For boilerplate units, do not turn table-of-contents entries, revision notes or legal text into entities or relationships; extract only what the text genuinely describes.

5. Quantity limits: output at most {max_records} records in total and at most {max_entities} entity records in this response. List the most significant relationships first. Output fewer records when the text contains fewer high-value items; do not pad.

6. The Document and Section lines are background only. Use them to disambiguate references and to ground descriptions; do NOT extract entities or relationships from the document name or the section heading text itself, and do not mention them unless they also appear in the text.

7. When finished, output <|COMPLETE|>

######################
-Examples-
######################
{examples}

######################
-Real Data-
######################
Entity_types: {entity_types}
Predicates: {predicates}
Document: {document}
Section: {section}
Text: {input_text}
######################
Output:"""

CONTINUE_PROMPT = (
    "MANY entities and relationships were missed in the last extraction. Remember to ONLY emit "
    "entities that match any of the previously extracted types. Do NOT re-output entities and relationships "
    "that were correctly and fully extracted; add the missing ones below using the same format:\n"
)
LOOP_PROMPT = (
    "It appears some entities and relationships may have still been missed. Answer Y if there are "
    "still entities or relationships that need to be added, or N if there are none. Please answer "
    "with a single letter Y or N.\n"
)

SUMMARIZE_PROMPT = """
You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or more entities, and a list of descriptions, all related to the same entity or group of entities.
Each description is one JSON object per line with "text" and, when known, "source" (the document it came from) and "when" (the date or version of that document).
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
Rules:
- If descriptions from different sources or different times say different things, keep the differences and attribute them (for example "the 2024 report gives …; the 2025 report gives …") instead of collapsing them into one claim.
- If the descriptions clearly refer to distinct things that merely share a name, describe each one separately within the summary.
- Only when descriptions from the same source contradict each other, reconcile them and note the uncertainty.
Make sure it is written in third person, and include the entity names so we have the full context.
Write the summary in {language}. Limit the final description length to {max_length} words.

#######
-Data-
Entities: {entity_name}
Description List (JSON lines):
{description_list}
#######
Output:
"""

# ── Qualified facts (structured extraction from table / list units, domain-independent) ──
FACTS_PROMPT = """You extract structured facts (values, ratings, attributes, specifications) from a text unit that contains tables or lists.
Document: {document}
Document date or version, if known: {axis}
Section: {section}
Main subjects of this document (the things the values most likely belong to when the table does not name one): {subjects}

Text:
{input_text}

Return JSON only, in this exact shape:
{{"facts": [{{"subject": "...", "property": "...", "symbol": "...", "value": "...", "min": "...", "typ": "...", "max": "...", "unit": "...", "conditions": {{"...": "..."}}, "flag": "...", "ref_min": "...", "ref_max": "...", "valid_from": "...", "valid_until": "...", "period_text": "...", "note": "..."}}]}}

Rules:
- One fact per (subject, property, set of conditions). When columns or rows differ by a condition (grade, range, mode, period, variant, temperature, version, configuration...), emit one fact per condition and spell the condition out in "conditions" as {{"condition name": "condition value"}}.
- "subject" is the thing the value belongs to. Prefer the most specific name over a generic class word (a product name over "device", a person's name over "patient", a company name over "the company"). When the text names only a generic class or nothing at all, use the most likely main subject listed above.
- "property" is what the value describes, in the text's own wording; "symbol" is its short symbol if the text shows one (e.g. tAS, HbA1c, ROE), otherwise "".
- Put a single value in "value"; ranges or min / typ / max columns in "min" / "typ" / "max"; the unit separately in "unit"; keep numbers exactly as written; leave fields you do not have as "".
- "flag": the document's own marker attached to this value (an arrow ↑ / ↓, H / L, 异常, out of range, exceeds the limit, changed since the previous version), copied as written; otherwise "".
- "ref_min" / "ref_max": the reference range or specification limits the document shows for this value (a 参考区间 / reference range / limit column); otherwise "".
- "valid_from" / "valid_until": ISO dates (YYYY-MM-DD, YYYY-MM or YYYY) of the period the value applies to, ONLY when the table, its caption or the text states a date, period, version or revision for it, and "period_text" is that statement copied verbatim. Never guess dates: leave all three "" when the text gives none.
- Do not invent or compute values; skip rows without a value; at most {max_facts} facts.
"""

# ── Schema sampling (domain / language / persona / type list: upstream prompt_tune;
#    parents / predicates / definitions / profile: local) ──

GENERATE_DOMAIN_PROMPT = """
You are an intelligent assistant that helps a human to analyze the information in a text document.
Given a sample text, help the user by assigning a descriptive domain that summarizes what the text is about.
Example domains are: "Social studies", "Algorithmic analysis", "Medical science", among others.

Text: {input_text}
Domain:"""

DETECT_LANGUAGE_PROMPT = """
You are an intelligent assistant that helps a human to analyze the information in a text document.
Given a sample text, help the user by determining what's the primary language of the provided texts.
Examples are: "English", "Spanish", "Japanese", "Portuguese" among others. Reply ONLY with the language name.

Text: {input_text}
Language:"""

GENERATE_PERSONA_PROMPT = """
You are an intelligent assistant that helps a human to analyze the information in a text document.
Given a specific type of task and sample text, help the user by generating a 3 to 4 sentence description of an expert who could help solve the problem.
Use a format similar to the following:
You are an expert {{role}}. You are skilled at {{relevant skills}}. You are adept at helping people with {{specific task}}.

task: {sample_task}
persona description:"""

# The upstream generate_entity_types default task is the single sentence "identify the relations and structure
# within the {domain} domain". The constraints added here come from measurement: kb_004's first type list had
# feature ID, which yielded 7,847 entities, 64% of them occurring once with a median degree of 1 -- an
# identifier type is a unique string in every document and only becomes a node connected to nothing.
ENTITY_TYPE_TASK = """
Identify the relations and structure of the community of interest, specifically within the {domain} domain.

An entity type must name a CATEGORY OF THINGS that recurs across many different documents,
so that separate mentions can be merged into one node and linked to other nodes.

Do NOT propose a type whose instances are identifiers or literal values -- for example
IDs, codes, reference numbers, ticket numbers, version strings, dates, URLs, file names or
paths. Those are ATTRIBUTES of an entity, not entity types: each instance occurs exactly
once, never recurs across documents, and yields an isolated node that carries no relations.

When a candidate type names how something is LABELLED rather than what something IS,
drop it and keep the type for the thing itself.

Always include a type for the SUBJECT a whole document is about (the product a datasheet
describes, the person a medical report is about, the module a source file implements, the
parties of a contract), so that every document can be attached to what it describes.
"""

ENTITY_TYPE_GENERATION_PROMPT = """
The goal is to study the connections and relations between the entity types and their features in order to understand all available information from the text.
The user's task is to {task}.
As part of the analysis, you want to identify the entity types present in the following text.
The entity types must be relevant to the user task.
Avoid general entity types such as "other" or "unknown".
This is VERY IMPORTANT: Do not generate redundant or overlapping entity types. For example, if the text contains "company" and "organization" entity types, you should return only one of them.
Don't worry about quantity, always choose quality over quantity. And make sure EVERYTHING in your answer is relevant to the context of entity extraction.
Return at most 30 entity types, the most important first; fold fine-grained variants into their general type (one "timing parameter", not one type per parameter family).
Return the entity types in JSON format with "entity_types" as the key and the entity types as an array of strings.
=====================================================================
EXAMPLE SECTION: The following section includes example output. These examples **must be excluded from your answer**.

EXAMPLE 1
Task: Determine the connections and organizational hierarchy within the specified community.
Text: Example_Org_A is a company in Sweden. Example_Org_A's director is Example_Individual_B.
JSON RESPONSE:
{{"entity_types": ["organization", "person"] }}
END OF EXAMPLE 1

EXAMPLE 2
Task: Identify the key concepts, principles, and arguments shared among different philosophical schools of thought, and trace the historical or ideological influences they have on each other.
Text: Rationalism, epitomized by thinkers such as René Descartes, holds that reason is the primary source of knowledge. Key concepts within this school include the emphasis on the deductive method of reasoning.
JSON RESPONSE:
{{"entity_types": ["concept", "person", "school of thought"] }}
END OF EXAMPLE 2

EXAMPLE 3
Task: Identify the full range of basic forces, factors, and trends that would indirectly shape an issue.
Text: Industry leaders such as Panasonic are vying for supremacy in the battery production sector. They are investing heavily in research and development and are exploring new technologies to gain a competitive edge.
JSON RESPONSE:
{{"entity_types": ["organization", "technology", "sectors", "investment strategies"] }}
END OF EXAMPLE 3
======================================================================

======================================================================
REAL DATA: The following section is the real data. You should use only this real data to prepare your answer. Generate Entity Types only.
Task: {task}
Text: {input_text}
JSON response format:
{{"entity_types": [<entity_types>] }}
"""

PARENT_TYPES_PROMPT = """
{persona}

The knowledge graph for the "{domain}" domain uses these entity types:
{entity_types}

Assign every entity type to exactly one of these fixed parent categories (use the category names verbatim):
{upper_parents}

Return JSON only, in this exact shape:
{{"parent_types": {{"<entity type>": "<parent category>", ...}} }}
"""

UPPER_MAPPING_PROMPT = """
A knowledge graph uses these entity types (with the parent category their author gave them, if any):
{entity_types}

Map every entity type to exactly one of these fixed parent categories (use the category names verbatim):
{upper_parents}

Return JSON only, in this exact shape:
{{"parent_types": {{"<entity type>": "<parent category>", ...}} }}
"""

PREDICATES_PROMPT = """
{persona}

We are building a knowledge graph for the "{domain}" domain. Entities are grouped under these parent categories:
{parent_types}

Sample text from the corpus:
{input_text}

Propose between {min_predicates} and {max_predicates} relationship predicates that recur throughout this corpus and would let a reader answer questions such as "what does X consist of", "what does X require", "which standard does X conform to", "what does X indicate or cause".

Rules:
- Each predicate is a short lower_snake_case verb phrase (e.g. part_of, requires, depends_on, conforms_to, has_parameter, recommends, calls, signed_by, located_in).
- Each predicate carries a one-sentence definition and the parent categories allowed at its source and target ends (use the parent categories above; use ["*"] when any category is allowed).
- Do not include the generic fallback related_to; it is always available.
- Prefer predicates that are specific to this domain over generic ones.

Return JSON only, in this exact shape:
{{"predicates": [{{"name": "<lower_snake_case>", "description": "<one sentence>", "source_parents": ["<parent>", ...], "target_parents": ["<parent>", ...]}}, ...] }}
"""

# Type definitions: the type menu in the extraction prompt needs a one-sentence definition per type, or the model
# cannot tell neighbouring types apart (pin vs signal)
TYPE_DEFINITIONS_PROMPT = """
{persona}

The knowledge graph for the "{domain}" domain uses these entity types (with their parent category):
{entity_types}

Sample text from the corpus:
{input_text}

For every type write ONE sentence (at most 25 words) defining it as it is used in this corpus, so that an annotator can decide which type a mention belongs to. When two types are easily confused, say what the type is NOT.
Describe each type in general terms; do NOT quote specific names from the sample text as examples (the extractor would otherwise copy those names into documents where they never appear).

Return JSON only, in this exact shape:
{{"definitions": {{"<entity type>": "<one sentence>", ...}} }}
"""

# Scenario profile: subject types, axis, conclusion / boilerplate / listing headings, type words, extension
# predicates. The pipeline has no domain words; all of these are induced from this KB.
PROFILE_PROMPT = """
{persona}

The knowledge graph for the "{domain}" domain uses these entity types (with their parent category):
{entity_types}
and these relationship predicates:
{predicates}

Sample text from the corpus:
{input_text}

Describe the scenario profile of this corpus. Return JSON only, in this exact shape:
{{"subject_types": ["<entity type>", ...],
  "axis": "date" | "version" | "none",
  "conclusion_headings": ["<heading>", ...],
  "boilerplate_headings": ["<heading>", ...],
  "listing_headings": ["<heading>", ...],
  "type_words": ["<word>", ...],
  "extension_predicates": ["<predicate>", ...]}}

Where:
- subject_types: the entity types that name what a WHOLE document is about (the product a datasheet describes, the person a medical report is about, the module a source file implements, the parties of a contract). One to three types from the list above, ordered by importance: the FIRST type is what the document is really about (its measurements, specifications and findings belong to entities of that type); later types are secondary.
- axis: how documents in this corpus are ordered in time: "date" when each document carries a date it applies to (reports, records, minutes), "version" when documents are identified by a version or revision (datasheets, manuals, specifications, code), "none" otherwise.
- conclusion_headings: heading words, in the corpus language, that introduce a document's own summary of results or changes (for example 结论, 综合建议, 异常结果汇总, Summary, Key findings, Conclusions, Revision history). Up to eight.
- boilerplate_headings: headings of pages that carry no domain content in this corpus: legal notices, sales or contact pages, reading instructions, promotional pages, document conventions (for example Legal Information, 报告阅读说明, 免责声明, Acknowledgements, License). Up to eight; may be empty.
- listing_headings: headings of bare enumerations with no descriptions: position maps, code or ordering tables, indexes, reference lists (for example Ordering Information, 引脚分布, Index, 参考文献, 物料清单). Up to six; may be empty.
- type_words: generic class nouns of this corpus that are sometimes appended to a name and sometimes dropped, so that the name with and without the word means the same thing (for example 信号 in 复位信号 / 复位, register in status register / status, 指标 in 血红蛋白指标 / 血红蛋白, 函数 in 排序函数 / 排序). Single words only, up to ten; may be empty.
- extension_predicates: the predicates from the list above that EXPLAIN or CONNECT things (cause, indicate, require, depend on, recommend, conform to), useful for extending an answer beyond what was asked. Up to six; may be empty.
"""

# ── Property concept sameness judging (concepts.py): uncertain pairs among the vector neighbours, asked in one batch ──
CONCEPT_JUDGE_PROMPT = """Below are pairs of measured properties or attributes taken from the same knowledge base, each with its symbol, unit and an example value.
For each pair decide whether the two names denote the SAME property (the same measured quantity or attribute, possibly written differently, abbreviated, or in another language), so that their values could be compared on one timeline.
Different quantities that merely belong to the same topic (total cholesterol vs LDL cholesterol; setup time vs hold time; revenue vs net profit) are NOT the same.
Two names that differ only in a side marker (left vs right), a grade (mild vs severe), a numbering or sign (I vs II, +1/2 vs -1/2), a version suffix, or an identifier that appears on one side only denote DIFFERENT properties, however similar they look.
Answer one line per pair in the form "<index>. yes" or "<index>. no", nothing else.

{pairs}
"""

# ── View-layer narration (compile.py): the subject page's overview, using only what the page contains ──
PAGE_NARRATE_PROMPT = """Below is a compiled reference page about "{title}", assembled from a knowledge base: its facts grouped by property and ordered by date or version, its relationships, and its source documents.
Write a concise narrative overview of "{title}" in {language}, 80 to 200 words, in plain prose (no headings, no bullet lists, no tables).
Use ONLY what the page says. Name the properties that changed over time and how (first value to last value, with dates or versions), point out flagged or out-of-range values and which document they come from, and mention the most informative relationships. Do not invent values, do not add general knowledge, do not repeat the tables.

Page:
{page}

Overview:"""
