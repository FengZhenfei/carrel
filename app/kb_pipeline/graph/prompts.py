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
Entity_types: ORGANIZATION: a company, institution or committee (parent: agent); PERSON: a named human being (parent: agent)
Predicates: chairs, member_of, related_to
Document: verdantis-policy-brief.txt
Document date: 2024-06-10
Section: Monetary policy > Meeting schedule
Text:
The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where Central Institution Chair Martin Smith will take questions. Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.
######################
Output:
("unit"<|>body<|>news paragraph about a policy meeting)
##
("entity"<|>Central Institution<|>ORGANIZATION<|>The Central Institution is the Federal Reserve of Verdantis, which meets on Monday 2024-06-10 and Thursday 2024-06-13 and releases its policy decision on 2024-06-13<|>instance)
##
("entity"<|>Martin Smith<|>PERSON<|>Martin Smith is the chair of the Central Institution<|>instance)
##
("entity"<|>Market Strategy Committee<|>ORGANIZATION<|>The Central Institution committee makes key decisions about interest rates and the growth of Verdantis's money supply<|>instance)
##
("relationship"<|>Martin Smith<|>Central Institution<|>chairs<|>Martin Smith is the Chair of the Central Institution and will answer questions at a press conference on 2024-06-13<|>9<|>)
##
("relationship"<|>Market Strategy Committee<|>Central Institution<|>member_of<|>The Market Strategy Committee is a committee of the Central Institution that sets the benchmark interest rate<|>7<|>)
<|COMPLETE|>

######################
Example 2:
Entity_types: ORGANIZATION: a company, institution or committee (parent: agent); PRODUCT: a product, product line or product category (parent: entity)
Predicates: owned_by, listed_on, supplies, related_to
Document: markets-ipo-report.txt
Document date: (unknown)
Section: Markets > IPO
Text:
TechGlobal's (TG) stock skyrocketed in its opening day on the Global Exchange Thursday. But IPO experts warn that the semiconductor corporation's debut on the public markets isn't indicative of how other newly listed companies may perform.

TechGlobal, a formerly public company, was taken private by Vision Holdings in 2014. The well-established chip designer says it powers 85% of premium smartphones.
######################
Output:
("unit"<|>body<|>news paragraph about an IPO)
##
("entity"<|>TechGlobal<|>ORGANIZATION<|>TechGlobal is a stock now listed on the Global Exchange which powers 85% of premium smartphones<|>instance)
##
("entity"<|>Vision Holdings<|>ORGANIZATION<|>Vision Holdings is a firm that previously owned TechGlobal<|>instance)
##
("entity"<|>premium smartphones<|>PRODUCT<|>Premium smartphones, 85% of which are powered by TechGlobal's chips<|>class)
##
("relationship"<|>TechGlobal<|>Vision Holdings<|>owned_by<|>Vision Holdings took TechGlobal private in 2014 and owned it until its new listing<|>5<|>2014..)
##
("relationship"<|>TechGlobal<|>premium smartphones<|>supplies<|>TechGlobal's chips power 85% of premium smartphones<|>6<|>)
<|COMPLETE|>

######################
Example 3:
Entity_types: ORGANIZATION: a company, institution or committee (parent: agent); GEO: a country, city or place (parent: entity); PERSON: a named human being (parent: agent)
Predicates: located_in, held_in, negotiated_with, related_to
Document: world-news-hostages.txt
Document date: (unknown)
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
("entity"<|>Firuzabad<|>GEO<|>Firuzabad held Aurelians as hostages<|>instance)
##
("entity"<|>Aurelia<|>GEO<|>Country seeking to release hostages<|>instance)
##
("entity"<|>Quintara<|>GEO<|>Country that negotiated a swap of money in exchange for hostages<|>instance)
##
("entity"<|>Tiruzia<|>GEO<|>Capital of Firuzabad where the Aurelians were being held<|>instance)
##
("entity"<|>Krohaara<|>GEO<|>Capital city in Quintara<|>instance)
##
("entity"<|>Cashion<|>GEO<|>Capital city in Aurelia<|>instance)
##
("entity"<|>Samuel Namara<|>PERSON<|>Aurelian who spent time in Tiruzia's Alhamia Prison<|>instance)
##
("entity"<|>Alhamia Prison<|>GEO<|>Prison in Tiruzia<|>instance)
##
("entity"<|>Durke Bataglani<|>PERSON<|>Aurelian journalist who was held hostage<|>instance)
##
("entity"<|>Meggie Tazbah<|>PERSON<|>Bratinas national and environmentalist who was held hostage<|>instance)
##
("relationship"<|>Firuzabad<|>Aurelia<|>negotiated_with<|>Firuzabad negotiated a hostage exchange with Aurelia<|>2<|>)
##
("relationship"<|>Quintara<|>Aurelia<|>negotiated_with<|>Quintara brokered the hostage exchange between Firuzabad and Aurelia<|>2<|>)
##
("relationship"<|>Quintara<|>Firuzabad<|>negotiated_with<|>Quintara brokered the hostage exchange between Firuzabad and Aurelia<|>2<|>)
##
("relationship"<|>Tiruzia<|>Firuzabad<|>located_in<|>Tiruzia is the capital of Firuzabad<|>8<|>)
##
("relationship"<|>Krohaara<|>Quintara<|>located_in<|>Krohaara is the capital of Quintara<|>8<|>)
##
("relationship"<|>Cashion<|>Aurelia<|>located_in<|>Cashion is the capital of Aurelia<|>8<|>)
##
("relationship"<|>Samuel Namara<|>Alhamia Prison<|>held_in<|>Samuel Namara was a prisoner at Alhamia prison<|>8<|>)
##
("relationship"<|>Samuel Namara<|>Meggie Tazbah<|>related_to<|>Samuel Namara and Meggie Tazbah were exchanged in the same hostage release<|>2<|>)
##
("relationship"<|>Samuel Namara<|>Durke Bataglani<|>related_to<|>Samuel Namara and Durke Bataglani were exchanged in the same hostage release<|>2<|>)
##
("relationship"<|>Meggie Tazbah<|>Durke Bataglani<|>related_to<|>Meggie Tazbah and Durke Bataglani were exchanged in the same hostage release<|>2<|>)
##
("relationship"<|>Samuel Namara<|>Firuzabad<|>held_in<|>Samuel Namara was a hostage in Firuzabad<|>2<|>)
##
("relationship"<|>Meggie Tazbah<|>Firuzabad<|>held_in<|>Meggie Tazbah was a hostage in Firuzabad<|>2<|>)
##
("relationship"<|>Durke Bataglani<|>Firuzabad<|>held_in<|>Durke Bataglani was a hostage in Firuzabad<|>2<|>)
<|COMPLETE|>"""

GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text unit taken from a document, a menu of entity types and a menu of relationship predicates, identify all entities of those types in the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, exactly as it is written in the text (keep original language, case and spelling; do not translate). For an entity of a type whose parent is event, add the event's date to the name when the text gives or implies one, written as YYYY-MM-DD or YYYY-MM (for example "Design review 2026-03-14"), so that the same event can be recognised in other documents.
- entity_type: One of the listed types. The menu below gives each type as "type: definition (parent: category)". If no listed type fits, use the closest listed type; never invent a new type.
- entity_description: Comprehensive description of the entity's attributes and activities
- entity_kind: "class" only when the name is a plain common noun for a whole group of things and has no name of its own (customers, sensors, suppliers, electric vehicles, the devices); "instance" for everything that has a name of its own -- a named person, organization, product, model, project, meeting, report or document, and also a named concept such as a particular disease or condition, method, standard, technology, test or measured indicator. When unsure, use "instance".
Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description><|><entity_kind>)

An entity must be a THING that can recur across documents. Do NOT emit entities for identifiers or literal values (part numbers used as codes, dates, URLs, file names, page numbers, plain numeric values); mention those inside descriptions instead.

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_predicate: one of the following predicates: [{predicates}]. Use related_to when none of them fits.
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score from 1 to 10 indicating strength of the relationship between source entity and target entity
- relationship_period: when the text states when the relationship began or ended (joined in 2021, the contract runs until June 2025, was responsible for it from March to May), write start..end with each side as YYYY-MM-DD, YYYY-MM or YYYY, leaving a side empty when it is not stated (2021.. or ..2025-06). Leave the field empty when the text gives no time for the relationship; never guess.
Format each relationship as ("relationship"<|><source_entity><|><target_entity><|><relationship_predicate><|><relationship_description><|><relationship_strength><|><relationship_period>)

3. Dates: the Document date line gives the date of the document when it is known. Turn relative or incomplete dates (next week, end of the month, last Friday, 下周, 月底, a day and month without a year) into absolute dates based on it, in descriptions, event names and periods alike. When the document date is unknown, keep such dates as written.

4. Return output as a single list of all the entities and relationships identified in steps 1 and 2. Use **##** as the list delimiter. Write all descriptions in {language}.

5. Before the entities, output exactly one record that classifies the text unit: ("unit"<|><unit_kind><|><one short reason>), where <unit_kind> is one of:
- body: narrative or specification content, including any table whose cells describe or define something (specification, characteristic, result, comparison or configuration tables)
- conclusion: the document's own summary of its results or findings: abnormal-result lists, key findings, conclusions, verdicts, recommendations, executive summaries
- listing: a bare enumeration with no descriptions at all, such as a code list, a parts or item list, a grid of positions or an index
- boilerplate: table of contents (目录 / CONTENTS with page numbers), reading instructions and disclaimers (阅读说明 / 免责声明), cover pages and greeting letters, revision or change history, legal notice, copyright or trademark statements, sales and contact pages, app or QR-code promotion pages
For boilerplate units, do not turn table-of-contents entries, revision notes or legal text into entities or relationships; extract only what the text genuinely describes.

6. Quantity limits: output at most {max_records} records in total and at most {max_entities} entity records in this response. List the most significant relationships first. Output fewer records when the text contains fewer high-value items; do not pad.

7. The Document and Section lines are background only. Use them to disambiguate references and to ground descriptions; do NOT extract entities or relationships from the document name or the section heading text itself, and do not mention them unless they also appear in the text.

8. When finished, output <|COMPLETE|>

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
Document date: {document_date}
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
Document date, if known: {document_date}
Document version, if known: {document_version}
Section: {section}
Likely subjects (things named in this part of the document first, then the document's main subjects): {subjects}

Text:
{input_text}

Return JSON only, in this exact shape:
{{"facts": [{{"subject": "...", "property": "...", "symbol": "...", "value": "...", "min": "...", "typ": "...", "max": "...", "unit": "...", "conditions": {{"...": "..."}}, "flag": "...", "ref_min": "...", "ref_max": "...", "valid_from": "...", "valid_until": "...", "period_text": "...", "note": "..."}}]}}

Rules:
- One fact per (subject, property, set of conditions). When columns or rows differ by a condition (grade, range, mode, period, variant, temperature, version, configuration...), emit one fact per condition and spell the condition out in "conditions" as {{"condition name": "condition value"}}.
- "subject" is the thing the value belongs to. Prefer the most specific name over a generic class word (a product name over "device", a person's name over "patient", a company name over "the company"). When the text names only a generic class or nothing at all, use the likely subject above that the values belong to.
- Skip advice addressed to the reader (what someone should eat, take, do or avoid) and reference rows that explain every possible grade or case instead of giving this subject's own result (what each risk level means, what to do at each level). A recommended value or setting of a thing (a recommended operating voltage) is a fact.
- "property" is what the value describes, in the text's own wording; "symbol" is its short symbol if the text shows one (e.g. tAS, HbA1c, ROE), otherwise "".
- Put a single value in "value"; ranges or min / typ / max columns in "min" / "typ" / "max"; the unit separately in "unit"; keep numbers exactly as written; leave fields you do not have as "".
- "flag": the document's own marker attached to this value (an arrow ↑ / ↓, H / L, 异常, out of range, exceeds the limit, changed since the previous version), copied as written; otherwise "".
- "ref_min" / "ref_max": the reference range or specification limits the document shows for this value (a 参考区间 / reference range / limit column); otherwise "".
- "valid_from" / "valid_until": ISO dates (YYYY-MM-DD, YYYY-MM or YYYY) of the period the value applies to, ONLY when the table, its caption or the text states a date or period for it, and "period_text" is that statement copied verbatim. Turn relative or year-less dates (next month, 月底, 3月5日) into absolute dates using the document date; when it is unknown, leave them in "period_text" only. Never guess dates: leave all three "" when the text gives none.
- Do not invent or compute values; skip rows without a value; at most {max_facts} facts.
"""

# ── 叙述句里的状态事实(按日期组织的资料:周报、纪要、记录;方案第 7 条)。只对画像轴是 date 的库开,控制成本 ──
NARRATIVE_FACTS_PROMPT = """You extract state facts from narrative text in a dated document (a report, a set of minutes, a record or a log).
Document: {document}
Document date, if known: {document_date}
Section: {section}
Likely subjects (things named in this part of the document first, then the document's main subjects): {subjects}

Text:
{input_text}

A state fact says what something's state, assignment or measured value IS at a time: who is responsible for or assigned to something, its status or progress, a deadline or due date, a decision taken, a quantity, price, amount or measurement stated in the text, a condition or diagnosis found. Opinions, advice and recommendations, plans without a commitment, definitions and aliases, and general background that holds for every case rather than for the things this document is about are not state facts.

Return JSON only, in this exact shape:
{{"facts": [{{"subject": "...", "property": "...", "symbol": "", "value": "...", "min": "", "typ": "", "max": "", "unit": "...", "conditions": {{}}, "flag": "...", "ref_min": "", "ref_max": "", "valid_from": "...", "valid_until": "...", "period_text": "...", "note": "..."}}]}}

Rules:
- "subject" is the specific thing the state belongs to (a project, a task, a person, a product, a customer), by its name in the text; when the text does not name one, use the likely subject above that the state belongs to.
- "property" is what the state is about, in a short phrase in the text's own wording (owner, status, progress, due date, decision, budget, blood pressure ...). Use the same phrase for the same kind of state, so that states at different dates can be compared.
- "value" is the state itself, copied from the text (a person's name, a status word, a percentage, a date, a number); put a separate unit in "unit"; leave fields you do not have as "".
- "valid_from": the date the state applies from when the text states one; otherwise the document date. "valid_until": only when the text states an end. Turn relative or year-less dates (next week, end of the month, 下周, 月底, 3月5日) into absolute dates (YYYY-MM-DD, YYYY-MM or YYYY) using the document date; "period_text" is the time expression copied verbatim.
- "flag": a marker the text attaches to the state (delayed, at risk, abnormal, ↑, ↓) copied as written; otherwise "".
- Do not invent or infer states the text does not state; at most {max_facts} facts. Return {{"facts": []}} when there are none.
"""

# ── 表格画像(graph/tables.py):每张表问一次,之后逐行按规则建图 ──
TABLE_PROFILE_PROMPT = """A spreadsheet table from a knowledge base. Decide whether each data row describes one thing of a single type, and how its columns map onto the knowledge graph.
Document: {document}
Sheet: {sheet}
Entity types (type: definition (parent: category)): {entity_types}
Predicates: {predicates}
Columns: {columns}
Sample rows (column: value pairs):
{rows}

Return JSON only, in this exact shape:
{{"row_type": "<one of the entity types above, or none>", "key_column": "<column>", "name_column": "<column>", "references": [{{"column": "<column>", "target_type": "<entity type>", "predicate": "<predicate>"}}], "date_column": "<column>"}}

Rules:
- "row_type" is "none" unless every data row is one separate thing of the same type (a list of customers, employees, products, requirements, tasks, orders, scenarios). Matrices that compare several things (a feature x product grid), report layouts, calculation sheets, totals, and tables whose rows are mixed kinds of items are "none".
- "key_column": the column holding an identifier that is unique for each row (a code, number or ID); "" when there is none.
- "name_column": the column holding each row's human-readable name; it may be the key column when there is no separate name.
- "references": columns whose cells name OTHER things (a person, a customer, a product, a department, another row's identifier), with the entity type of those things and the predicate that links the row to them (use related_to when no listed predicate fits). Columns of plain values, statuses, dates, amounts and descriptions are not references.
- "date_column": the column holding the date each row applies to (an order date, a report date, a deadline); "" when there is none.
- Use the column names exactly as listed above.
"""

# ── 模式采样(领域 / 语言 / 人设 / 类型表:上游 prompt_tune;父类 / 谓词 / 定义 / 画像:本地)──

GENERATE_DOMAIN_PROMPT = """
You are an intelligent assistant that helps a human to analyze the information in a text document.
Given a sample text, help the user by assigning a descriptive domain that summarizes what the text is about.
Example domains are: "Social studies", "Algorithmic analysis", "Medical science", among others.
Reply ONLY with the domain: a short noun phrase of a few words, with no explanation.

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

When the corpus records things that happen at a particular time (meetings, decisions,
action items, incidents, inspections, examinations, releases), include a type for them,
so that what happened, when and in what order can be linked.
"""

ENTITY_TYPE_GENERATION_PROMPT = """
The goal is to study the connections and relations between the entity types and their features in order to understand all available information from the text.
The user's task is to {task}.
As part of the analysis, you want to identify the entity types present in the following text.
The entity types must be relevant to the user task.
Avoid general entity types such as "other" or "unknown".
Name each entity type with a short noun phrase for ONE kind of thing: do not join two kinds or two names into one type with "/" or "and".
Do not use the top-level classes themselves (agent, entity, part, property, process, event, concept, document) as entity types: every type is filed under one of them later, so name the more specific kind of thing these documents talk about (principle, research paper, meeting, not concept, document, event).
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
{{"entity_types": ["principle", "person", "school of thought"] }}
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

Give every entity type a parent. The parent is normally one of these fixed upper categories (use the category names verbatim):
{upper_parents}

When a type is clearly a narrower kind of another type in the list (every instance of the first is also an instance of the second), give that other type as its parent instead; it then belongs to that type's upper category. Use this only for true sub-kinds, never make a type its own ancestor, and when in doubt use an upper category.

Return JSON only, in this exact shape:
{{"parent_types": {{"<entity type>": "<upper category or another entity type from the list>", ...}} }}
"""

PREDICATES_PROMPT = """
{persona}

We are building a knowledge graph for the "{domain}" domain. Entities are grouped under these parent categories:
{parent_types}

Sample text from the corpus:
{input_text}

Propose between {min_predicates} and {max_predicates} relationship predicates that recur throughout this corpus and would let a reader answer questions such as "what does X consist of", "what does X require", "which standard does X conform to", "what does X indicate or cause", "who is responsible for X".

Rules:
- Each predicate is a short lower_snake_case verb phrase (e.g. part_of, requires, depends_on, conforms_to, has_parameter, recommends, calls, signed_by, located_in).
- Each predicate carries a one-sentence definition and the parent categories allowed at its source and target ends (use the parent categories above; use ["*"] when any category is allowed).
- "single_valued": true only when each source can have at most one target for this predicate at any one time, so that a newer statement replaces an older one (for example a person's reports_to, a project's owned_by, a task's has_status); false for everything that can hold several targets at once or describes general knowledge (where something occurs, what it is related to, what it consists of).
- Do not include the generic fallback related_to; it is always available.
- Relations between events (one event before another, at the same time, causing another, resulting in an outcome) are built in; do not propose them.
- Prefer predicates that are specific to this domain over generic ones.

Return JSON only, in this exact shape:
{{"predicates": [{{"name": "<lower_snake_case>", "description": "<one sentence>", "source_parents": ["<parent>", ...], "target_parents": ["<parent>", ...], "single_valued": false}}, ...] }}
"""

# 类型定义:抽取提示词里的类型菜单要带一句定义,模型才分得清相近的类型(引脚 vs 信号);
# 作用域同时判:同名在不同文档里是不是同一个东西(方案第 12 条,以前按上层类一刀切)
TYPE_DEFINITIONS_PROMPT = """
{persona}

The knowledge graph for the "{domain}" domain uses these entity types (with their parent category):
{entity_types}

Sample text from the corpus:
{input_text}

For every type:
1. Write ONE sentence (at most 25 words) defining it as it is used in this corpus, so that an annotator can decide which type a mention belongs to. When two types are easily confused, say what the type is NOT. Describe each type in general terms; do NOT quote specific names from the sample text as examples (the extractor would otherwise copy those names into documents where they never appear).
2. Decide its scope:
   - "global" when the same name in different documents normally refers to the same thing (a named person, organization, product, place, standard, technology), so that mentions from all documents become one node;
   - "document" when the same name in different documents usually refers to different things, each belonging to whatever that document is about (a parameter or section of the device one datasheet describes, a module of the system one design document describes, a step of one document's procedure, a meeting recorded in one set of minutes).

Return JSON only, in this exact shape:
{{"types": {{"<entity type>": {{"definition": "<one sentence>", "scope": "global" | "document"}}, ...}} }}
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
