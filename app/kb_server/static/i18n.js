/* Chinese / English switching for the console. Strings are keyed by the Chinese original: the static page
   is translated once at load (text nodes / placeholder / title), strings assembled in app.js go through
   t(), and stage names / timer jobs / diagnostic reasons / error messages from the server go through t()
   as well -- exact match first, then pattern match (stage strings with numbers), finally split by
   separator and translate segment by segment. Server strings are English originals: in Chinese mode t()
   looks up the same dictionary in reverse (EN_ZH) plus EN_ZH_PATTERNS to restore the Chinese, while the
   UI's own Chinese strings are returned as-is with only the {0} placeholders substituted.
   The language lives in the browser (localStorage kb.lang) and switching reloads the page; data such as
   directory names, file names and entity names is not translated. */
"use strict";

const I18N = (() => {
  let lang = "zh";
  try { lang = localStorage.getItem("kb.lang") === "en" ? "en" : "zh"; } catch (e) { /* private mode: Chinese */ }
  return { lang, locale: lang === "en" ? "en-CA" : "zh-CN" };
})();

const ZH_EN = {
  // ── top bar / sidebar / workspace skeleton ──
  "Carrel 控制台": "Carrel Console",
  "控制台设置了访问令牌,请输入 KB_WEB_TOKEN 的值": "This console requires an access token; enter the value of KB_WEB_TOKEN",
  "模型地址或协议变了,请重新填写 API Key;不需要 Key 的端点请勾选「清空 Key」": "The endpoint or protocol changed; enter the API key again, or tick \"clear key\" for endpoints that need none",
  "界面语言 / Language": "Interface language",
  "中": "中",                                   // the Chinese-side label on the language switch is itself not translated
  "模型列表": "Models",
  "服务状态": "Services",
  "知识库": "Knowledge base",
  "筛选知识库": "Filter knowledge bases",
  "载入中…": "Loading…",
  "还没有知识库:在 mirror 目录下新建文件夹,扫描后会出现在这里": "No knowledge bases yet: create a folder under the mirror directory and it will show up here after the next scan",
  "没有名字含「{0}」的知识库": "No knowledge base whose name contains “{0}”",
  "选择一个知识库": "Select a knowledge base",
  "未选择知识库": "No knowledge base selected",
  "库的编号:Qdrant 集合、日志、缓存目录都用它": "Knowledge base id: used by the Qdrant collection, logs and cache directory",
  "这个目录是已登记的库改了名?": "Is this directory a registered knowledge base that was renamed?",
  "沿用原库,不重新解析": "Adopt the original, no re-parse",
  "配置管理": "Configuration",
  "文件管理": "Files",
  "图谱预览": "Graph",
  "{0} 个文件": "{0} files",
  "等待扫描登记": "Waiting for scan",
  "解析状态": "Parse status",
  "建图状态": "Graph status",
  "重新解析": "Re-parsing",
  "排队中": "Queued",
  "解析 {0}/{1} · {2}%": "Parsing {0}/{1} · {2}%",
  "建图 · {0}%": "Building graph · {0}%",
  "时间进度:{0}%": "elapsed: {0}%",
  "新增占比:{0}%": "new content: {0}%",
  "新增:{0} 条": "new: {0} chunks",
  "已开启": "Enabled",
  "未开启": "Not enabled",
  "已移出": "Removed",
  "(目录消失)": "(directory missing)",
  "{0} 天后彻底删除": "permanently deleted in {0} days",
  "已移出(目录已恢复) · 重新开启即可复原": "Removed (directory is back) · re-enable to restore",
  "目录是符号链接,已按策略拒绝读取": "Directory is a symbolic link; refused by policy (set KB_MIRROR_ALLOW_LINKED_DIRS=1 to read it)",
  "目录消失,等待处理": "Directory missing, awaiting action",
  ";重新打开开关可恢复": "; turn the switch back on to restore",
  "{0} 失败": "{0} failed",
  "文件": "Files",
  "已入库": "Indexed",
  "待解析": "Pending",
  "失败": "Failed",
  "先确认下方策略,点「保存配置」才会真正开启并开始解析;关闭开关可取消": "Review the policy below; “Save” enables the knowledge base and starts parsing. Turn the switch off to cancel",
  "该知识库尚未开启:打开右上角开关,确认策略后开始解析": "This knowledge base is not enabled yet: turn on the switch at the top right, confirm the policy, and parsing starts",
  "把「{0}」认作原来的「{1}」({2})改的名?\n\n沿用它的编号、已入库内容、图谱与缓存;内容没变的文件只刷新路径,不重新解析。": "Treat “{0}” as the renamed “{1}” ({2})?\n\nIts id, indexed content, graph and cache are kept; files whose content is unchanged only get their path refreshed, no re-parse.",
  "沿用原库": "Adopt original",
  "已沿用 {0}:{1}/{2} 个文件对得上": "Adopted {0}: {1}/{2} files matched",
  "(相似度不高,请确认没认错)": " (low similarity, please make sure it is the right one)",
  "沿用失败: ": "Adoption failed: ",
  "当前知识库处于解析任务中,暂不支持操作建图,请稍候再试。": "This knowledge base is being parsed; graph operations are unavailable until it finishes.",
  "知道了": "OK",
  "确认": "Confirm",
  "取消": "Cancel",

  // ── config page ──
  "开启知识库": "Enable knowledge base",
  "多模态结构化描述提示词(留空 = 内置默认)": "Multimodal structured description prompt (empty = built-in default)",
  "内置默认(自定义后替换此段):": "Built-in default (replaced by your own text):",
  "(独立图片文件会额外附上文件名作为主题参考)": "(Standalone image files also get their file name as a topic hint)",
  "保存只更新策略,解析由自动扫描或「立即解析」触发;要应用到已入库内容用「整库重新解析」。": "Saving only updates the policy; parsing is triggered by the automatic scan or “Parse now”. Use “Re-parse all” to apply it to indexed content.",
  "保存配置": "Save",
  "立即解析": "Parse now",
  "整库重新解析": "Re-parse all",
  "删除知识库": "Delete knowledge base",
  "知识图谱": "Knowledge graph",
  "开启知识图谱": "Enable knowledge graph",
  "每单元合并切片数": "Chunks per unit",
  "1(不合并,一片一单元)": "1 (no merging, one chunk per unit)",
  "2 片合一": "2 chunks per unit",
  "3 片合一(默认)": "3 chunks per unit (default)",
  "4 片合一": "4 chunks per unit",
  "5 片合一": "5 chunks per unit",
  "6 片合一": "6 chunks per unit",
  "7 片合一": "7 chunks per unit",
  "8 片合一": "8 chunks per unit",
  "补漏轮数": "Gleaning rounds",
  "0(只抽一遍)": "0 (single pass)",
  "1 轮(默认)": "1 round (default)",
  "2 轮": "2 rounds",
  "实体标签模型": "Label extraction model",
  "采样数量": "Sample size",
  "立即/重新抽取标签": "Extract labels",
  "抽取中…": "Extracting…",
  "「{0}」已抽出 {1} 个实体标签、{2} 个谓词,切回该库查看并保存": "“{0}”: extracted {1} entity labels and {2} predicates; switch back to that knowledge base to review and save",
  "输出语言": "Output language",
  "尚未抽取": "Not extracted yet",
  "标签版本": "Label version",
  "删除选中的这一版标签": "Delete the selected label version",
  "删除": "Delete",
  "实体标签": "Entity labels",
  "类型父类(端点约束用)": "Type parents (for endpoint constraints)",
  "关系谓词": "Relation predicates",
  "实体抽取模型": "Entity extraction model",
  "描述摘要模型": "Description summary model",
  "新增内容自动并入": "Auto-append new content",
  "解析完的新增 / 变更文档每 2 小时自动增量并入现行图:只抽新单元,沿用上一版的归并结论与向量,不等整库重建": "Newly parsed / changed documents are appended to the current graph every 2 hours: only new units are extracted, the previous version’s merge decisions and vectors are reused, no full rebuild needed",
  "自动重建·时间条件": "Auto-rebuild · time condition",
  "留空 = 不按时间": "Empty = ignore time",
  "天": "days",
  "周": "weeks",
  "月": "months",
  "自动重建·新增内容条件": "Auto-rebuild · new content condition",
  "留空 = 不按增量": "Empty = ignore delta",
  "个,新增 Chunk 数": "new chunks (count)",
  "%,新增百分比": "new content (%)",
  "自动重建·条件组合": "Auto-rebuild · combination",
  "满足其一即重建": "Rebuild when either is met",
  "须同时满足": "Both must be met",
  "立即/重新建图": "Build / rebuild now",
  "继续建图": "Resume build",
  "继续并入": "Resume append",
  "只抽新增 / 变更文档的单元,回放上一版的归并结论、沿用没变的向量,作为新版本切换": "Extract only the units of new / changed documents, replay the previous version’s merge decisions, reuse unchanged vectors, and switch to the result as a new version",
  "并入新增内容": "Append new content",
  "暂停建图": "Pause build",
  "删除知识图谱": "Delete knowledge graph",
  "(未选择)": "(not selected)",
  "已完成 {0},续跑跳过": "Done: {0}; skipped when resuming",
  "缓存 {0} 条可续用": "{0} cache entries reusable",
  "{0},缓存不可续用,这次是从头建": "{0}; the cache cannot be reused, this build starts from scratch",
  "配置载入失败: ": "Failed to load config: ",
  "\n\n正在建图,关闭会停止当前任务;已跑完的部分留在缓存里,下次建图从断点继续": "\n\nA build is running; disabling stops it. Finished parts stay in the cache and the next build resumes from there",
  "\n\n已建好的图谱保留,只是不再自动重建": "\n\nThe built graph is kept; it just stops rebuilding automatically",
  "关闭「{0}」的知识图谱?": "Disable the knowledge graph of “{0}”?",
  "确认关闭": "Disable",
  "已关闭知识图谱并停止建图;进度与缓存保留,下次建图从断点继续": "Knowledge graph disabled and the build stopped; progress and cache are kept, the next build resumes from there",
  "已关闭知识图谱;已建好的图谱数据保留,只是不再自动重建": "Knowledge graph disabled; the built graph data is kept, it just stops rebuilding automatically",
  "关闭失败: ": "Disable failed: ",
  "「{0}」已恢复(保留期内,不重新解析)": "“{0}” restored (within the retention window, no re-parse)",
  "「{0}」已开启,开始解析": "“{0}” enabled, parsing started",
  "关闭知识库 {0} 天后会自动删除当前解析结果,确认关闭?": "Disabling deletes the current parse results automatically after {0} days. Disable?",
  "\n\n正在解析的任务会停止,已入库的内容保留;重新开启时恢复数据并接着解析剩下的文件": "\n\nRunning parse jobs stop and indexed content is kept; re-enabling restores the data and continues with the remaining files",
  "已关闭,{0} 个文件进入失活队列": "Disabled; {0} files queued for deactivation",
  "操作失败: ": "Operation failed: ",
  "{0} 需要填一个整数": "{0} must be an integer",
  "自动重建·新增内容条件需为不小于 1 的整数": "The auto-rebuild new-content condition must be an integer of at least 1",
  "自动重建·时间条件需为 1–30 的整数": "The auto-rebuild time condition must be an integer from 1 to 30",
  "「{0}」已开启,按所配策略开始解析": "“{0}” enabled, parsing with the configured policy",
  "开启失败: ": "Enable failed: ",
  "配置已保存,随自动扫描生效": "Config saved; takes effect with the next automatic scan",
  "有未保存的修改": "Unsaved changes",
  "保存失败: ": "Save failed: ",
  "已触发扫描:空库全量解析,已有内容则处理增量文件": "Scan triggered: an empty knowledge base is parsed in full, otherwise only new files are processed",
  "触发失败: ": "Trigger failed: ",
  "图谱配置已保存;建图由自动重建条件或「立即/重新建图」触发": "Graph config saved; builds are triggered by the auto-rebuild conditions or “Build / rebuild now”",
  "图谱配置已保存,建图已启动": "Graph config saved, build started",
  "建图启动失败: ": "Failed to start the build: ",
  "并入已启动:": "Append started: ",
  "并入启动失败: ": "Failed to start the append: ",
  "暂停「{0}」的建图?\n\n进度与缓存保留。暂停期间改模型、标签、谓词、语言或合并切片数会使缓存失效。": "Pause the build of “{0}”?\n\nProgress and cache are kept. Changing the model, labels, predicates, language or chunks per unit while paused invalidates the cache.",
  "已暂停 · 缓存 {0} 条可复用": "Paused · {0} cache entries reusable",
  "暂停失败: ": "Pause failed: ",
  "删除标签版本「{0}」?\n\n删掉就没有了,只能重新抽一次;本库当前生效的标签不受影响。": "Delete label version “{0}”?\n\nIt cannot be recovered, only re-extracted; the labels currently in effect are not affected.",
  "确认删除": "Delete",
  "已删除该标签版本": "Label version deleted",
  "删除失败: ": "Delete failed: ",
  "采样 {0}/{1} 片段(覆盖 {2}/{3} 份文档": "Sampled {0}/{1} chunks (covering {2}/{3} documents",
  ",排除样板 {0} 片": ", {0} boilerplate chunks excluded",
  ",按向量挑代表": ", representatives picked by vector",
  "（触及 token 预算）": " (hit the token budget)",
  " · 领域:{0}": " · domain: {0}",
  "已抽出 {0} 个实体标签、{1} 个谓词,确认后点「保存配置」": "Extracted {0} entity labels and {1} predicates; review them and click “Save”",
  "抽取失败: ": "Extraction failed: ",
  "删除「{0}」的知识图谱?\n\n仅删除图谱数据(图谱向量、图谱库、建图缓存与记录),知识库与解析结果不受影响": "Delete the knowledge graph of “{0}”?\n\nOnly graph data is removed (graph vectors, graph database, build cache and records); the knowledge base and its parse results are not affected",
  "\n\n正在建图:删除会终止任务并清空全部进度,不保留缓存": "\n\nA build is running: deleting terminates it and clears all progress, no cache is kept",
  "删除图谱": "Delete graph",
  "「{0}」的知识图谱已删除": "Knowledge graph of “{0}” deleted",
  "确认对「{0}」重新解析?": "Re-parse “{0}”?",
  "确认重解析": "Re-parse",
  "已重排 {0} 个文件": "{0} files re-queued",
  "失败: ": "Failed: ",
  "彻底删除「{0}」?\n\n此操作立即生效,不可恢复": "Permanently delete “{0}”?\n\nThis takes effect immediately and cannot be undone",
  "\n\n正在运行的解析/建图任务会被终止,已入库内容与缓存一并清除": "\n\nRunning parse / build jobs are terminated; indexed content and caches are removed as well",
  "彻底删除": "Delete permanently",
  "「{0}」已彻底删除": "“{0}” permanently deleted",
  "正在彻底删除「{0}」,数据多的要几分钟": "Deleting “{0}” permanently; a large knowledge base takes a few minutes",
  "「{0}」删除未完成,系统会在下次维护时接着删": "The deletion of “{0}” did not finish; it continues at the next maintenance run",
  "正在彻底删除…": "Deleting permanently…",
  "删除未完成 · 下次维护时接着删": "Deletion unfinished · continues at the next maintenance run",
  "正在彻底删除这个知识库,数据多的要几分钟;可以离开或刷新页面,删完这里会更新": "This knowledge base is being deleted permanently; a large one takes a few minutes. You can leave or reload the page, it updates when the deletion is done",
  "上次彻底删除没有完成:系统会在下次维护时接着删,也可以再点一次「删除知识库」;删完之前不能重新开启": "The last permanent deletion did not finish: it continues at the next maintenance run, or click “Delete knowledge base” again; it cannot be re-enabled until the deletion is done",

  // ── files page ──
  "按文件名 / 路径筛选": "Filter by file name / path",
  "全部状态": "All statuses",
  "待解析 / 排队": "Pending / queued",
  "解析中": "Parsing",
  "有切块提示": "Chunking notes",
  "重新解析中,旧切片仍可用": "Re-parsing; the old chunks stay available",
  "等待重新解析": "Waiting to re-parse",
  "上次到:": "Last reached: ",
  " · {0} 个有切块提示": " · {0} with chunking notes",
  "暂无文件(等待同步或扫描)": "No files yet (waiting for sync or scan)",
  "没有符合筛选条件的文件": "No files match the filter",
  "大小": "Size",
  "切片": "Chunks",
  "切块诊断": "Chunk check",
  "入库时间": "Indexed at",
  "状态": "Status",
  "通过": "Passed",
  "提示": "Note",
  "查看任务时间线": "Show the job timeline",
  " · 均长 {0} token": " · mean {0} tokens",
  "切块预览": "Chunk preview",
  "上一页": "Previous",
  "下一页": "Next",
  "已重新排队": "Re-queued",
  "载入失败: ": "Failed to load: ",

  // ── chunk preview drawer ──
  "读取切片中…": "Reading chunks…",
  "截图核验后已拆开:": "Split after screenshot check: ",
  "截图核验": "screenshot-checked",
  "未核验的粘连值:": "Unverified merged values: ",
  "表格粘连": "Merged table cells",
  " · {0} 片": " · {0} chunks",
  "这份文件入库时还没有切块诊断": "This file was indexed before chunk checks existed",
  "切块验收通过": "Chunk check passed",
  "{0} 层×{1}": "level {0} ×{1}",
  "切片 / {0} 个块": "chunks / {0} blocks",
  "均长 token · σ {0}": "mean tokens · σ {0}",
  "最短 – 最长": "shortest – longest",
  "碎片": "Fragments",
  "超预算": "over budget",
  "标题 · 补认 {0}": "headings · {0} inferred",
  " · 深度 {0}": " · depth {0}",
  "全部": "All",
  "渲染": "Rendered",
  "原文": "Raw text",
  "没有符合筛选的切片": "No chunks match the filter",
  "只显示前 {0} 片": "Only the first {0} chunks are shown",

  // ── merge audit drawer ──
  "实体归并审计": "Entity merge audit",
  "这一版实体归并的账:每一对并掉的实体并进了谁、走的哪一路、依据是什么;以及被规则或第二道判定挡下的对": "The merge ledger of this version: which entity each merged pair went into, by which route and on what grounds, plus the pairs blocked by rules or the second-pass check",
  "读取归并日志中…": "Reading the merge log…",
  "还没有建成的图谱版本": "No graph version has been built yet",
  "版本": "Version",
  "按实体名过滤": "Filter by entity name",
  "实体": "Entity",
  "并入实体": "Merged into",
  "来源": "Route",
  "依据": "Grounds",
  "合并实体": "Merged entities",
  "未合并实体": "Blocked pairs",
  "没有": "None",
  "规则": "rule",
  "字面": "lexical",
  "向量": "vector",
  "回放": "replay",
  "记号相等": "same identifier",
  "去类型 / 后缀词相同": "same after type / suffix words",
  "括号别名": "parenthesized alias",
  "缩写": "abbreviation",
  "拼写": "spelling",
  "翻译": "translation",
  "别名": "alias",
  "未给依据": "no grounds given",
  "上一版": "previous version",
  "极性相反": "opposite polarity",
  "第二道判定": "second-pass check",
  "上位": "broader",
  "下位": "narrower",
  "不同": "different",
  "未答": "no answer",

  // ── job timeline drawer ──
  "任务时间线": "Job timeline",
  "等待重试": "Waiting to retry",
  "已完成": "Done",
  "已取消": "Cancelled",
  " · 已请求取消": " · cancel requested",
  "当前阶段": "Current stage",
  "开始": "Started",
  "结束": "Finished",
  "时长": "Duration",
  "重试次数": "Retries",
  "解析 profile": "Parse profile",
  "文件大小": "File size",
  "任务 id": "Job id",
  "还没有时间线记录": "No timeline records yet",
  "这条任务早于时间线记录,下面是按开始/结束时间合成的概要": "This job predates timeline records; below is a summary synthesized from its start / end times",
  "取消任务": "Cancel job",
  "已请求取消,任务会在下一个阶段边界退出": "Cancel requested; the job exits at the next stage boundary",
  "任务已结束,无需取消": "The job has already finished, nothing to cancel",
  "取消失败: ": "Cancel failed: ",
  "已有任务在飞,不重复排队": "A job is already running, not queued again",

  // ── graph page ──
  "新增 {0} 份 · 变更 {1} 份 · 删除 {2} 份 · 新切片 {3}": "{0} added · {1} changed · {2} removed · {3} new chunks",
  "未开启知识图谱": "Knowledge graph not enabled",
  "在「配置管理」里打开「开启知识图谱」、选好模型后保存,在那里建图": "Turn on “Enable knowledge graph” under Configuration, pick the models and save; builds start from there",
  "构建中": "Building",
  "已运行 {0}": "running for {0}",
  "失败于「{0}」": "Failed at “{0}”",
  "未知步骤": "unknown step",
  "已暂停于「{0}」": "Paused at “{0}”",
  "已停止于「{0}」": "Stopped at “{0}”",
  "LLM 缓存 {0} 条可复用": "{0} LLM cache entries reusable",
  "已存抽取记录 {0} 个单元": "extraction records for {0} units stored",
  "自动重建已跳过": "auto-rebuild skipped",
  "待构建": "Waiting to build",
  "首次开启无条件构建(每 2 小时检查一次,解析忙时让路);之后新增内容自动并入,达到重建条件整库重来": "The first build runs unconditionally (checked every 2 hours, yielding while parsing is busy); afterwards new content is appended automatically and a full rebuild runs once the rebuild conditions are met",
  "关系": "Relations",
  "单元": "Units",
  "更新于": "Updated",
  "{0} 个单元抽取失败": "Extraction failed for {0} units",
  "(涉及 {0} 份文档)": " (across {0} documents)",
  ",已成功的都在图里;下次建图只补这些": "; everything that succeeded is in the graph, the next build only fills these in",
  "续跑时跳过": "skipped when resuming",
  "已建成": "Built",
  "输入实体名定位,回车;留空看连接最多的实体": "Type an entity name and press Enter; leave empty for the most connected entities",
  "画多少个实体": "How many entities to draw",
  "100 个": "100",
  "200 个": "200",
  "300 个": "300",
  "400 个": "400",
  "500 个": "500",
  "1000 个": "1000",
  "1500 个": "1500",
  "2000 个": "2000",
  "3000 个": "3000",
  "分区": "Cluster",
  "按上层本体分区排布:同类聚成一团、类与类分开;再点一次回到纯力导向": "Lay out by upper ontology: each class clusters together, classes apart; click again for plain force-directed layout",
  "上一层": "Previous level",
  "下一层": "Next level",
  "未分类": "unclassified",
  "事物": "Thing",
  "部件": "Part",
  "属性": "Property",
  "过程": "Process",
  "标准": "Standard",
  "文档": "Document",
  "物质": "Substance",
  "开启并建成图谱后,这里画出现行版本的实体与关系": "Once the graph is enabled and built, the entities and relations of the current version are drawn here",
  "这一层没有可画的实体": "Nothing to draw at this level",
  "还没有建成的版本": "No version has been built yet",
  "画的是现行版本;最近一次建图没有完成": "Showing the current version; the latest build did not finish",
  "这份文档": "this document",
  "文档内实体 · 来自 {0}": "Document-scoped entity · from {0}",
  " · 另有 {0} 个同名实体来自其他文档": " · {0} more with the same name from other documents",
  "全局实体 · 出现在 {0} 份文档": "Global entity · appears in {0} documents",
  "{0} 条关系 · 出现 {1} 次": "{0} relations · {1} mentions",
  " · 样板实体": " · boilerplate entity",
  "共 {0} 条,点它看全部": "{0} in total, click it to see all",

  // ── service status drawer ──
  "数据库服务": "Databases",
  "解析服务": "Parser",
  "文本向量服务": "Text embedding",
  "跨模态向量服务": "Visual embedding",
  "多模态服务": "Vision-language model",
  "重排序服务": "Rerankers",
  "上次 {0}": "last {0}",
  "上次失败({0})": "last run failed ({0})",
  "未知": "unknown",
  "定时器未启用": "timer not enabled",
  "系统服务": "System services",
  "全部重启": "Restart all",
  "全部关闭": "Stop all",
  "重启": "Restart",
  "定时任务": "Scheduled tasks",
  "控制台后端不可达": "Console backend unreachable",
  "关闭全部系统服务?\n\n{0}都会停止;解析队列会等服务回来再继续": "Stop all system services?\n\n{0} will stop; the parse queue waits until they are back",
  ",进行中的解析步骤会失败并自动重试": "; running parse steps fail and retry automatically",
  "。重新打开用「全部重启」。": ". Use “Restart all” to bring them back.",
  "重启全部系统服务?\n\n重启期间各服务短暂不可用": "Restart all system services?\n\nEach service is briefly unavailable while restarting",
  ";状态点全部转绿即恢复完成。": "; recovery is complete once every status dot is green.",
  "{0}指令已发出: ": "{0} command sent: ",
  "\n\n仍要强制{0}吗?": "\n\nForce {0} anyway?",
  "强制{0}": "Force {0}",
  "已强制{0}: ": "Forced {0}: ",
  "{0}失败: ": "{0} failed: ",
  "确认重启「{0}」?\n\n重启期间该服务短暂不可用": "Restart “{0}”?\n\nThe service is briefly unavailable while restarting",
  ";状态点转绿即恢复完成。": "; recovery is complete once its status dot is green.",
  "确认重启": "Restart",
  "重启指令已发出: ": "Restart command sent: ",
  "\n\n仍要强制重启「{0}」吗?": "\n\nForce restart “{0}” anyway?",
  "强制重启": "Force restart",
  "已强制重启: ": "Forced restart: ",
  "重启失败: ": "Restart failed: ",
  "扫描镜像": "Mirror scan",
  "解析队列": "Parse queue",
  "图谱维护检查": "Graph maintenance check",
  "失活点 / 解析资产回收": "Inactive point / parse asset GC",
  "缓存周清理": "Weekly cache cleanup",
  "日志月归档": "Monthly log rotation",

  // ── label versions / schema layer ──
  "当前(无版本信息) · {0} 个标签": "Current (no version info) · {0} labels",
  "时间未知": "time unknown",
  "模型未知": "model unknown",
  "采样 {0}": "sample {0}",
  "{0} 个标签": "{0} labels",
  "{0} 个谓词": "{0} predicates",
  "自动·首次建图": "auto (first build)",
  "自动·重建前重抽": "auto (re-extracted before rebuild)",
  "尚未抽取标签": "No labels extracted yet",
  "这一条只是当前生效标签的只读视图,不是保存下来的版本": "This entry is only a read-only view of the labels in effect, not a saved version",
  "正在生效的那一版不能删:图就是按它建的": "The version in effect cannot be deleted: the graph was built with it",
  "尚未抽取,建图将使用全局默认标签": "Not extracted yet; builds use the global default labels",
  "未保存:点「保存配置」后下次建图按这一版": "Unsaved: click “Save” and the next build uses this version",
  "这一版没有父类信息(重抽一次会补上);没有父类就不做端点类型检查": "This version has no parent types (re-extract to add them); without them no endpoint type check is done",
  "这一版没有谓词表:关系一律记为 related_to(重抽一次会补上)": "This version has no predicate table: every relation is recorded as related_to (re-extract to add one)",
  "任意": "any",
  "{0} 条": "{0} edges",
  "建图 {0} 结束时写回的端点账:这个谓词实际用了多少条边;够 20 条且占 5% 以上的端点组合列在这里,下次归并直接放行、重抽标签时规则保住": "Endpoint account written back when build {0} finished: how many edges this predicate actually got; endpoint pairs with at least 20 edges and 5% share are listed here, passed directly at the next merge and preserved by rule when labels are re-extracted",
  "这一版还没建过图,没有端点账": "No graph has been built with this version yet, so there is no endpoint account",
  "谓词": "Predicate",
  "起点父类": "Source parents",
  "终点父类": "Target parents",
  "数据放行": "Confirmed by data",
  "说明": "Description",
  "统计中…": "Counting…",
  "{0} 份文档共 {1} 个采样片段": "{0} documents, {1} chunks available for sampling",
  "该库还没有入库切片": "This knowledge base has no indexed chunks yet",
  "小于 {0}": "less than {0}",
  "、": ", ",

  // ── model registry ──
  "收起": "Close",
  "添加模型": "Add model",
  "＋ 添加模型": "＋ Add model",
  "Key 已配置": "Key set",
  "编辑": "Edit",
  "\n\n以下知识图谱在引用它,删除后对应栏位会清空,需要重新选:\n": "\n\nThese knowledge graphs reference it; after deletion the fields are cleared and must be re-selected:\n",
  "删除模型「{0}」?": "Delete model “{0}”?",
  "已删除;{0} 个知识图谱的对应栏位已清空": "Deleted; the fields of {0} knowledge graphs were cleared",
  "已删除": "Deleted",
  "名称": "Name",
  "(唯一,保存后不可改)": " (unique, cannot be changed after saving)",
  "如 deepseek-v3": "e.g. deepseek-v3",
  "如 deepseek-chat": "e.g. deepseek-chat",
  "接口协议": "Protocol",
  "OpenAI 兼容(/chat/completions)": "OpenAI-compatible (/chat/completions)",
  "Anthropic(/v1/messages)": "Anthropic (/v1/messages)",
  "已配置,留空保持不变": "Set; leave empty to keep it",
  "本地服务可留空": "Optional for local services",
  "保存": "Save",
  "https://api.anthropic.com(不带 /v1,自动补 /v1/messages)": "https://api.anthropic.com (without /v1, /v1/messages is appended)",
  "测试连通性…": "Testing connectivity…",
  "连通性测试通过,已保存": "Connectivity test passed, saved",
  "还没有注册模型。点击下方「添加模型」创建;开启建图的知识库至少需要选择一个。": "No models registered yet. Click “Add model” below to create one; a knowledge base with the graph enabled needs at least one.",

  // ── from the server: parse stages / timeline / graph build stages / timer jobs / errors ──
  "文档解析": "Parsing document",
  "图片描述": "Describing images",
  "切块": "Chunking",
  "文本向量化": "Text embedding",
  "视觉向量化": "Visual embedding",
  "写入向量库": "Writing vectors",
  "关键词索引": "Keyword indexing",
  "完成": "Done",
  "已被同一文件后来的任务取代": "Superseded by a later task for the same file",
  "入队": "Queued",
  "开始处理": "Started",
  "准备语料": "Preparing corpus",
  "实体抽取": "Entity extraction",
  "合并归并": "Merge & resolution",
  "结构化事实": "Structured facts",
  "编译视图页": "Compiling view pages",
  "图谱库导入": "Graph database import",
  "实体归并": "Entity resolution",
  "描述摘要": "Description summaries",
  "计算权重与切片归属": "Computing weights and chunk attribution",
  "属性概念归一": "Normalizing property concepts",
  "切换版本别名": "Switching version aliases",
  "清理旧版本": "Cleaning up old versions",
  "验收通过": "check passed",
  "没有切出任何块": "No chunks were produced",
  "配置读不出来,无法判断缓存能否续用": "Config unreadable, cannot tell whether the cache can be reused",
  "这次建图早于指纹记录": "this build predates the fingerprint record",
  "标签/谓词/模型/单元参数改过": "labels / predicates / model / unit settings changed",
  "语料变过": "corpus changed",
  "建图任务被「暂停建图」停止": "Build stopped by “Pause build”",
  "语料相对现行图谱没有新增、变更或删除的文档,不需要并入": "No documents were added, changed or removed relative to the current graph; nothing to append",
  "还没有建成的图,先整库建图": "No graph has been built yet; run a full build first",
  "模型 / 标签 / 谓词 / 单元参数改过,上一版的抽取与归并不能沿用,请整库重建": "Model / labels / predicates / unit settings changed; the previous extraction and merge cannot be reused, run a full rebuild",
  "建图步骤还没选择模型": "No model selected for the graph build steps",
  "建图已暂停,先「继续建图」": "The build is paused; use “Resume build” first",
  "该知识库还没有活跃切片,先完成解析入库": "This knowledge base has no active chunks yet; finish parsing first",
  "该文件已从来源移除": "This file has been removed from the source",
  "overlap_tokens 必须在 0 与 max_tokens 之间": "overlap_tokens must be between 0 and max_tokens",
  "该文件尚未解析完成,预览需要解析缓存;等它入库后再试": "This file has not finished parsing; the preview needs the parse cache, try again once it is indexed",
  "标签版本不存在或已被淘汰": "The label version does not exist or has been rotated out",
  "正在生效的那一版不能删:图就是按它建的,删了就再也选不回来": "The version in effect cannot be deleted: the graph was built with it and it could never be selected again",
  "建图已经结束,无需暂停": "The build has already finished; there is nothing to pause",
  "当前没有正在进行的建图任务": "No graph build is running",
  "已有建图任务进行中": "A graph build is already running",
  "当前知识库处于解析任务中,暂不支持抽取标签,请稍候再试。": "This knowledge base is being parsed; label extraction is unavailable until it finishes.",
  "该知识库还没有活跃切片,先完成解析入库再抽取标签": "This knowledge base has no active chunks yet; finish parsing before extracting labels",
  "这次标签抽取超过了占用期,已被另一次抽取接管,结果没有发表;请看最新的标签版本": "This label extraction ran past its claim window and another extraction took over; its result was not published. Check the latest label version",
  "已有一次标签抽取在进行中(建图前的自动抽取或另一个窗口),请等它完成后再看标签版本": "A label extraction is already running (the automatic one before the build, or another window); wait for it to finish, then check the label versions",
  "活跃切片取不到正文,没有可抽取的单元": "The active chunks have no text, nothing to extract from",
  "该文件已从来源移除,无法重新解析": "This file has been removed from the source and cannot be re-parsed",
  "连通性测试失败: 响应不是所选协议的有效对话格式": "Connectivity test failed: the response is not a valid chat format for the selected protocol",
  "模型名称只能包含中英文、数字、空格、点、下划线与短横线(不超过 64 字)": "Model names may only contain letters, CJK characters, digits, spaces, dots, underscores and hyphens (64 characters at most)",
  "标签版本由服务端维护,不能直接写入": "Label versions are maintained by the server and cannot be written directly",
  "暂停状态由建图操作维护,不能直接写入": "The paused state is maintained by the build operations and cannot be written directly",
  "跨站写请求已被拒绝(控制台仅接受同源操作)": "Cross-site write request rejected (the console only accepts same-origin operations)",
  "自动重建周期格式无效: 用 7d / 2w / 1m 或纯天数,留空表示不按时间": "Invalid auto-rebuild interval: use 7d / 2w / 1m or a plain number of days; empty means no time condition",
  "自动重建新增切片数无效: 需为不小于 1 的整数,留空表示不按增量": "Invalid auto-rebuild new chunk count: it must be an integer of at least 1; empty means no delta condition",
  "自动重建新增内容条件只能二选一: 百分比或切片数": "The auto-rebuild new-content condition must be either a percentage or a chunk count, not both",
  // ── server strings (the app emits this English directly; the Chinese UI looks it up in reverse via EN_ZH) ──
  "缺少目录名": "Missing directory name",
  "自动重建新增比例格式无效: 如 20% 或 0.2,留空表示不按增量": "Invalid auto-rebuild new-content percentage: use 20% or 0.2; empty means no delta condition",
  "自动重建条件组合方式只能是 or 或 and": "The auto-rebuild condition operator must be or / and",
  "缺少标签版本 id": "Missing label version id",
  "缺少 graph_version": "graph_version is required",
  "建图任务进行中,暂不能修改步骤模型,请等它结束或先关闭知识图谱": "A graph build is running; step models cannot be changed now. Wait for it to finish or turn the knowledge graph off first",
  "建图任务被「关闭知识图谱」停止": "Build stopped by “Turn off knowledge graph”",
  "知识库未开启": "The knowledge base is not enabled",
  "建图进程无法终止,请稍后重试": "The graph build process could not be terminated; try again later",
  "知识库未开启,无法建图": "The knowledge base is not enabled, so no graph can be built",
  "该知识库的目录当前不存在,无法建图": "The knowledge base directory does not exist right now, so no graph can be built",
  "尚未开启知识图谱": "The knowledge graph is not turned on",
  "另一个知识库正在建图(建图一次只能跑一个),等它结束后再试": "Another knowledge base is building its graph (only one build runs at a time); try again when it finishes",
  "知识库未开启,无法重新解析": "The knowledge base is not enabled, so it cannot be re-parsed",
  "控制台取消": "Cancelled from the console",
  "name / base_url / model_id 均为必填": "name, base_url and model_id are all required",
  "protocol 只能是 'openai' 或 'anthropic'": "protocol must be 'openai' or 'anthropic'",
  "dir 必填": "dir is required",
  "file_id 必填": "file_id is required",
  "状态库繁忙,请稍后重试": "The state database is busy; try again later",
  "建图任务被「删除知识图谱」终止": "Build terminated by “Delete knowledge graph”",
  "建图进程无法终止,已放弃删除以免删到一半;请稍后重试": "The graph build process could not be terminated; deletion abandoned rather than left half done. Try again later",
  "仍有建图进程持有建图锁,已放弃删除以免删到一半;请稍后重试": "A graph build still holds the build lock; deletion abandoned rather than left half done. Try again later",
  "建图任务被「删除知识库」终止": "Build terminated by “Delete knowledge base”",
  "另一个知识库正在建图,等它结束后再删除": "Another knowledge base is building its graph; delete after it finishes",
  "另一个知识库正在建图或删除,等它结束后再删除": "Another knowledge base is building its graph or being deleted; delete after it finishes",
  "这个库正在抽取标签(建图前的准备),这一步停不下来;等它抽完(通常几分钟)再删除": "This knowledge base is extracting labels (the preparation before a build), which cannot be stopped; delete after it finishes, usually a few minutes",
  "另一个知识库正在彻底删除,等它删完再建图": "Another knowledge base is being permanently deleted; build the graph when that finishes",
  "这个知识库正在删除中,等它删完": "This knowledge base is already being deleted; wait for it to finish",
  "这个知识库还没有任何活跃切片(没有文件解析入库,或全部已失活/删除),建图没有语料可用。先完成解析再建图。": "This knowledge base has no active chunks (nothing parsed and stored, or everything deactivated / deleted), so there is no corpus to build from. Finish parsing first.",
  "这个库还没有建成的图,不能增量并入,先整库建图": "This knowledge base has no completed graph yet, so nothing can be appended; run a full build first",
  "实体抽取结果为空:所有单元都没有抽出任何实体,检查模型与类型表": "Entity extraction returned nothing: no unit produced any entity; check the model and the type table",
  "合并后没有任何实体": "No entities left after merging",
  "知识库在建图期间被关闭,这一版不发表": "The knowledge base was turned off during the build; this version is not published",
  "建图被停止信号中断": "Graph build interrupted by a stop signal",
  "知识图谱已被关闭,这次建图不启动": "The knowledge graph has been turned off; this build does not start",
  "建图已被暂停,这次建图不启动": "The graph build has been paused; this build does not start",
  "知识图谱已被关闭,这一版不发表": "The knowledge graph has been turned off; this version is not published",
  "建图已被暂停,这一版不发表": "The graph build has been paused; this version is not published",
  "活跃切片取不到正文,无法采样": "The active chunks have no text to sample from",
  "采样数量必须是整数": "Sample size must be an integer",
  "graph_parent_types 必须是对象": "graph_parent_types must be an object",
  "graph_examples 必须是字符串": "graph_examples must be a string",
  "graph_llm 必须是对象": "graph_llm must be an object",
  "graph_schema_active 必须是字符串": "graph_schema_active must be a string",
  "控制台需要访问令牌": "console token required",
  "retention_days 必须不小于 1": "retention_days must be >= 1",
  "Neo4j 密码未配置,图谱投影原样保留": "neo4j password not configured; projection left untouched",
};

// Server strings with numbers / variables: stage strings, timeline events, chunk diagnostics, validation
// errors. The first match wins. Both full-width and ASCII brackets and commas are accepted; $1... are
// capture groups, and a function replacement can translate the Chinese fragments inside.
const ZH_EN_PATTERNS = [
  [/^图片描述[(（]VLM (\d+)\/(\d+)[)）]$/, "Describing images (VLM $1/$2)"],
  [/^文本向量化[(（](\d+) 块[)）]$/, "Text embedding ($1 blocks)"],
  [/^开始解析 size=(\S+) profile=(.+)$/, "Parsing started, size=$1, profile=$2"],
  [/^解析得到 (\d+) 个块$/, "Parsed into $1 blocks"],
  [/^文字层丢字未补回[,，](\d+) 个块带降级标记$/, "Characters lost from the text layer were not recovered; $1 blocks carry the degraded mark"],
  [/^表格粘连 (\d+) 张[:：]截图核验拆开 (\d+) 张[,，]未核验 (\d+) 张[(（]未核验的值不进可信参数[)）]$/, "Merged cells in $1 tables: $2 split after screenshot check, $3 unverified (unverified values are excluded from trusted parameters)"],
  [/^切块 (\d+) 片 · 验收通过$/, "$1 chunks · check passed"],
  [/^切块 (\d+) 片 · 切块提示[:：] ?([\s\S]*)$/, (m, n, rest) => `${n} chunks · chunking notes: ${translateSegments(rest)}`],
  [/^文本向量 (\d+) 条$/, "$1 text vectors"],
  [/^关键词索引失败[,，]已另排任务重试[:：] ?([\s\S]*)$/, "Keyword indexing failed, a retry task was queued: $1"],
  [/^服务未就绪[,，](\d+)s 后再试[(（]不计重试次数[)）][:：] ?([\s\S]*)$/, "Service not ready, trying again in $1s (not counted as a retry): $2"],
  [/^视觉嵌入拒收 (\d+) 张图[,，]这些图没有视觉向量$/, "The visual embedding service rejected $1 images; they have no visual vector"],
  [/^入库 (\d+) 点[(（]跨模态向量 (\d+)[)）][,，]耗时 (.+)$/, "Stored $1 points ($2 visual vectors), took $3"],
  [/^VLM 描述失败 (\d+)\/(\d+) 张[,，]本次解析不入库以免留下没有图片描述的文档$/, "VLM description failed for $1/$2 images; this parse was not stored so that no document is left without image descriptions"],
  // A job's error string is the repr of the exception (NonRetryableParseError('...')); the outer layer is kept as it is
  [/^(\w+\(["'])?嵌入服务拒收第 (\d+)\/(\d+) 片[(（]block_id=(.+?)[,，](\d+) token[)）][:：] ?([\s\S]*)$/, (m, wrap, i, n, block, tokens, rest) => `${wrap || ""}The embedding service rejected chunk ${i}/${n} (block_id=${block}, ${tokens} tokens): ${rest}`],
  [/^整份文档只有 1 块却有 (\d+) token[,，]超过预算两倍$/, "The whole document is a single chunk of $1 tokens, over twice the budget"],
  [/^最长的块只有 (\d+) token[,，]远小于预算 (\d+)$/, "The longest chunk has only $1 tokens, far below the budget of $2"],
  [/^(\d+) 个文本块超过预算 (\d+) token 的 ([\d.]+) 倍$/, "$1 text chunks exceed $3× the budget of $2 tokens"],
  [/^(\d+) 个只有标题的碎片$/, "$1 heading-only fragments"],
  [/^(\d+) 张图片的识读文字超过预算 ([\d.]+) 倍[,，]VLM \/ MinerU 输出可能失控$/, "$1 images whose recognized text exceeds $2× the budget; VLM / MinerU output may have run away"],
  [/^(\d+)\/(\d+) 个表格片没有表头$/, "$1/$2 table chunks have no header"],
  [/^(.+?)[(（]已完成[,，]跳过[)）]$/, (m, label) => `${t(label)} (done, skipped)`],
  [/^(.+?)[(（]第 (\d+) 次失败[,，](\d+)s 后重试[)）]$/, (m, label, n, s) => `${t(label)} (attempt ${n} failed, retrying in ${s}s)`],
  [/^实体抽取 (\d+)\/(\d+) 单元[(（]缓存 (\d+)[)）]$/, "Entity extraction $1/$2 units (cached $3)"],
  [/^实体归并 (\d+)\/(\d+) 批$/, "Entity resolution $1/$2 batches"],
  [/^描述摘要 · 实体 (\d+)\/(\d+)$/, "Description summaries · entities $1/$2"],
  [/^描述摘要 · 关系 (\d+)\/(\d+)$/, "Description summaries · relations $1/$2"],
  [/^结构化事实 (\d+)\/(\d+) 表格单元[(（]缓存 (\d+)[)）]$/, "Structured facts $1/$2 table units (cached $3)"],
  [/^编译视图页 · (主体|时间线|来源|叙述)( \d+\/\d+)?$/, (m, part, n) => `Compiling view pages · ${({ "主体": "subjects", "时间线": "timelines", "来源": "sources", "叙述": "narratives" })[part]}${n || ""}`],
  [/^写入向量库 · (实体|关系|事实|页面) (\d+)\/(\d+)(?:[(（]沿用 (\d+)[)）])?$/, (m, what, a, b, reused) => `Writing vectors · ${({ "实体": "entities", "关系": "relations", "事实": "facts", "页面": "pages" })[what]} ${a}/${b}${reused ? ` (reused ${reused})` : ""}`],
  [/^建图步骤 (\S+) 引用的模型 (.+) 不在注册表中$/, "Graph step $1 references model $2, which is not in the registry"],
  [/^关键词索引创建失败[(（](.+?)[)）][,，]解析写入时会重试$/, "Keyword index creation failed ($1); it is retried when parse results are written"],
  [/^删除未完成[,，]以下部分失败[(（]系统会在下次维护时接着删[,，]也可以再点一次「删除知识库」[)）][:：]([\s\S]*)$/, (m, rest) => `Deletion incomplete, these parts failed (it continues at the next maintenance run, or click “Delete knowledge base” again): ${rest.replace(/；/g, "; ")}`],
  [/^(\S+) 的彻底删除还没有完成[,，]不能重新开启或沿用[;；]系统会在下次维护时接着删[,，]也可以在控制台再点一次「删除知识库」$/, "The permanent deletion of $1 has not finished, so it cannot be re-enabled or adopted; it continues at the next maintenance run, or click “Delete knowledge base” again in the console"],
  [/^图谱删除未完成[,，]以下部分失败[(（]开关保持开启以便重试[)）][:：]([\s\S]*)$/, (m, rest) => `Graph deletion incomplete, these parts failed (the switch stays on so it can be retried): ${rest.replace(/；/g, "; ")}`],
  [/^系统正忙[,，](.+?)会中断正在进行的任务[:：]([\s\S]*?)。确认要强制(.+?)请再点一次。$/, (m, action, reasons) => `The system is busy; ${t(action).toLowerCase()} would interrupt running work: ${translateSegments(reasons)}. Click again to force it.`],
  // Errors from the API and service layers
  [/^(\w*Error)[:：] ([\s\S]+)$/, (m, name, rest) => `${name}: ${t(rest)}`],
  [/^(\S+) 必填$/, "$1 is required"],
  [/^(\S+) 必须是(整数|字符串|布尔值|列表|对象)$/, (m, key, kind) => `${key} must be ${({ "整数": "an integer", "字符串": "a string", "布尔值": "a boolean", "列表": "a list", "对象": "an object" })[kind]}`],
  [/^外部服务不可达[:：] ?([\s\S]*)$/, "External service unreachable: $1"],
  [/^无法触发扫描\/解析[(（]scan=(\S+), worker=(\S+)[)）][;；]请检查后台服务$/, "Could not trigger the scan / parse (scan=$1, worker=$2); check the background services"],
  [/^当前不能并入[(（](.*)[)）]$/, "Cannot append right now ($1)"],
  [/^建图步骤未选择模型[:：] ?(.+)$/, "No model selected for the graph build steps: $1"],
  [/^未知建图步骤[:：] ?(.+)$/, "Unknown graph build step: $1"],
  [/^现行版本 (\S+) 的图谱产物不在磁盘上[(（](.+?)[)）][,，]重新建图后再看$/, "The graph artefact of the current version $1 is not on disk ($2); rebuild the graph first"],
  [/^抽取失败[(（](\w+)[)）][:：]([\s\S]*)$/, (m, name, rest) => `Extraction failed (${name}): ${t(rest)}`],
  [/^向量库集合创建失败[(（](.+?)[)）][,，]首次解析时会自动重建$/, "Vector collection creation failed ($1); it is created again at the first parse"],
  [/^连通性测试失败[:：] ?无法访问 (\S+?)[(（](\w+)[)）]$/, "Connectivity test failed: cannot reach $1 ($2)"],
  [/^连通性测试失败[:：] ?服务端要求跳转[(（](\d+) → (.*?)[)）][,，]请直接填写最终地址$/, "Connectivity test failed: the server asked for a redirect ($1 → $2); enter the final address directly"],
  [/^连通性测试失败[:：] ?HTTP (\d+) ?([\s\S]*)$/, "Connectivity test failed: HTTP $1 $2"],
  [/^连通性测试未通过[:：] ?模型回复「([\s\S]*)」[,，]未包含 OK$/, "Connectivity test not passed: the model replied “$1”, which does not contain OK"],
  [/^内置模型「(.+?)」不可删除$/, "The built-in model “$1” cannot be deleted"],
  [/^「(.+?)」引用的模型 (.+) 不在注册表中$/, (m, step, name) => `The model ${name} referenced by “${t(step)}” is not in the registry`],
  [/^知识库「(.+?)」还没有选择「(.+?)」[:：]在控制台该库的「知识图谱」一栏里选一个[,，]然后点「保存配置」$/, (m, kb, step) => `Knowledge base “${kb}” has no “${t(step)}” selected: pick one under “Knowledge graph” in the console, then click “Save”`],
  [/^建图进程在另一台主机[(（](.+?)[)）]上[,，]无法从这里终止$/, "The build process runs on another host ($1) and cannot be stopped from here"],
  [/^解析任务无法终止[,，]已放弃删除以免留下僵尸集合[;；]请稍后重试[(（]terminated=([\s\S]*)[)）]$/, "The parse jobs could not be stopped; nothing was deleted, to avoid leaving an orphan collection. Try again in a moment (terminated=$1)"],
  [/^(\S+) 的目录「(.+?)」还在[,，]不能同时指向「(.+?)」$/, "The directory “$2” of $1 still exists; it cannot point to “$3” as well"],
  [/^目录「(.+?)」已经登记为 (\S+)$/, "Directory “$1” is already registered as $2"],
  [/^标签版本不存在或已被淘汰[:：] ?(.+)$/, "The label version does not exist or has been rotated out: $1"],
  // Config validation (limits.py)
  [/^max_tokens 必须在 (\d+)–(\d+) 之间[(（]上限 = (.+) 的 (\d+)%[)）]$/, (m, lo, hi, base, pct) => `max_tokens must be between ${lo} and ${hi} (cap = ${pct}% of ${t(base)})`],
  [/^embedding 服务 max_model_len=(\d+)$/, "the embedding service’s max_model_len=$1"],
  [/^embedding 服务未响应[,，]按 (\d+) 兜底$/, "the fallback $1, the embedding service did not respond"],
  [/^overlap_tokens 必须在 0–(\d+) 之间[(（]需小于 min[(（](\d+) − (\d+), (\d+) ÷ 2[)）] = (\d+)[)）]$/, "overlap_tokens must be between 0 and $1 (below min($2 − $3, $4 ÷ 2) = $5)"],
  [/^每个抽取单元合并的切片数必须在 (\d+)–(\d+) 之间[(（]默认 (\d+)[,，]1 = 不合并[;；]过大会让实体抽取漏得多[)）]$/, "Chunks per unit must be between $1 and $2 (default $3, 1 = no merging; too many makes entity extraction miss more)"],
  [/^补漏轮数必须在 (\d+)–(\d+) 之间$/, "Gleaning rounds must be between $1 and $2"],
  [/^实体类型最多 (\d+) 项[(（]当前 (\d+) 项[)）][;；]类型表越窄抽取越准[,，]留空则回退到全局默认$/, "At most $1 entity types (currently $2); a narrower type list extracts more accurately, empty falls back to the global default"],
  [/^实体类型 ([\s\S]+?)… 过长[(（]上限 (\d+) 字符[)）]$/, "Entity type $1… is too long (at most $2 characters)"],
  [/^采样数量必须在 (\d+)–(\d+) 之间[(（]默认 (\d+)[;；]样本全文会拼进一个 prompt[)）]$/, "The sample size must be between $1 and $2 (default $3; the samples are concatenated into one prompt)"],
  [/^谓词最多 (\d+) 条$/, "At most $1 predicates"],
  [/^类型父类最多 (\d+) 个$/, "At most $1 type parents"],
];

const ZH_RE = /[一-鿿]/;
const PUNCT = { "（": "(", "）": ")", "：": ":", "，": ",", "；": ";" };
const normPunct = s => s.replace(/[（）：，；]/g, c => PUNCT[c]);
const ZH_EN_NORM = new Map(Object.entries(ZH_EN).map(([k, v]) => [normPunct(k), v]));

function lookupExact(s) {
  if (Object.prototype.hasOwnProperty.call(ZH_EN, s)) return ZH_EN[s];
  const hit = ZH_EN_NORM.get(normPunct(s));
  return hit == null ? null : hit;
}
function lookupPattern(s) {
  for (const [re, rep] of ZH_EN_PATTERNS) {
    if (re.test(s)) return typeof rep === "function" ? s.replace(re, rep) : s.replace(re, rep);
  }
  return null;
}
// Strings assembled by the server: "this build predates the fingerprint record · corpus changed", "A; B",
// "x, y" -- split by separator, translate each segment, then join with the English separator
const SEG_SEPS = [[" · ", " · "], ["；", "; "], ["、", ", "], [";", "; "], [",", ", "]];
function translateSegments(s) {
  for (const [sep, join] of SEG_SEPS) {
    if (!s.includes(sep)) continue;
    const parts = s.split(sep);
    const out = parts.map(p => { const q = p.trim(); if (!q || !ZH_RE.test(q)) return p; const hit = lookupExact(q) ?? lookupPattern(q); return hit == null ? p : hit; });
    if (out.some((p, i) => p !== parts[i])) return out.join(join);
  }
  return s;
}
function translate(s) {
  const exact = lookupExact(s);
  if (exact != null) return exact;
  if (!ZH_RE.test(s)) return s;
  const trimmed = s.trim();
  if (trimmed !== s) {
    const inner = translate(trimmed);
    if (inner !== trimmed) return s.replace(trimmed, inner);
  }
  const pat = lookupPattern(s);
  if (pat != null) return pat;
  return translateSegments(s);
}
// ── server strings in the Chinese UI: English from the app, reverse lookup in ZH_EN; variable ones by pattern ──
const EN_ZH = new Map();
for (const [zh, en] of Object.entries(ZH_EN)) if (!EN_ZH.has(en)) EN_ZH.set(en, zh);
// When one English string maps to several Chinese ones, server strings use these
for (const [en, zh] of Object.entries({ "Done": "完成", "Queued": "入队", "Started": "开始处理", "Text embedding": "文本向量化",
  "Visual embedding": "视觉向量化", "Files": "文件", "Delete": "删除", "Restart": "重启", "Save": "保存" })) EN_ZH.set(en, zh);
const STAGE_PART_ZH = { subjects: "主体", timelines: "时间线", sources: "来源", narration: "叙述", entities: "实体", relations: "关系", facts: "事实", pages: "页面" };
const EN_ZH_PATTERNS = [
  [/^Describing images \(VLM (\d+)\/(\d+)\)$/, "图片描述(VLM $1/$2)"],
  [/^Knowledge base directory (.+?) is a symbolic link; set KB_MIRROR_ALLOW_LINKED_DIRS=1 to read linked directories$/, "知识库目录 $1 是符号链接;要读取链接目录请设置 KB_MIRROR_ALLOW_LINKED_DIRS=1"],
  [/^Knowledge base directory (.+?) is missing$/, "知识库目录 $1 不存在"],
  [/^File (.+?) is outside the enrolled directory \(symbolic link\); not parsed$/, "文件 $1 在登记目录之外(符号链接),不解析"],
  [/^Text embedding \((\d+) blocks\)$/, "文本向量化($1 块)"],
  [/^Parsing started, size=(\S+), profile=(.+)$/, "开始解析 size=$1 profile=$2"],
  [/^Parsed into (\d+) blocks$/, "解析得到 $1 个块"],
  [/^Characters lost from the text layer were not recovered; (\d+) blocks carry the degraded mark$/, "文字层丢字未补回,$1 个块带降级标记"],
  [/^Merged cells in (\d+) tables: (\d+) split after screenshot check, (\d+) unverified \(unverified values are excluded from trusted parameters\)$/, "表格粘连 $1 张:截图核验拆开 $2 张,未核验 $3 张(未核验的值不进可信参数)"],
  [/^(\d+) chunks · check passed$/, "切块 $1 片 · 验收通过"],
  [/^(\d+) chunks · chunking notes: ([\s\S]*)$/, (m, n, rest) => `切块 ${n} 片 · 切块提示: ${translateSegmentsZh(rest)}`],
  [/^(\d+) text vectors$/, "文本向量 $1 条"],
  [/^Keyword indexing failed, a retry task was queued: ([\s\S]*)$/, "关键词索引失败,已另排任务重试: $1"],
  [/^Service not ready, trying again in (\d+)s \(not counted as a retry\): ([\s\S]*)$/, "服务未就绪,$1s 后再试(不计重试次数): $2"],
  [/^The visual embedding service rejected (\d+) images; they have no visual vector$/, "视觉嵌入拒收 $1 张图,这些图没有视觉向量"],
  [/^Stored (\d+) points \((\d+) visual vectors\), took (.+)$/, "入库 $1 点(跨模态向量 $2),耗时 $3"],
  [/^VLM description failed for (\d+)\/(\d+) images; this parse was not stored so that no document is left without image descriptions$/, "VLM 描述失败 $1/$2 张,本次解析不入库以免留下没有图片描述的文档"],
  // A job's error string is the repr of the exception (NonRetryableParseError('...')); the outer layer is kept as it is
  [/^(\w+\(["'])?The embedding service rejected chunk (\d+)\/(\d+) \(block_id=(.+?), (\d+) tokens\): ([\s\S]*)$/, (m, wrap, i, n, block, tokens, rest) => `${wrap || ""}嵌入服务拒收第 ${i}/${n} 片(block_id=${block},${tokens} token):${rest}`],
  [/^The whole document is a single chunk of (\d+) tokens, over twice the budget$/, "整份文档只有 1 块却有 $1 token,超过预算两倍"],
  [/^The longest chunk has only (\d+) tokens, far below the budget of (\d+)$/, "最长的块只有 $1 token,远小于预算 $2"],
  [/^(\d+) text chunks exceed ([\d.]+)× the budget of (\d+) tokens$/, "$1 个文本块超过预算 $3 token 的 $2 倍"],
  [/^(\d+) heading-only fragments$/, "$1 个只有标题的碎片"],
  [/^(\d+) images whose recognized text exceeds ([\d.]+)× the budget; VLM \/ MinerU output may have run away$/, "$1 张图片的识读文字超过预算 $2 倍,VLM / MinerU 输出可能失控"],
  [/^(\d+)\/(\d+) table chunks have no header$/, "$1/$2 个表格片没有表头"],
  [/^(.+?) \(done, skipped\)$/, (m, label) => `${translateZh(label)}(已完成,跳过)`],
  [/^(.+?) \(attempt (\d+) failed, retrying in (\d+)s\)$/, (m, label, n, sec) => `${translateZh(label)}(第 ${n} 次失败,${sec}s 后重试)`],
  [/^Entity extraction (\d+)\/(\d+) units \(cached (\d+)\)$/, "实体抽取 $1/$2 单元(缓存 $3)"],
  [/^Entity resolution (\d+)\/(\d+) batches$/, "实体归并 $1/$2 批"],
  [/^Description summaries · entities (\d+)\/(\d+)$/, "描述摘要 · 实体 $1/$2"],
  [/^Description summaries · relations (\d+)\/(\d+)$/, "描述摘要 · 关系 $1/$2"],
  [/^Structured facts (\d+)\/(\d+) table units \(cached (\d+)\)$/, "结构化事实 $1/$2 表格单元(缓存 $3)"],
  [/^Compiling view pages · (subjects|timelines|sources|narration)( \d+\/\d+)?$/, (m, part, n) => `编译视图页 · ${STAGE_PART_ZH[part]}${n || ""}`],
  [/^Writing vectors · (entities|relations|facts|pages) (\d+)\/(\d+)(?: \(reused (\d+)\))?$/, (m, what, a, b, reused) => `写入向量库 · ${STAGE_PART_ZH[what]} ${a}/${b}${reused ? `(沿用 ${reused})` : ""}`],
  [/^The system is busy; a (\w+) would interrupt running work: ([\s\S]*?)\. Click again to force the \1\.$/, (m, action, reasons) => {
    const a = ({ restart: "重启", shutdown: "关闭" })[action] || action;
    return `系统正忙,${a}会中断正在进行的任务:${translateSegmentsZh(reasons)}。确认要强制${a}请再点一次。`; }],
  [/^Deletion incomplete, these parts failed \(it continues at the next maintenance run, or click ["“]Delete knowledge base["”] again\): ([\s\S]*)$/, (m, rest) => `删除未完成,以下部分失败(系统会在下次维护时接着删,也可以再点一次「删除知识库」):${translateSegmentsZh(rest)}`],
  [/^The permanent deletion of (\S+) has not finished, so it cannot be re-enabled or adopted; it continues at the next maintenance run, or click ["“]Delete knowledge base["”] again in the console$/, "$1 的彻底删除还没有完成,不能重新开启或沿用;系统会在下次维护时接着删,也可以在控制台再点一次「删除知识库」"],
  [/^Graph deletion incomplete, these parts failed \(the switch stays on so it can be retried\): ([\s\S]*)$/, (m, rest) => `图谱删除未完成,以下部分失败(开关保持开启以便重试):${translateSegmentsZh(rest)}`],
  [/^Vector collection creation failed \((.+?)\); it is recreated automatically at the first parse$/, "向量库集合创建失败($1),首次解析时会自动重建"],
  [/^Keyword index creation failed \((.+?)\); it is retried when parse results are written$/, "关键词索引创建失败($1),解析写入时会重试"],
  [/^Unknown graph step: (.+)$/, "未知建图步骤: $1"],
  [/^Graph step (\S+) references model (.+), which is not in the registry$/, "建图步骤 $1 引用的模型 $2 不在注册表中"],
  [/^Could not trigger the scan \/ worker \((.+?)\); check the background services$/, "无法触发扫描/解析($1);请检查后台服务"],
  [/^No model selected for graph steps: (.+)$/, "建图步骤未选择模型: $1"],
  [/^Cannot append right now \((.+?)\)$/, "当前不能并入($1)"],
  [/^Extraction failed \((.+?)\): ([\s\S]*)$/, (m, name, rest) => `抽取失败(${name}):${translateZh(rest)}`],
  [/^The graph artifacts of the current version (\S+) are not on disk \((.+?)\); rebuild the graph and try again$/, "现行版本 $1 的图谱产物不在磁盘上($2),重新建图后再看"],
  [/^Connectivity test failed: cannot reach (\S+) \((.+?)\)$/, "连通性测试失败: 无法访问 $1($2)"],
  [/^Connectivity test failed: the service redirected \((\d+) → (.*?)\); enter the final address directly$/, "连通性测试失败: 服务返回重定向 $1 → $2,请直接填写最终地址"],
  [/^Connectivity test failed: HTTP ([\s\S]*)$/, "连通性测试失败: HTTP $1"],
  [/^Connectivity test did not pass: the model replied “([\s\S]*)”, which does not contain OK$/, "连通性测试未通过: 模型回复「$1」,未包含 OK"],
  [/^(\w+) must be an integer$/, "$1 必须是整数"],
  [/^(\w+) must be a string$/, "$1 必须是字符串"],
  [/^(\w+) must be a boolean$/, "$1 必须是布尔值"],
  [/^(\w+) must be a list$/, "$1 必须是列表"],
  [/^(\w+) must be an object$/, "$1 必须是对象"],
  [/^External service unreachable: ([\s\S]*)$/, "外部服务不可达: $1"],
  [/^Parse jobs could not be terminated; deletion abandoned rather than leaving orphaned collections\. Try again later \((.+?)\)$/, "解析任务无法终止,已放弃删除以免留下僵尸集合;请稍后重试($1)"],
  [/^The graph build runs on another host \((.+?)\) and cannot be terminated from here$/, "建图进程在另一台主机($1)上,无法从这里终止"],
  [/^The graph build process did not confirm stopping \((.+?)\); the record stays running$/, "建图进程没有确认停止($1),记录保持 running"],
  [/^Graph build process no longer exists \(pid=(\d+)\); marked failed by the reaper$/, "建图进程已不存在(pid=$1),记录由回收流程判为失败"],
  [/^Graph build process ended with the last shutdown or power loss \(pid=(\d+)\); marked failed by the reaper$/, "建图进程随上次关机或断电结束(pid=$1),记录由回收流程判为失败"],
  [/^Graph build process no longer exists \(pid=(\d+), it does not hold the build lock\); marked failed by the reaper$/, "建图进程已不存在(pid=$1,建图锁不在它手里),记录由回收流程判为失败"],
  [/^Graph build heartbeat timed out \((\d+)s\); marked failed$/, "建图记录心跳超时($1s),判为失败"],
  [/^Built-in model “(.+?)” cannot be deleted$/, "内置模型「$1」不可删除"],
  [/^Retry in (\d+)s: ([\s\S]*)$/, "$1s 后重试: $2"],
  [/^The directory “(.+?)” of (\S+) still exists; it cannot also point to “(.+?)”$/, "$2 的目录「$1」还在,不能同时指向「$3」"],
  [/^Directory “(.+?)” is already registered as (\S+)$/, "目录「$1」已经登记为 $2"],
  [/^Knowledge base “(.+?)” has no “(.+?)” selected: pick one in the console's Knowledge graph section and click Save$/, (m, kb, step) => `知识库「${kb}」还没有选择「${translateZh(step)}」:在控制台该库的「知识图谱」一栏里选一个,然后点「保存配置」`],
  [/^“(.+?)” references model (.+), which is not in the registry$/, (m, step, name) => `「${translateZh(step)}」引用的模型 ${name} 不在注册表中`],
  [/^Structured facts failed for (\d+) units \((.+?)\): ([\s\S]*?); if retries keep failing, set KB_GRAPH_FACTS_PARTIAL_OK=1 to accept a partial release explicitly$/, "结构化事实有 $1 个单元没抽成($2):$3;重试后仍失败可设 KB_GRAPH_FACTS_PARTIAL_OK=1 明确接受部分发布"],
  [/^The frozen chunk ledger does not match the main store: (\d+) points missing, (\d+) text fingerprints differ, (\d+) content versions differ\. This build does not start; the next check retries$/, "冻结的切片账与主库对不上:主库缺 $1 个点、正文指纹不符 $2 个、内容版本不符 $3 个。这次建图不启动,等下一轮检查重试"],
  [/^Neo4j is unavailable \((.+?)\); the build does not start so LLM calls are not wasted\. Fix Neo4j or set GRAPH_NEO4J_IMPORT_AFTER_BUILD=0$/, "Neo4j 不可用($1),建图不启动以免白跑 LLM;修好 Neo4j 或把 GRAPH_NEO4J_IMPORT_AFTER_BUILD 设为 0"],
  [/^Entity extraction failed for (\d+)\/(\d+) units \(over (\d+%)\); successful units are stored and a rerun only redoes the failed ones\. Last error: ([\s\S]*)$/, "实体抽取有 $1/$2 个单元失败(超过 $3),已成功的单元已落库,重跑只补失败的;最后一个错误:$4"],
  [/^Graph build interrupted by signal (\d+)$/, "建图被信号 $1 中断"],
  [/^Graph build interrupted by a stop signal \(([\s\S]+)\)$/, "建图被停止信号中断($1)"],
  [/^Chunks per extraction unit must be between (\d+) and (\d+) \(default (\d+); 1 = no merging; large values make entity extraction miss more\)$/, "每个抽取单元合并的切片数必须在 $1–$2 之间(默认 $3,1 = 不合并;过大会让实体抽取漏得多)"],
  [/^Gleaning rounds must be between (\d+) and (\d+)$/, "补漏轮数必须在 $1–$2 之间"],
  [/^At most (\d+) entity types \(currently (\d+)\); a narrower type table extracts more precisely, and an empty one falls back to the global default$/, "实体类型最多 $1 项(当前 $2 项);类型表越窄抽取越准,留空则回退到全局默认"],
  [/^Entity type (.+?)… is too long \(at most (\d+) characters\)$/, "实体类型 $1… 过长(上限 $2 字符)"],
  [/^Sample size must be between (\d+) and (\d+) \(default (\d+); the full sample text goes into one prompt\)$/, "采样数量必须在 $1–$2 之间(默认 $3;样本全文会拼进一个 prompt)"],
  [/^At most (\d+) predicates$/, "谓词最多 $1 条"],
  [/^At most (\d+) parent types$/, "类型父类最多 $1 个"],
  [/^max_tokens must be between (\d+) and (\d+) \(the cap is (\d+)% of (.+)\)$/, (m, lo, hi, pct, base) => `max_tokens 必须在 ${lo}–${hi} 之间(上限 = ${translateZh(base)} 的 ${pct}%)`],
  [/^the embedding service's max_model_len=(\d+)$/, "embedding 服务 max_model_len=$1"],
  [/^the embedding service did not respond, falling back to (\d+)$/, "embedding 服务未响应,按 $1 兜底"],
  [/^overlap_tokens must be between 0 and (\d+) \(it must be below min\((.+?)\) = (\d+)\)$/, "overlap_tokens 必须在 0–$1 之间(需小于 min($2) = $3)"],
  [/^The label version does not exist or has been rotated out: (.+)$/, "标签版本不存在或已被淘汰: $1"],
  [/^(\d+)\/(\d+) text chunks are under (\d+) tokens: too fragmented$/, "$1/$2 个文本块不足 $3 token,碎片过多"],
  [/^File exceeds the single-file limit \((\d+) bytes > (\d+)\) and was skipped; raise KB_MAX_FILE_BYTES if it must be ingested$/, "文件超过单文件上限($1 字节 > $2),已跳过;如确需入库请调大 KB_MAX_FILE_BYTES"],
  // An exception name in front of a message (a 500 from the console backend): the message itself is translated
  [/^(\w*Error): ([\s\S]+)$/, (m, name, rest) => `${name}: ${translateZh(rest)}`],
];
function lookupPatternZh(s) {
  for (const [re, rep] of EN_ZH_PATTERNS) {
    if (re.test(s)) return s.replace(re, rep);
  }
  return null;
}
// Strings the server joins with "; ", ", " or " · ": reverse-look up each segment, then join with the
// Chinese separator; returned unchanged when no segment is recognized
const SEG_SEPS_EN = [[" · ", " · "], ["; ", "；"], [", ", "、"]];
function translateSegmentsZh(s) {
  for (const [sep, join] of SEG_SEPS_EN) {
    if (!s.includes(sep)) continue;
    const parts = s.split(sep);
    const out = parts.map(p => { const q = p.trim(); if (!q) return p; const hit = EN_ZH.get(q) ?? lookupPatternZh(q); return hit == null ? p : hit; });
    if (out.some((p, i) => p !== parts[i])) return out.join(join);
  }
  return s;
}
function translateZh(s) {
  if (!s || ZH_RE.test(s)) return s;
  const exact = EN_ZH.get(s);
  if (exact != null) return exact;
  const trimmed = s.trim();
  if (trimmed !== s) {
    const inner = translateZh(trimmed);
    if (inner !== trimmed) return s.replace(trimmed, inner);
  }
  const pat = lookupPatternZh(s);
  if (pat != null) return pat;
  return translateSegmentsZh(s);
}
function t(s, ...args) {
  if (typeof s !== "string" || !s) return s;
  const out = I18N.lang === "en" ? translate(s) : translateZh(s);
  return args.length ? out.replace(/\{(\d+)\}/g, (m, i) => (args[+i] === undefined ? m : String(args[+i]))) : out;
}

// Static page: text nodes and placeholder / title / aria-label, multi-line values line by line. data-en
// overrides when the same Chinese needs different English in different places
function applyStatic(root) {
  if (!root) return;
  const attrs = ["placeholder", "title", "aria-label"];
  for (const el of root.querySelectorAll("*")) {
    for (const a of attrs) {
      const v = el.getAttribute(a);
      if (v && ZH_RE.test(v)) el.setAttribute(a, v.includes("\n") ? v.split("\n").map(line => t(line)).join("\n") : t(v));
    }
  }
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  for (const node of nodes) {
    const parent = node.parentElement;
    if (!parent || parent.tagName === "SCRIPT" || parent.tagName === "STYLE") continue;
    const v = node.nodeValue;
    if (!ZH_RE.test(v)) continue;
    const m = v.match(/^(\s*)([\s\S]*?)(\s*)$/);
    const en = parent.dataset && parent.dataset.en && parent.childNodes.length === 1 ? parent.dataset.en : t(m[2]);
    if (en !== m[2]) node.nodeValue = m[1] + en + m[3];
  }
}

if (typeof document !== "undefined") {
  document.documentElement.lang = I18N.lang === "en" ? "en" : "zh-CN";
  if (I18N.lang === "en") {
    document.title = t(document.title);
    applyStatic(document.body);
  }
  const sw = document.getElementById("lang-switch");
  if (sw) {
    sw.querySelectorAll("button[data-lang]").forEach(b => {
      b.classList.toggle("on", b.dataset.lang === I18N.lang);
      b.addEventListener("click", () => {
        if (b.dataset.lang === I18N.lang) return;
        try { localStorage.setItem("kb.lang", b.dataset.lang); } catch (e) { /* if it cannot be stored, switch just this once */ }
        location.reload();
      });
    });
  }
}
if (typeof module !== "undefined" && module.exports) module.exports = { t, translate, translateSegments, translateZh, ZH_EN, ZH_EN_PATTERNS, EN_ZH, EN_ZH_PATTERNS, I18N };
