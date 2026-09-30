/* Knowledge base console: plain JS, no build chain. Polling /api/overview drives the whole UI.
   Every UI string goes through t() (i18n.js): the Chinese original is the key and the English UI follows
   the dictionary; data such as directory names, file names and entity names is not translated.
   Layout: top bar + knowledge base navigation on the left + workspace on the right (three tabs: config /
   files / graph); the model registry and service status are global and live in a right-hand drawer; the
   file table's "chunk preview" and the job timeline use the right-hand drawer too.
   The config form (#cfg-*) logic is carried over almost verbatim: draft state, the cfgKey overwrite guard,
   the label version ring and monotonic progress clamping were all settled after real mishaps, and
   regression tests guard them. */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

let state = {
  overview: null,
  llms: [],
  selected: null,          // selected top-level directory name (the identity from the user's point of view)
  cfgKey: null,            // dir|state the form was filled for; refill only when it changes, so edits in progress are not overwritten
  draft: null,             // draft state: directory whose switch is on but which is not enrolled yet (save = enroll + parse)
  activePanel: null,       // right-hand drawer: llms / health / side (chunk preview, job timeline)
  side: { kind: null, fileId: null },   // what the side drawer currently holds: chunks / job
  tab: null,               // current workspace tab
  tabByKb: recall("tabs", {}),   // the tab each knowledge base was last on, restored on switching back (kept in localStorage)
  // Derived results of "Extract labels". Not kept in the DOM -- the output language and entity labels are
  // read-only displays with no input to read back from, so collectGraphConfig takes them from here.
  // The label version ring (newest first, <=3) plus the currently selected id. The labels and language in
  // effect are derived server-side from the selected version -- the console never sends them directly,
  // only graph_schema_active.
  // dirty: you touched the dropdown, or a version was just extracted. Only then does a save carry the
  // selection.
  graphSchema: { versions: [], active: "", savedActive: "", note: "", dirty: false, kbId: "" },
  gp: { kbId: null, version: null, viewKey: null, data: null, pos: null, sim: null, hist: [], cur: -1, cam: { k: 1, tx: 0, ty: 0 },
        at: 0, hover: null, drag: null, error: null, raf: 0, loading: null, loadingAt: 0 },   // graph preview: data, layout simulation, levels, zoom / pan; viewKey = KB + version + view parameters; loading = the viewKey of the request in flight
  files: { kbId: null, rows: [], at: 0, page: 1, sort: "path", dir: 1, etag: null },
  jobs: { selected: null },   // the job being viewed in the drawer
  preview: null,           // result of the most recent chunk preview
};

let toastTimer = null;
function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.add("show");
  // Not clearing the previous timer would dismiss this toast early; long messages (several validation
  // errors joined together) also need more time to read.
  if (toastTimer) clearTimeout(toastTimer);
  const ms = Math.min(9000, 2600 + String(msg).length * 40);
  toastTimer = setTimeout(() => { el.classList.remove("show"); toastTimer = null; }, ms);
}

// Access token (KB_WEB_TOKEN on the server): kept in the browser and sent with every /api request; a 401
// prompts once, and after cancelling it is not asked again for a minute
let memToken = "";     // the token typed into the prompt; kept here so a browser that blocks storage still works
function authHeaders() {
  let token = memToken;
  if (!token) { try { token = localStorage.getItem("kb.token") || ""; } catch (e) { /* storage blocked */ } }
  return token ? { "Authorization": "Bearer " + token } : {};
}
let tokenPrompt = null, tokenDeclinedUntil = 0;
async function askToken() {
  if (Date.now() < tokenDeclinedUntil) return false;
  if (tokenPrompt) return tokenPrompt;
  tokenPrompt = (async () => {
    const v = window.prompt(t("控制台设置了访问令牌,请输入 KB_WEB_TOKEN 的值"));
    if (v && v.trim()) {
      memToken = v.trim();
      try { localStorage.setItem("kb.token", memToken); } catch (e) { /* storage blocked: the in-memory copy carries this page */ }
      return true;
    }
    tokenDeclinedUntil = Date.now() + 60000;
    return false;
  })();
  try { return await tokenPrompt; } finally { tokenPrompt = null; }
}

async function api(path, opts = {}) {
  const send = () => fetch("/api" + path, {
    ...opts,
    headers: { "Content-Type": "application/json", ...authHeaders(), ...(opts.headers || {}) },
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const sentWith = authHeaders().Authorization || "";
  let res = await send();
  // several requests go out together on first load; once one of them has obtained the token the others resend
  // with it instead of asking again
  if (res.status === 401 && ((authHeaders().Authorization || "") !== sentWith || await askToken())) res = await send();
  if (!res.ok) {
    let detail = res.statusText || `HTTP ${res.status}`;   // statusText is empty under HTTP/2
    try {
      const body = await res.json();
      const d = body.detail;
      if (Array.isArray(d)) {
        // FastAPI's 422 validation errors are an array; naive string concatenation gives [object Object]
        detail = d.map(x => x?.msg || JSON.stringify(x)).join("；");
      } else if (typeof d === "string" && d) {
        detail = d;
      }
    } catch {}
    throw new Error(detail);
  }
  return res.json();
}

function esc(s) { return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function fmtTs(ts) { return ts ? new Date(ts * 1000).toLocaleString(I18N.locale, { hour12: false }) : "—"; }
function fmtTsShort(ts) {
  return ts ? new Date(ts * 1000).toLocaleString(I18N.locale, { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false }) : "—";
}
function fmtDur(sec) {
  if (sec == null || !Number.isFinite(sec)) return "";
  sec = Math.max(0, Math.round(sec));
  if (sec < 60) return `${sec}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m${sec % 60}s`;
  return `${Math.floor(sec / 3600)}h${Math.floor((sec % 3600) / 60)}m`;
}
function fmtSize(n) {
  if (n == null) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}
function fmtNum(n) { return n == null ? "—" : Number(n).toLocaleString(I18N.locale); }

// Touch the DOM only when the content really changed: polling redraws every few seconds, and overwriting
// as-is makes the button being clicked vanish under the finger and flickers highlights and scroll
// positions. Returns whether it actually wrote -- event listeners are re-attached only on write, the old
// ones stay otherwise. Every write to these containers must go through here, or the cache and the DOM
// drift apart.
const htmlCache = new WeakMap();
function setHtml(el, html) {
  if (htmlCache.get(el) === html) return false;
  htmlCache.set(el, html);
  el.innerHTML = html;
  return true;
}

// Remember the last selected KB and each KB's tab: a page reload lands back where it was. If it cannot be
// stored (private mode), never mind.
function remember(key, value) { try { localStorage.setItem("kb." + key, JSON.stringify(value)); } catch {} }
function recall(key, fallback) {
  try { const v = localStorage.getItem("kb." + key); return v == null ? fallback : JSON.parse(v); }
  catch { return fallback; }
}

function confirmDialog(text, yesLabel = t("确认")) {
  return new Promise(resolve => {
    const dlg = $("#confirm-dlg");
    $("#confirm-text").textContent = text;
    $("#confirm-yes").textContent = yesLabel;
    let settled = false;
    const finish = v => { if (settled) return; settled = true; dlg.removeEventListener("close", onClose); resolve(v); };
    // ESC fires the native cancel/close: unhandled, the caller would hang forever (buttons stay disabled too)
    const onClose = () => finish(false);
    dlg.addEventListener("close", onClose);
    const done = v => { finish(v); dlg.close(); };
    $("#confirm-yes").onclick = () => done(true);
    $("#confirm-no").onclick = () => done(false);
    dlg.showModal();
  });
}

const PARSE_BUSY_MSG = t("当前知识库处于解析任务中,暂不支持操作建图,请稍候再试。");

function noticeDialog(text) {
  // Single-button notice: reuse the confirm dialog, hide Cancel for now and give the confirm button a
  // neutral style
  return new Promise(resolve => {
    const dlg = $("#confirm-dlg");
    $("#confirm-text").textContent = text;
    const no = $("#confirm-no"), yes = $("#confirm-yes");
    no.style.display = "none";
    yes.textContent = t("知道了");
    yes.classList.remove("danger");
    yes.classList.add("primary");
    let settled = false;
    const restore = () => {
      // The styles must be restored: after ESC closes this notice, without restoring, every later confirm
      // dialog would lack its "Cancel" button and its confirm button would no longer be red.
      no.style.display = "";
      yes.classList.add("danger");
      yes.classList.remove("primary");
    };
    const finish = () => {
      if (settled) return;
      settled = true;
      dlg.removeEventListener("close", onClose);
      restore();
      resolve();
    };
    const onClose = () => finish();
    dlg.addEventListener("close", onClose);
    const done = () => { finish(); dlg.close(); };
    yes.onclick = done;
    no.onclick = done;
    dlg.showModal();
  });
}

function selectedEntry() {
  return state.overview?.kbs.find(k => k.dir === state.selected) || null;
}

function kbDot(kb) {
  if (!kb) return "gray";
  if (kb.state === "inactive") return "half";   // deactivated retention period: green / yellow split; after permanent deletion there is no row, back to grey
  if (kb.state !== "active") return "gray";
  if (kb.files_failed > 0) return "red";
  if (kb.jobs_active > 0) return "yellow pulse";   // parsing: pulsing dot, same as the graph dot
  if (kb.files_pending > 0) return "yellow";       // queued, not started yet: static yellow
  if (kb.files_total > 0 && kb.files_indexed === kb.files_total) return "green";
  if (!kb.files_total) return "gray";   // empty directory: nothing to parse, must not show "in progress" forever
  return "yellow";
}

function graphDot(kb) {
  // Graph status dot; the mapping mirrors the parse dot (kbDot)
  if (!kb || !kb.graph_status || kb.graph_status === "disabled") return "gray";
  if (kb.state === "inactive") return "half";
  if (kb.state !== "active") return "gray";
  if (kb.graph_status === "failed") return "red";
  if (kb.graph_status === "ok") return "green";
  if (kb.graph_status === "running") return "yellow pulse";   // building: pulsing dot
  return "yellow";   // pending (enabled, not built yet) / stopped (interrupted, cache kept, resumable)
}

function kbStateText(kb) {
  if (kb.state === "active") return t("已开启");
  if (kb.state === "deleting") return t("正在彻底删除…");
  if (kb.state === "delete_failed") return t("删除未完成 · 下次维护时接着删");
  if (kb.state === "inactive") {
    // Deactivated because the directory vanished, but the directory is back: GC skips it forever, so a day
    // count would be a lie. What matters here is the remedy -- the scan does not revive it automatically,
    // it can only be re-enabled from the console.
    if (kb.gc_exempt) return t("已移出(目录已恢复) · 重新开启即可复原");
    const days = kb.gc_in_seconds != null ? Math.ceil(kb.gc_in_seconds / 86400) : null;
    return t("已移出") + (kb.inactive_reason === "unenrolled" ? "" : t("(目录消失)")) + (days != null ? " · " + t("{0} 天后彻底删除", days) : "");
  }
  if (kb.state === "directory_missing") return t("目录消失,等待处理");
  if (kb.state === "directory_linked") return t("目录是符号链接,已按策略拒绝读取");
  return t("未开启");
}

/* ── right-hand drawers (models / services), mutually exclusive ────────────────────── */
const PANELS = { health: "#health-win", llms: "#llm-win", side: "#side-win" };
const PANEL_BUTTONS = { health: "#btn-health", llms: "#btn-llms" };
let healthWatch = null;

document.addEventListener("keydown", ev => {
  if (ev.key === "Escape" && state.activePanel && !document.querySelector("dialog[open]")) setPanel(null);
});

function setPanel(name) {
  state.activePanel = name;
  for (const [key, sel] of Object.entries(PANELS)) $(sel).classList.toggle("show", name === key);
  for (const [key, sel] of Object.entries(PANEL_BUTTONS)) $(sel).classList.toggle("active", name === key);
  $("#scrim").classList.toggle("show", !!name);
  if (name === "health") refreshHealth();
  if (name === "llms") {
    refreshLlms();
    // Measure only once the panel is really visible: with display:none every width is 0 and a name would
    // be misjudged as fitting, so it would never scroll.
    scheduleNameMarquee($("#llm-win"));
  }
  clearInterval(healthWatch);
  healthWatch = null;
  if (name === "health") healthWatch = setInterval(refreshHealth, 5000);
}
$("#btn-health").addEventListener("click", ev => { ev.stopPropagation(); setPanel(state.activePanel === "health" ? null : "health"); });
$("#btn-llms").addEventListener("click", ev => { ev.stopPropagation(); setPanel(state.activePanel === "llms" ? null : "llms"); });
$$(".panel-close").forEach(b => b.addEventListener("click", () => setPanel(null)));
$("#scrim").addEventListener("click", () => setPanel(null));

/* ── knowledge base navigation (left) ──────────────────────────────────── */
function renderKbNav() {
  const host = $("#kb-nav");
  const all = state.overview?.kbs || [];
  $("#side-count").textContent = all.length ? `${all.filter(k => k.state === "active").length}/${all.length}` : "";
  if (!all.length) {
    setHtml(host, `<div class="side-empty">${t("还没有知识库:在 mirror 目录下新建文件夹,扫描后会出现在这里")}</div>`);
    return;
  }
  // Sidebar filter: case-insensitive substring of the directory name or id; the count still covers all KBs
  const q = ($("#kb-q")?.value || "").trim().toLowerCase();
  const kbs = q ? all.filter(k => k.dir.toLowerCase().includes(q) || String(k.kb_id || "").toLowerCase().includes(q)) : all;
  if (!kbs.length) {
    setHtml(host, `<div class="side-empty">${t("没有名字含「{0}」的知识库", esc(q))}</div>`);
    return;
  }
  const jobsByKb = {};
  (state.overview.active_jobs || []).forEach(j => (jobsByKb[j.kb_id] ||= []).push(j));
  const html = kbs.map(kb => {
    const active = kb.state === "active";
    // Just enabled, scan has not enrolled the files yet: show the disk count for now, so "N files" does not
    // jump to "0 files"
    const count = active ? (kb.files_total || kb.dir_files || 0) : kb.dir_files;
    let sub = "";
    if (active && !kb.files_total && kb.dir_files > 0) {
      sub = `<div class="kb-row-sub"><span class="lbl">${t("等待扫描登记")}</span></div>`;
    } else if (active && kb.files_total > 0) {
      const jobs = jobsByKb[kb.kb_id] || [];
      const reparsing = kb.files_reparsing || 0;
      const fresh = (kb.files_pending || 0) > 0;        // files never indexed remain: draw whole-KB progress
      const busy = kb.jobs_active > 0 || kb.jobs_waiting > 0 || fresh || reparsing > 0;
      // The three kinds of ongoing work (parse, re-parse, graph build) share one thin bar plus one short
      // text in the same spot; the phase name goes only into the hover tip, long text would squeeze the
      // bar away. A running parse draws parse (yellow), otherwise a running build draws build (purple)
      const gs = kb.graph_status, gb = kb.graph_build || {};
      const running = jobs.filter(j => j.status === "running");
      let text = "", bar = null, tip = "";
      if (busy && !fresh && reparsing > 0) {
        // Re-parsed files stay indexed throughout; counting the whole KB would give "113/114 · 99%". Count
        // this round instead: how many files in the round (max seen), how many done, which phase the
        // current one is in
        const round = reparseRound(kb.kb_id, reparsing);
        const frac = running.length ? running.reduce((s, j) => s + stageFraction(j.stage, j, kb.parse_stage_weights), 0) / running.length : 0;
        bar = Math.min(Math.round(((round - reparsing) + frac) / round * 100), 99);
        text = t("重新解析") + (round > 1 ? ` ${round - reparsing}/${round}` : "") + ` · ${bar}%`;
        tip = running.length ? shortStage(running[0].stage) : t("排队中");
      } else if (busy) {
        const p = parseProgress(kb, jobs);
        bar = p.pct;
        text = t("解析 {0}/{1} · {2}%", p.done, p.total, p.pct);
        tip = running.length ? shortStage(running[0].stage) : t("排队中");
      } else if (gs === "running") {
        bar = graphPctFloor(kb.kb_id, gb.started_at, graphStagePct(gb.stage, gb.stage_weights));
        text = t("建图 · {0}%", bar);
        tip = shortStage(gb.stage);
      } else {
        text = "";      // idle: no status text, the two dots already say how parse and build are doing; text only while work runs
      }
      if (!reparsing) reparseRound(kb.kb_id, 0);
      if (text) {
        sub = `<div class="kb-row-sub">
          ${bar != null ? `<span class="mini${busy ? "" : " graph"}"><i style="width:${Math.max(bar, 2)}%"></i></span>` : ""}
          <span class="lbl" title="${esc(tip)}">${esc(text)}</span></div>`;
      }
    } else if (kb.state === "inactive" || kb.state === "directory_missing" || kb.state === "directory_linked"
               || kb.state === "deleting" || kb.state === "delete_failed") {
      // Only states needing a human, like "removed · permanently deleted in N days", get a line; KBs that are
      // not enabled do not say "not enabled"
      sub = `<div class="kb-row-sub"><span>${esc(kbStateText(kb))}</span></div>`;
    }
    return `<div class="kb-row ${kb.dir === state.selected ? "sel" : ""}" data-dir="${esc(kb.dir)}">
      <div class="kb-row-top">
        <span class="dots"><span class="dot ${kbDot(kb)}" title="${t("解析状态")}"></span><span class="dot ${graphDot(kb)}" title="${t("建图状态")}"></span></span>
        <span class="kb-row-name" title="${esc(kb.dir)}${kb.kb_id ? " · " + esc(kb.kb_id) : ""}">${esc(kb.dir)}</span>
        ${count != null ? `<span class="kb-row-n">${t("{0} 个文件", count)}</span>` : ""}
      </div>${sub}</div>`;
  }).join("");
  if (setHtml(host, html)) $$(".kb-row", host).forEach(row => row.addEventListener("click", () => selectKb(row.dataset.dir)));
}
$("#kb-q")?.addEventListener("input", renderKbNav);

// Compress a stage string into "stage name n/m" for the sidebar: "Describing images (VLM 10/13)" ->
// "Describing images 10/13", "Description summaries · entities 23/511" -> "Description summaries 23/511",
// "Entity extraction 104/113 units (cached 101)" -> "Entity extraction 104/113"
function shortStage(stage) {
  const s = String(stage || "");
  const name = s.match(/^(.+?)(?:\s*[(（·]|\s+\d+\s*\/|$)/);
  const n = s.match(/(\d+)\s*\/\s*(\d+)/);
  return name ? t(name[1]) + (n ? ` ${n[1]}/${n[2]}` : "") : t(s);
}

// How many files are in this re-parse round: files_reparsing drops as files finish and grows as new ones
// are clicked. Denominator = the count first seen + every later increase (clicking "re-parse" on other
// files midway merges into the same round and recounts); the round ends when it drops to 0
const reparseRounds = {};
function reparseRound(kbId, remaining) {
  const key = String(kbId);
  if (!remaining) { delete reparseRounds[key]; return 0; }
  const r = reparseRounds[key];
  if (!r) { reparseRounds[key] = { total: remaining, last: remaining }; return remaining; }
  if (remaining > r.last) r.total += remaining - r.last;
  r.last = remaining;
  return r.total;
}

function parseProgress(kb, jobs) {
  // Files being re-parsed stay indexed, but do not count as done in this round's progress
  const done = Math.max(kb.files_indexed - (kb.files_reparsing || 0), 0), total = kb.files_total;
  const running = jobs.filter(j => j.status === "running");
  const frac = running.reduce((s, j) => s + stageFraction(j.stage, j, kb.parse_stage_weights), 0);
  const value = total ? Math.min((done + frac) / total, 1) : 0;
  const pct = Math.min(Math.round(value * 100), done === total ? 100 : 99);
  return { done, total, pct };
}

function selectKb(dir) {
  if (dir === state.selected) return;
  stashDraft(state.selected);   // keep the half-edited content of the current KB first
  if (state.draft && state.draft !== dir) state.draft = null;  // switching KBs abandons the enrollment draft
  state.selected = dir;
  remember("selected", dir);
  state.jobs.selected = null;
  resetWorkspace();
  renderKbNav();
  renderConfig();
}

// Clear the previous KB's table when switching: otherwise, until the new KB's data arrives, a few hundred
// milliseconds show someone else's files
function resetWorkspace() {
  const loading = `<div class="empty-state">${t("载入中…")}</div>`;
  setHtml($("#fl-wrap"), loading);
  setHtml($("#fl-pager"), "");
  $("#fl-count").textContent = "";
  if (state.activePanel === "side") setPanel(null);   // the chunks / timeline in the drawer belong to the previous KB
  state.side = { kind: null, fileId: null };
  state.gp = { ...state.gp, kbId: null, version: null, viewKey: null, data: null, pos: null, sim: null, hist: [], cur: -1, cam: { k: 1, tx: 0, ty: 0 }, hover: null, error: null,
               loading: GP_PENDING };
  renderGraphPreview();                               // canvas and legend still show the previous KB: clear them to "Loading..." first
  state.files = { ...state.files, kbId: null, rows: [], page: 1, etag: null };
}

/* ── workspace tabs ──────────────────────────────────────── */
const TABS = ["config", "files", "graph"];

function setTab(name) {
  if (!TABS.includes(name)) name = "config";
  state.tab = name;
  if (state.selected) { state.tabByKb[state.selected] = name; remember("tabs", state.tabByKb); }
  $$(".ws-tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  TABS.forEach(t => $(`#tab-${t}`).classList.toggle("show", t === name));
  refreshTab(true);
}
$$(".ws-tabs button").forEach(b => b.addEventListener("click", () => setTab(b.dataset.tab)));

// A KB that is not enabled only has "Config": files and graph are post-indexing content
function updateTabs(kb) {
  const active = !!kb && kb.state === "active";
  $$(".ws-tabs button").forEach(b => { b.disabled = !active && b.dataset.tab !== "config"; });
  const failed = active ? (kb.files_failed || 0) : 0;
  const n = $("#tabn-files");
  n.textContent = failed ? t("{0} 失败", failed) : (active && kb.files_total ? kb.files_total : "");
  n.classList.toggle("bad", failed > 0);
  const wanted = state.tabByKb[state.selected] || "config";
  const tab = active ? wanted : "config";
  if (tab !== state.tab || !$(`#tab-${tab}`).classList.contains("show")) setTab(tab);
}

// Refresh the current tab when polling brings new state; force = just switched in, fetch unconditionally
function refreshTab(force = false) {
  const kb = selectedEntry();
  if (!kb || kb.state !== "active") return;
  if (state.tab === "files") refreshFilesTab(force);
  if (state.tab === "graph") { renderGraphPanel(); refreshGraphPreview(force); }
  if (state.activePanel === "side" && state.side.kind === "job") refreshJobDetail();   // the timeline in the drawer follows the poll
}

/* ── config form ────────────────────────────────────────── */
// Preselected when the three model slots are empty, the same name as the backend fallback in
// resolve_llm_specs: what the form shows is what graph build / label extraction will actually use. When the
// name does not match the registry (renamed or deleted) it falls back to "(not selected)" instead of
// pretending. Since 2026-09-09 the label extraction model is preselected too: the automatic label
// extraction before a blank graph's first build uses it, so it must not show as "(not selected)".
const DEFAULT_GRAPH_LLM = "DeepSeek V4 Flash";

// Same rule as build.default_graph_llm on the backend: use the default name if present; if the registry
// has exactly one model, use it
function defaultGraphLlm() {
  if (state.llms.some(m => m.name === DEFAULT_GRAPH_LLM)) return DEFAULT_GRAPH_LLM;
  return state.llms.length === 1 ? state.llms[0].name : "";
}

// Progress against the auto-rebuild conditions: the verdict of the scheduled check (evaluate_rebuild) gives
// one percentage per condition, shown after the matching label. Time condition = elapsed days / threshold
// days; new-content ratio condition = new chunks relative to the corpus size at the last full build (the
// same measure as the percentage configured in the policy); a count-based new-content condition shows the
// count. Nothing is shown when the graph is off, the policy is empty or the condition was not computed
// (2026-09-23, user request)
function rebuildProgressParts(rc) {
  const out = { time: "", fresh: "" };
  if (!rc || !Array.isArray(rc.conditions)) return out;
  const pct = v => { const x = Math.round(v * 1000) / 10; return Number.isInteger(x) ? String(x) : x.toFixed(1); };
  for (const c of rc.conditions) {
    if (c.name === "interval" && c.interval_days > 0) out.time = t("时间进度:{0}%", pct((c.elapsed_days || 0) / c.interval_days));
    else if (c.name === "new_chunk_ratio") out.fresh = t("新增占比:{0}%", pct(c.ratio || 0));
    else if (c.name === "new_chunk_count") out.fresh = t("新增:{0} 条", c.new_chunks || 0);
  }
  return out;
}

function llmOptions(selectedName, { fallback = false } = {}) {
  let picked = selectedName;
  if (!picked && fallback) picked = defaultGraphLlm();
  return `<option value="">${t("(未选择)")}</option>` +
    state.llms.map(m => `<option value="${esc(m.name)}" ${picked === m.name ? "selected" : ""}>${esc(m.name)}</option>`).join("");
}

async function renderConfig() {
  // Freeze config rendering while the confirm dialog is open: polling must not flip the switch / form back
  // to the old state while the user decides
  if ($("#confirm-dlg").open) return;
  const kb = selectedEntry();
  const dot = $("#sel-dot"), name = $("#sel-name"), stats = $("#sel-stats"),
        toggle = $("#sel-toggle"), hint = $("#sel-hint"), fields = $("#cfg-fields");
  if (state.draft && (!kb || kb.dir !== state.draft || kb.state !== "unenrolled")) state.draft = null;
  const draft = !!state.draft;

  $("#ws-empty").style.display = kb ? "none" : "";
  $("#ws").hidden = !kb;
  dot.className = "dot " + kbDot(kb);
  name.textContent = kb ? kb.dir : t("未选择知识库");
  $("#sel-id").textContent = kb && kb.kb_id ? kb.kb_id : "";   // id: used for logs and the Qdrant collection name
  renderAdoptRow(kb, draft);
  toggle.disabled = !kb || kb.state === "directory_missing" || kb.state === "directory_linked"
    || kb.state === "deleting" || kb.state === "delete_failed";      // a half-deleted KB cannot be switched back on as it is
  toggle.checked = !!kb && (kb.state === "active" || draft);

  if (!kb) {
    stats.innerHTML = "";
    hint.style.display = "none";
    fields.disabled = true;
    $("#cfg-save").disabled = true;
    $("#cfg-parse-now").disabled = true;
    $("#cfg-reparse").disabled = true;
    $("#cfg-gsave").disabled = true;
    $("#cfg-gbuild").disabled = true;
    $("#cfg-gappend").disabled = true;
    $("#cfg-gdelete").disabled = true;
    $("#cfg-delete").disabled = true;
    $("#cfg-graph").checked = false;
    $("#cfg-graph").disabled = true;
    $("#graph-fields").disabled = true;
    state.cfgKey = null;
    return;
  }

  const active = kb.state === "active";
  if (!active) draftCache.delete(kb.dir);   // a draft is unsaved edits of an enabled KB: once it is closed, deleted or not yet enabled, it must not be filled back in when the KB is enabled next
  if (active) {
    stats.innerHTML = `<span>${t("文件")} <b>${kb.files_total || kb.dir_files || 0}</b>${!kb.files_total && kb.dir_files ? `<span class="muted">(${t("等待扫描登记")})</span>` : ""}</span>
      <span style="color:var(--green)">${t("已入库")} <b>${kb.files_indexed}</b></span>
      <span style="color:var(--yellow)">${t("待解析")} <b>${kb.files_pending}</b></span>
      <span style="color:${kb.files_failed ? "var(--red)" : "inherit"}">${t("失败")} <b>${kb.files_failed}</b></span>`;
    hint.style.display = "none";
  } else {
    stats.innerHTML = "";
    hint.textContent = draft
      ? t("先确认下方策略,点「保存配置」才会真正开启并开始解析;关闭开关可取消")
      : kb.state === "unenrolled"
        ? t("该知识库尚未开启:打开右上角开关,确认策略后开始解析")
      : kb.state === "deleting"
        ? t("正在彻底删除这个知识库,数据多的要几分钟;可以离开或刷新页面,删完这里会更新")
      : kb.state === "delete_failed"
        ? t("上次彻底删除没有完成:系统会在下次维护时接着删,也可以再点一次「删除知识库」;删完之前不能重新开启")
        : kbStateText(kb) + (kb.state === "inactive" ? t(";重新打开开关可恢复") : "");
    hint.style.display = "";
  }
  updateTabs(kb);
  syncTuneButton(kb);

  fields.disabled = !(active || draft);
  // Lower "Knowledge graph" section: the switch follows edit rights, the fields are further gated by the
  // switch itself (an unsaved flip is not overwritten by polling)
  $("#cfg-graph").disabled = !(active || draft);
  $("#graph-fields").disabled = !((active || draft) && $("#cfg-graph").checked);
  // The action row is permanent and never adds or removes controls per state: unavailable ones are greyed
  // out, so the template stays the same in every state
  $("#cfg-save").disabled = !(active || draft);
  $("#cfg-parse-now").disabled = !active;
  $("#cfg-reparse").disabled = !active;
  // Graph save button: enabled when the switch is on, or the saved state is on (to persist turning it off);
  // the build button only when the switch is on
  const savedGraphOn = !!(kb.graph_status && kb.graph_status !== "disabled");
  $("#cfg-gsave").disabled = !(active && ($("#cfg-graph").checked || savedGraphOn));
  const building = kb.graph_status === "running";
  const paused = kb.graph_status === "stopped";
  const halted = paused || kb.graph_status === "failed";
  $("#cfg-gbuild").disabled = !(active && $("#cfg-graph").checked) || building;   // no build without the graph turned on
  // Appending needs a completed graph; a KB whose last append stopped midway (paused or failed) while a built
  // version is still live can append as well, a KB that was never built can only do a full build
  const liveGraph = kb.graph_status === "ok" || (halted && !!(kb.graph_build || {}).active_graph_version);
  $("#cfg-gappend").disabled = !(active && $("#cfg-graph").checked && liveGraph) || building;
  // For a KB stopped midway (paused or failed), "build" may mean "carry on" -- but only when the cache can
  // really be resumed. The cache key includes model and messages: labels re-extracted, build model
  // switched or chunking parameters changed and the old cache is entirely missed; clicking then means a
  // full re-run of several hours, which must not be called "Resume". The server compares the fingerprint
  // taken at build start with the current config in _paused_cache_reuse; completed phases
  // (graph_build_phases) are skipped wholesale when resuming.
  const gb = kb.graph_build || {};
  // What stopped midway was an append and a built version is still live: carrying on means appending once
  // more (the same rule as trigger_graph_build on the server)
  const resumeAppend = halted && gb.build_kind === "append" && !!gb.active_graph_version;
  const canResume = halted && (resumeAppend || gb.cache_reusable === true);
  $("#cfg-gbuild").textContent = t(resumeAppend ? "继续并入" : canResume ? "继续建图" : "立即/重新建图");
  const parts = savedGraphOn ? rebuildProgressParts(kb.rebuild_check) : { time: "", fresh: "" };
  for (const [id, text] of [["#cfg-ri-progress", parts.time], ["#cfg-rp-progress", parts.fresh]]) {
    const el = $(id);
    if (el) { el.textContent = text; el.hidden = !text; }
  }
  const gnote = $("#cfg-gnote");
  if (gnote) {
    const phases = (gb.phases_done || []).length ? t("已完成 {0},续跑跳过", gb.phases_done.map(ph => t(ph)).join(t("、"))) : "";
    const cache = gb.cache_entries ? t("缓存 {0} 条可续用", gb.cache_entries) : "";
    const text = canResume
      ? [phases, cache].filter(Boolean).join(" · ")
      : (halted && gb.cache_stale ? t("{0},缓存不可续用,这次是从头建", t(gb.cache_stale)) : "");
    gnote.textContent = text;
    gnote.style.display = text ? "inline-block" : "none";   // .note is hidden by default in the stylesheet; clearing the value would not show it
  }
  $("#cfg-gpause").disabled = !(active && building);
  // Allow deletion whenever artifacts exist, even with the switch already off (turned off mid-build, or the
  // last delete failed, both end up here). Looking only at graph_status would turn an existing graph into
  // an orphan that cannot be cleaned up.
  $("#cfg-gdelete").disabled = !(active && (kb.graph_artifacts
    || (kb.graph_status && kb.graph_status !== "disabled")));
  // Deletable only when enrolled (including the deactivated retention period and a delete that did not finish);
  // unenrolled / draft and a KB being deleted are greyed out (like the other buttons, not hidden)
  $("#cfg-delete").disabled = !(kb.kb_id && !draft) || kb.state === "deleting";

  if (state.toggling) return;   // switch request in flight: do not let polling flick it back for a moment
  const key = `${kb.dir}|${draft ? "draft" : kb.state}`;
  if (state.cfgKey !== key) {
    state.cfgKey = key;
    // When the same enabled KB re-fetches its config the form still holds its own content; for another KB or
    // state the form only counts once it is filled below
    if (!(active && formBase && formBase.dir === kb.dir)) formBase = null;
    $("#cfg-note").style.display = "none";
    if (!active) refreshCorpusHint(null);   // the previous KB's "N documents, M chunks" must not linger under this KB's name
    if (active) {
      try {
        const [cfg, llms] = await Promise.all([api(`/kbs/${encodeURIComponent(kb.kb_id)}/config`), api("/llms")]);
        // When two KBs are clicked in quick succession the earlier request may arrive later. cfgKey is
        // already another KB by then, and filling the form would show KB A's policy under KB B's name --
        // one click on save would then write it to the wrong KB.
        if (state.cfgKey !== key) return;
        state.llms = llms;
        const c = cfg.effective, lim = cfg.limits, g = cfg.config.graph_llm || {};
        setChunkHint(lim);
        $("#cfg-max").value = c.max_tokens;
        $("#cfg-ov").value = c.overlap_tokens;
        setOverlapHint();   // must come after max_tokens is filled in: the bound is computed from it
        $("#cfg-graph").checked = !!c.graph_enabled;
        $("#cfg-prompt").value = cfg.config.vlm_prompt || "";
        $("#cfg-guc").value = String(c.graph_unit_chunks ?? 3);
        $("#cfg-ggl").value = String(c.graph_max_gleanings ?? 1);
        // All three model slots preselect the default model when empty
        $("#cfg-ge").innerHTML = llmOptions(g.extract, { fallback: true });
        $("#cfg-gs").innerHTML = llmOptions(g.summarize, { fallback: true });
        $("#cfg-gt").innerHTML = llmOptions(g.tune, { fallback: true });
        // A version just extracted but not yet saved must not be wiped by an unrelated re-fetch. Turning the
        // graph switch off or switching panels both trigger a re-fetch -- that is how kb_004 lost its
        // labels on 2026-08-24: the refresh from the switch reset dirty to false, save was clicked right
        // after, the selection was not sent, none of the 21 extracted labels took effect, and the UI said
        // nothing.
        const keepPending = state.graphSchema.dirty && state.graphSchema.kbId === kb.kb_id;
        setSchemaView(cfg.schema, keepPending
          ? { note: state.graphSchema.note, dirty: true,
              kbId: kb.kb_id, active: state.graphSchema.active }
          : { kbId: kb.kb_id });
        $("#cfg-gss").value = c.graph_tune_sample_size ?? 8;
        const ri = splitInterval(c.graph_rebuild_interval);
        $("#cfg-ri-n").value = ri.n;
        $("#cfg-ri-u").value = ri.u;
        const nc = splitNewChunk(c.graph_rebuild_new_chunk_pct, c.graph_rebuild_new_chunk_count);
        $("#cfg-rp-n").value = nc.n;
        $("#cfg-rp-u").value = nc.u;
        $("#cfg-ro").value = (c.graph_rebuild_operator || "or").toLowerCase();
        $("#cfg-gaa").checked = c.graph_auto_append !== false;
        $("#graph-fields").disabled = !c.graph_enabled;
        formBase = { dir: kb.dir, ...parseFormValues() };
        if (restoreDraft(kb.dir)) {   // unsaved edits made earlier on this KB are still there on switching back
          $("#cfg-note").textContent = t("有未保存的修改");
          $("#cfg-note").style.display = "inline-block";   // .note is hidden by default in the stylesheet
        }
        refreshCorpusHint(kb);  // no await: a slow count must not block the form
        refreshTab();           // draw once more when the form is ready: the graph tab's mode label reads the form
      } catch (e) {
        toast(t("配置载入失败: ") + t(e.message));
        state.cfgKey = null;   // let the next round retry instead of treating the previous KB's values as this one's
      }
    } else if (draft) {
      // Draft: prefill from the directory presets (if any) + global defaults, then wait for the user to
      // confirm
      const dd = kb.dir_defaults || {};
      let lim = null, llms = null;
      try { [lim, llms] = await Promise.all([api("/limits"), api("/llms")]); } catch (e) { /* the range hint stays empty, the form is filled anyway */ }
      if (state.cfgKey !== key) return;   // the KB was switched while waiting: do not fill this directory's presets into another KB's form
      if (llms) state.llms = llms;
      if (lim) setChunkHint(lim); else $("#cfg-max-hint").textContent = "";
      $("#cfg-max").value = dd.max_tokens ?? 400;
      $("#cfg-ov").value = dd.overlap_tokens ?? 80;
      setOverlapHint();
      $("#cfg-graph").checked = false;   // enabling a KB does not enable the graph: the knowledge graph must be switched on explicitly
      $("#cfg-prompt").value = "";
      $("#cfg-guc").value = String(dd.graph_unit_chunks ?? 3);
      $("#cfg-ggl").value = String(dd.graph_max_gleanings ?? 1);
      $("#cfg-ge").innerHTML = llmOptions(null, { fallback: true });
      $("#cfg-gs").innerHTML = llmOptions(null, { fallback: true });
      $("#cfg-gt").innerHTML = llmOptions(null, { fallback: true });
      // A draft KB has no indexed chunks yet, so no labels can be extracted: empty falls back to the global
      // default
      setSchemaView(null);
      $("#cfg-gss").value = dd.graph_tune_sample_size ?? 8;
      const ri = splitInterval(dd.graph_rebuild_interval);
      $("#cfg-ri-n").value = ri.n;
      $("#cfg-ri-u").value = ri.u;
      const ncd = splitNewChunk(dd.graph_rebuild_new_chunk_pct, dd.graph_rebuild_new_chunk_count);
      $("#cfg-rp-n").value = ncd.n;
      $("#cfg-rp-u").value = ncd.u;
      $("#cfg-ro").value = (dd.graph_rebuild_operator || "or").toLowerCase();
      $("#cfg-gaa").checked = dd.graph_auto_append !== false;
      $("#graph-fields").disabled = true;
    } else {
      $("#cfg-max").value = 400; $("#cfg-ov").value = 80;
      $("#cfg-graph").checked = false; $("#cfg-prompt").value = "";
      $("#cfg-ri-n").value = ""; $("#cfg-ri-u").value = "d";
      $("#cfg-rp-n").value = ""; $("#cfg-rp-u").value = "p"; $("#cfg-ro").value = "or";
      setSchemaView(null); $("#cfg-gss").value = 8;
      $("#cfg-guc").value = "3"; $("#cfg-ggl").value = "1";
      $("#graph-fields").disabled = true;
    }
  }
}

// Unenrolled directory + a KB whose directory vanished / was closed: offer "this is that KB renamed". The
// scan recognizes a rename automatically when the content matches; this is the manual fallback (e.g. a
// rename made after the KB was closed).
function renderAdoptRow(kb, draft) {
  const row = $("#sel-adopt");
  if (!row) return;
  const orphans = (state.overview?.kbs || []).filter(k => k.kb_id && (k.state === "inactive" || k.state === "directory_missing"));
  if (!kb || kb.state !== "unenrolled" || draft || !orphans.length) { row.style.display = "none"; return; }
  const sel = $("#adopt-from");
  const cur = sel.value;
  if (setHtml(sel, orphans.map(k => `<option value="${esc(k.kb_id)}">${esc(k.dir)}(${esc(k.kb_id)})</option>`).join(""))
      && [...sel.options].some(o => o.value === cur)) sel.value = cur;
  row.style.display = "";
}

$("#adopt-go").addEventListener("click", async ev => {
  const kb = selectedEntry();
  const from = $("#adopt-from").value;
  const old = (state.overview?.kbs || []).find(k => k.kb_id === from);
  if (!kb || kb.state !== "unenrolled" || !old) return;
  const ok = await confirmDialog(
    t("把「{0}」认作原来的「{1}」({2})改的名?\n\n沿用它的编号、已入库内容、图谱与缓存;内容没变的文件只刷新路径,不重新解析。", kb.dir, old.dir, from),
    t("沿用原库"));
  if (!ok) return;
  ev.target.disabled = true;
  try {
    const r = await api(`/kbs/${encodeURIComponent(from)}/adopt`, { method: "POST", body: { dir: kb.dir } });
    toast(t("已沿用 {0}:{1}/{2} 个文件对得上", from, r.matched, r.total) + (r.looks_like_rename ? "" : t("(相似度不高,请确认没认错)")));
    state.cfgKey = null;
  } catch (e) { toast(t("沿用失败: ") + t(e.message)); }
  ev.target.disabled = false;
  await refreshOverview(true);
});

$("#cfg-graph").addEventListener("change", async ev => {
  const kb = selectedEntry();
  if (kb && kb.state === "active" && (kb.jobs_active || 0) > 0) {
    // Parse jobs in progress: the graph section is not greyed out, but every graph action shows a notice
    // and reverts
    ev.target.checked = !ev.target.checked;
    await noticeDialog(PARSE_BUSY_MSG);
    return;
  }
  const on = ev.target.checked;
  const activeKb = kb && kb.state === "active";
  const savedOn = !!(kb && kb.graph_status && kb.graph_status !== "disabled");
  $("#graph-fields").disabled = !on;
  $("#cfg-gbuild").disabled = !(activeKb && on);
  $("#cfg-gsave").disabled = !(activeKb && (on || savedOn));
  // Only compute when switching on. When switching off leave the hint alone for now -- a confirm dialog
  // follows, and if it is cancelled nothing happened, so the hint must not be cleared; it is cleared later
  // once the switch-off really succeeded.
  if (on) refreshCorpusHint(kb);

  // Turning on and off are deliberately asymmetric:
  //
  // "On" is only staged -- it is the entry point of "I want to configure a graph"; persisting it with no
  //   model selected is meaningless, so it is submitted together with the whole form via "Save".
  // "Off" persists immediately -- there is nothing to configure when turning off, and a switch that flips
  //   down and bounces back on refresh is a pure trap (the same-styled "Enable knowledge base" switch right
  //   above takes effect immediately).
  //
  // PUT only the graph_enabled key: the backend merges, other unsaved edits in the form are unaffected.
  // Turning off deletes no data -- the built graph and aliases stay as they are, retrieval is not
  // interrupted, it just no longer rebuilds automatically; to clear data use "Delete knowledge graph".
  if (on || !activeKb) return;

  // Needs a second confirmation, like the "Enable knowledge base" switch. Without a built graph the second
  // sentence is left out -- saying "data is kept" about an empty graph only confuses, and "turning off
  // merely saves the switch as off" carries no information.
  //
  // The criterion looks only at graph_artifacts (are there graph_builds rows). It must not also look at
  // graph_status !== "disabled": status pending is exactly "switch on, never built once", where artifacts
  // is necessarily false, and adding that test would say "data is kept" about an empty graph.
  const hasGraph = !!kb.graph_artifacts;
  const building = kb.graph_status === "running";
  // Turning off mid-build = stop, not discard: progress and the LLM cache are kept, and the build resumes
  // from where it stopped after re-enabling.
  const note = building ? t("\n\n正在建图,关闭会停止当前任务;已跑完的部分留在缓存里,下次建图从断点继续")
    : hasGraph ? t("\n\n已建好的图谱保留,只是不再自动重建") : "";
  const ok = await confirmDialog(t("关闭「{0}」的知识图谱?", kb.dir) + note, t("确认关闭"));
  if (!ok) {
    ev.target.checked = true;   // cancel = nothing happened, flip the switch back
    renderConfig();             // it recomputes the field area and the two buttons' disabled state from checked
    return;
  }
  ev.target.disabled = true;
  try {
    await api(`/kbs/${encodeURIComponent(kb.kb_id)}/config`, {
      method: "PUT", body: { graph_enabled: false },
    });
    toast(building ? t("已关闭知识图谱并停止建图;进度与缓存保留,下次建图从断点继续")
      : t("已关闭知识图谱;已建好的图谱数据保留,只是不再自动重建"));
    state.cfgKey = null;   // force the next round to re-fetch the config instead of refilling old values
    refreshCorpusHint(null);
  } catch (e) {
    // If persisting failed (e.g. refused by the backend during a build), flip the switch back so the UI
    // does not lie
    ev.target.checked = true;
    $("#graph-fields").disabled = false;
    $("#cfg-gbuild").disabled = !activeKb;
    $("#cfg-gsave").disabled = !activeKb;
    toast(t("关闭失败: ") + t(e.message));
  } finally {
    ev.target.disabled = false;
  }
  await refreshOverview(true);
});

$("#sel-toggle").addEventListener("change", async ev => {
  state.toggling = true;
  const kb = selectedEntry();
  if (!kb) { ev.target.checked = false; state.toggling = false; return; }
  if (ev.target.checked && kb.state === "unenrolled") {
    // Not enrolled immediately: enter the draft state, "Save" is the real trigger
    state.draft = kb.dir;
    state.cfgKey = null;
    state.toggling = false;
    renderConfig();
    return;
  }
  if (!ev.target.checked && state.draft === kb.dir) {
    // Switching off in draft state = cancel, nothing happened
    state.draft = null;
    state.cfgKey = null;
    state.toggling = false;
    renderConfig();
    return;
  }
  ev.target.disabled = true;
  try {
    if (ev.target.checked) {
      // Only restoring within the retention period reaches here: restored immediately, no re-parse, no
      // config needed first
      const r = await api("/enroll", { method: "POST", body: { dir: kb.dir } });
      toast(r.outcome === "reactivated"
        ? t("「{0}」已恢复(保留期内,不重新解析)", kb.dir)
        : t("「{0}」已开启,开始解析", kb.dir));
    } else {
      const days = state.overview?.retention_days ?? 7;
      const busy = kb.jobs_active > 0 || kb.files_pending > 0;
      const ok = await confirmDialog(
        t("关闭知识库 {0} 天后会自动删除当前解析结果,确认关闭?", days)
        + (busy ? t("\n\n正在解析的任务会停止,已入库的内容保留;重新开启时恢复数据并接着解析剩下的文件") : ""),
        t("确认关闭"));
      if (!ok) { ev.target.checked = true; ev.target.disabled = false; state.toggling = false; renderConfig(); return; }
      const r = await api(`/kbs/${encodeURIComponent(kb.kb_id)}/unenroll`, { method: "POST" });
      toast(t("已关闭,{0} 个文件进入失活队列", r.files_delete_queued));
    }
  } catch (e) { toast(t("操作失败: ") + t(e.message)); }
  ev.target.disabled = false;
  state.toggling = false;
  await refreshOverview(true);
});

function requireInt(sel, label) {
  const raw = $(sel).value.trim();
  const n = Number(raw);
  if (!raw || !Number.isInteger(n)) throw new Error(t("{0} 需要填一个整数", label));
  return n;
}

// Switching KBs no longer makes half-finished edits vanish: drafts are kept in memory per directory and
// restored on switching back. Not persisted, so the established "reload returns to the pre-save state"
// semantics are unchanged.
const draftCache = new Map();
// Which KB's saved config the form is currently filled from, and the values filled in. Only edits made against it
// count as a draft: the form of a KB that is not enabled holds placeholder values, and until the config arrives it
// holds the previous KB's values; stashing those and filling them back in would cover the real config with other
// values, and the next click on save would write them.
let formBase = null;

function parseFormValues() {
  return { prompt: $("#cfg-prompt").value, max: $("#cfg-max").value, ov: $("#cfg-ov").value };
}

function stashDraft(dir) {
  if (!dir || !formBase || formBase.dir !== dir) return;
  const now = parseFormValues();
  if (now.prompt === formBase.prompt && now.max === formBase.max && now.ov === formBase.ov) draftCache.delete(dir);
  else draftCache.set(dir, now);
}

function restoreDraft(dir) {
  const d = draftCache.get(dir);
  if (!d) return false;
  $("#cfg-prompt").value = d.prompt;
  $("#cfg-max").value = d.max;
  $("#cfg-ov").value = d.ov;
  return true;
}

function collectParseConfig() {
  // parseInt used to be used here: empty values / decimals / "1e3" became NaN or were truncated, serialized
  // as null and treated by the backend as "clear the key, back to default", while the form did not refill
  // -- the user never learned the value had been replaced.
  return {
    max_tokens: requireInt("#cfg-max", "chunk max_tokens"),
    overlap_tokens: requireInt("#cfg-ov", "chunk overlap_tokens"),
    vlm_prompt: $("#cfg-prompt").value.trim() || null,
  };
}

function splitInterval(v) {
  // Stored format "7d/2w/1m" (Chinese units accepted too) -> {n, u}; empty / unrecognized -> empty, unit
  // defaults to days
  const m = String(v || "").trim().toLowerCase().match(/^(\d+)\s*([dwm日周月]?)$/);
  if (!m) return { n: "", u: "d" };
  const u = { "": "d", "d": "d", "日": "d", "w": "w", "周": "w", "m": "m", "月": "m" }[m[2]] || "d";
  return { n: m[1], u };
}

// The new-content condition is stored as two mutually exclusive keys: _pct ("20%") or _count (300).
// A bare decimal is a ratio (0.2 = 20%), a bare integer is a percentage as parse_ratio reads it (3 = 3%)
// -- this conversion must match the backend's to the letter, or the number the form shows and the
// threshold actually in effect would differ.
function splitNewChunk(pct, count) {
  const c = String(count ?? "").trim();
  if (c) return { n: c, u: "c" };
  const m = String(pct ?? "").trim().match(/^(\d+(?:\.\d+)?)\s*(%?)$/);
  if (!m) return { n: "", u: "p" };
  let r = Number(m[1]);
  if (m[2] !== "%" && r <= 1) r = r * 100;
  return { n: String(Math.round(r)), u: "p" };
}

function collectNewChunk() {
  const raw = $("#cfg-rp-n").value.trim();
  if (!raw) return { pct: null, count: null };
  let n = Number(raw);
  if (!Number.isInteger(n) || n < 1) {
    throw new Error(t("自动重建·新增内容条件需为不小于 1 的整数"));
  }
  if ($("#cfg-rp-u").value === "c") return { pct: null, count: n };
  // Percentages cap at 100: the denominator is the number of still-alive baseline chunks, doubling is
  // 100%, and a higher threshold only looks stricter than it is. Write the clamped value back into the
  // input so what you see is what gets saved.
  if (n > 100) { n = 100; $("#cfg-rp-n").value = "100"; }
  return { pct: `${n}%`, count: null };
}

function collectInterval() {
  const raw = $("#cfg-ri-n").value.trim();
  if (!raw) return null;
  const n = Number(raw);
  if (!Number.isInteger(n) || n < 1 || n > 30) {
    throw new Error(t("自动重建·时间条件需为 1–30 的整数"));
  }
  return `${n}${$("#cfg-ri-u").value}`;
}

function collectGraphConfig() {
  return {
    graph_enabled: $("#cfg-graph").checked,
    graph_unit_chunks: requireInt("#cfg-guc", t("每单元合并切片数")),
    graph_max_gleanings: requireInt("#cfg-ggl", t("补漏轮数")),
    graph_rebuild_interval: collectInterval(),
    ...(() => { const nc = collectNewChunk();
      return { graph_rebuild_new_chunk_pct: nc.pct,
               graph_rebuild_new_chunk_count: nc.count }; })(),
    graph_rebuild_operator: $("#cfg-ro").value,
    graph_auto_append: $("#cfg-gaa").checked,
    graph_llm: {
      extract: $("#cfg-ge").value || null,
      summarize: $("#cfg-gs").value || null,
      tune: $("#cfg-gt").value || null,
    },
    graph_tune_sample_size: requireInt("#cfg-gss", t("采样数量")),
    // Entity labels and the output language are **never sent by the console**: they are derived values,
    // expanded server-side from the selected label version. The console only expresses "which version".
    //
    // Previously these two values were sent unconditionally from the frontend's in-memory state, and any
    // path that emptied it (switching to a draft KB, a failed config request, an early return in a race)
    // followed by a save overwrote the labels stored on the server with null -- for a field the user never
    // touched. That is how a label table that took 2 h 25 min to extract was lost on 2026-08-24. That path
    // is now sealed: without a version id nothing is changed.
    ...(state.graphSchema.dirty && state.graphSchema.active
      ? { graph_schema_active: state.graphSchema.active }
      : {}),
  };
}

// After the graph section is persisted: clear the "unsaved" state
function markGraphConfigSaved() {
  state.graphSchema.dirty = false;
  renderGraphSchema();
}

$("#cfg-save").addEventListener("click", async () => {
  const kb = selectedEntry();
  if (!kb) return;
  const note = $("#cfg-note");
  if (state.draft === kb.dir && kb.state === "unenrolled") {
    // Draft confirmation: enroll + write the config (including the graph section) + start parsing, all in
    // one go
    try {
      await api("/enroll", { method: "POST", body: { dir: kb.dir, config: { ...collectParseConfig(), ...collectGraphConfig() } } });
      state.draft = null;
      state.cfgKey = null;
      // Freshly enabled KB: the next render switches to the files tab, where parsing can be seen
      // progressing
      state.tabByKb[kb.dir] = "files";
      remember("tabs", state.tabByKb);
      toast(t("「{0}」已开启,按所配策略开始解析", kb.dir));
    } catch (e) { toast(t("开启失败: ") + t(e.message)); return; }
    await refreshOverview(true);
    return;
  }
  if (kb.state !== "active") return;
  try {
    await api(`/kbs/${encodeURIComponent(kb.kb_id)}/config`, { method: "PUT", body: collectParseConfig() });
    toast(t("配置已保存,随自动扫描生效"));
    draftCache.delete(kb.dir);
    formBase = { dir: kb.dir, ...parseFormValues() };   // what the form holds are now the saved values
    note.style.display = "none";
  } catch (e) { toast(t("保存失败: ") + t(e.message)); }
  // No forced refill: the form already holds the values just saved, and a refill would only wipe the other
  // section's unsaved edits
  await refreshOverview();
});

$("#cfg-parse-now").addEventListener("click", async () => {
  const kb = selectedEntry();
  if (!kb || kb.state !== "active") return;
  try {
    await api(`/kbs/${encodeURIComponent(kb.kb_id)}/parse_now`, { method: "POST", body: {} });
    toast(t("已触发扫描:空库全量解析,已有内容则处理增量文件"));
    setTab("files");
  } catch (e) { toast(t("触发失败: ") + t(e.message)); }
  await refreshOverview();   // saves nothing, so it must not roll back unsaved edits either
});

// Changing the version only changes a pending selection: labels and language follow immediately (preview),
// it takes effect on save.
$("#cfg-gver").addEventListener("change", ev => {
  state.graphSchema.active = String(ev.target.value || "");
  state.graphSchema.dirty = true;
  state.graphSchema.note = "";
  renderGraphSchema();
});

$("#cfg-gsave").addEventListener("click", async () => {
  const kb = selectedEntry();
  if (!kb || kb.state !== "active") return;
  if ((kb.jobs_active || 0) > 0) { await noticeDialog(PARSE_BUSY_MSG); return; }
  try {
    await api(`/kbs/${encodeURIComponent(kb.kb_id)}/config`, { method: "PUT", body: collectGraphConfig() });
    markGraphConfigSaved();            // persisted, clear the "unsaved" hint
    toast(t("图谱配置已保存;建图由自动重建条件或「立即/重新建图」触发"));
  } catch (e) { toast(t("保存失败: ") + t(e.message)); }
  await refreshOverview();
});

let buildStarting = false;          // no second click while the request is in flight (final review F03: during preparation there is no running record yet, so the backend lock alone cannot block a duplicate submit)
$("#cfg-gbuild").addEventListener("click", async () => {
  if (buildStarting) return;
  const kb = selectedEntry();
  if (!kb || kb.state !== "active") return;
  if ((kb.jobs_active || 0) > 0) { await noticeDialog(PARSE_BUSY_MSG); return; }
  buildStarting = true;
  $("#cfg-gbuild").disabled = true;
  try {
    // The button looks at the switch and model selection on the page (possibly unsaved); the backend looks
    // at the saved config. Submit this section's draft first, then trigger, to avoid "button clickable but
    // the backend says the graph is off" or "building with the old model".
    await api(`/kbs/${encodeURIComponent(kb.kb_id)}/config`, { method: "PUT", body: collectGraphConfig() });
    markGraphConfigSaved();
    await api(`/kbs/${encodeURIComponent(kb.kb_id)}/graph_build`, { method: "POST", body: {} });
    toast(t("图谱配置已保存,建图已启动"));
  } catch (e) { toast(t("建图启动失败: ") + t(e.message)); }
  finally { buildStarting = false; }
  await refreshOverview();
});

$("#cfg-gappend").addEventListener("click", async () => {
  const kb = selectedEntry();
  if (!kb || kb.state !== "active") return;
  if ((kb.jobs_active || 0) > 0) { await noticeDialog(PARSE_BUSY_MSG); return; }
  try {
    // Same as "Build / rebuild now": submit this section's draft first, the backend decides from the saved
    // config whether an append is possible
    await api(`/kbs/${encodeURIComponent(kb.kb_id)}/config`, { method: "PUT", body: collectGraphConfig() });
    markGraphConfigSaved();
    const r = await api(`/kbs/${encodeURIComponent(kb.kb_id)}/graph_append`, { method: "POST", body: {} });
    toast(t("并入已启动:") + deltaText(r.delta || {}));
  } catch (e) { toast(t("并入启动失败: ") + t(e.message)); }
  await refreshOverview();
});

$("#cfg-gpause").addEventListener("click", async ev => {
  const kb = selectedEntry();
  if (!kb || kb.state !== "active") return;
  const ok = await confirmDialog(
    t("暂停「{0}」的建图?\n\n进度与缓存保留。暂停期间改模型、标签、谓词、语言或合并切片数会使缓存失效。", kb.dir),
    t("暂停建图"));
  if (!ok) return;
  ev.target.disabled = true;
  try {
    const r = await api(`/kbs/${encodeURIComponent(kb.kb_id)}/graph_pause`, { method: "POST" });
    toast(t("已暂停 · 缓存 {0} 条可复用", r.cache_entries || 0));
  } catch (e) { toast(t("暂停失败: ") + t(e.message)); }
  await refreshOverview(true);
});

$("#cfg-gvdel").addEventListener("click", async () => {
  const kb = selectedEntry();
  const chosen = activeVersion();
  if (!kb || !chosen) return;
  const ok = await confirmDialog(
    t("删除标签版本「{0}」?\n\n删掉就没有了,只能重新抽一次;本库当前生效的标签不受影响。", versionLabel(chosen)), t("确认删除"));
  if (!ok) return;
  try {
    const r = await api(
      `/kbs/${encodeURIComponent(kb.kb_id)}/graph_schema/${encodeURIComponent(chosen.id)}`,
      { method: "DELETE" });
    // If the selected entry is gone, setSchemaView falls back to the newest version
    setSchemaView(r.schema, { note: state.graphSchema.note, dirty: state.graphSchema.dirty,
                              kbId: kb.kb_id, active: state.graphSchema.active });
    toast(t("已删除该标签版本"));
  } catch (e) {
    toast(t("删除失败: ") + t(e.message));
  }
});

$("#cfg-gtune").addEventListener("click", async ev => {
  const kb = selectedEntry();
  if (!kb || kb.state !== "active") return;
  // While parsing runs, ready documents are still landing; sampling would see half a corpus. Same gate as
  // the build.
  if ((kb.jobs_active || 0) > 0) { await noticeDialog(PARSE_BUSY_MSG); return; }
  // The button's "extracting" state is no longer just text changed by this click: tuneInFlight and the
  // schema_suggest mark from the overview decide together (syncTuneButton), so switching KBs / tabs or
  // reloading shows the truth
  tuneInFlight = kb.kb_id;
  syncTuneButton(kb);
  try {
    // Submit this section's draft first: this step uses the **saved** label extraction model and sample
    // size; without saving it would run with the old model or report "no model selected". Same reasoning
    // as cfg-gbuild.
    await api(`/kbs/${encodeURIComponent(kb.kb_id)}/config`, { method: "PUT", body: collectGraphConfig() });
    markGraphConfigSaved();
    const r = await api(`/kbs/${encodeURIComponent(kb.kb_id)}/graph_schema`, { method: "POST", body: {} });
    const nTypes = (r.entity_types || []).length, nPreds = (r.predicates || []).length;
    if (selectedEntry()?.kb_id !== kb.kb_id) {
      // Switched to another KB during extraction: the result is already in the server's version ring, and
      // KB A's labels must not be filled into KB B's form
      toast(t("「{0}」已抽出 {1} 个实体标签、{2} 个谓词,切回该库查看并保存", kb.dir, nTypes, nPreds));
      return;
    }
    const domain = domainSummary(r.domain, 60);
    // Spell out both "chunks" and "documents": a chunk count alone is easily read as a document count
    // Sampling details (health check D1): documents covered, boilerplate chunks excluded, whether
    // representatives were picked by vector
    const s = r.sample || {};
    const note = t("采样 {0}/{1} 片段(覆盖 {2}/{3} 份文档", r.sampled, r.chunks_total, s.documents_covered ?? "?", r.documents)
      + (s.excluded_boilerplate ? t(",排除样板 {0} 片", s.excluded_boilerplate) : "")
      + (s.method === "vectors" ? t(",按向量挑代表") : "") + ")"
      + (r.truncated ? t("（触及 token 预算）") : "")
      + (domain ? t(" · 领域:{0}", domain) : "");
    // The version is already stored in the ring by the backend (it will not be lost by an unrelated save),
    // but is **not in effect yet** -- that still needs you to select it and save.
    setSchemaView(r.schema, { note, dirty: true, kbId: kb.kb_id });
    toast(t("已抽出 {0} 个实体标签、{1} 个谓词,确认后点「保存配置」", nTypes, nPreds));
  } catch (e) {
    toast(t("抽取失败: ") + t(e.message));
  } finally {
    tuneInFlight = null;
    syncTuneButton(selectedEntry());
  }
});

// State of the "Extract labels" button: when an extraction is running for this KB (your own click,
// tuneInFlight, or the server-side schema_suggest mark in the overview -- console-triggered and automatic
// pre-build ones alike) it shows "extracting" and is disabled, the same after switching KBs / tabs or
// reloading; when the mark disappears the extraction is done, so the config is re-fetched once so the new
// version enters the dropdown (your own click fills it back in the click handler, no re-fetch).
let tuneInFlight = null;
const suggestSeen = {};
function syncTuneButton(kb) {
  const btn = document.getElementById("cfg-gtune");
  if (!btn) return;
  const active = !!kb && kb.state === "active";
  const remote = active && kb.schema_suggest ? kb.schema_suggest : null;
  const running = active && (remote != null || tuneInFlight === kb.kb_id);
  btn.disabled = running;
  btn.textContent = running ? t("抽取中…") : t("立即/重新抽取标签");     // no elapsed time shown (2026-09-12, user decision)
  if (!kb || !kb.kb_id) return;
  const key = String(kb.kb_id);
  if (remote) suggestSeen[key] = remote.started_at;
  else if (suggestSeen[key] != null && tuneInFlight !== kb.kb_id) {
    delete suggestSeen[key];
    state.cfgKey = null;          // extraction done: the next renderConfig re-fetches the config so the new version shows up under "Label version"
  }
}

$("#cfg-gdelete").addEventListener("click", async ev => {
  const kb = selectedEntry();
  if (!kb || !kb.kb_id) return;
  const ok = await confirmDialog(
    t("删除「{0}」的知识图谱?\n\n仅删除图谱数据(图谱向量、图谱库、建图缓存与记录),知识库与解析结果不受影响", kb.dir)
    + (kb.graph_status === "running" ? t("\n\n正在建图:删除会终止任务并清空全部进度,不保留缓存") : ""),
    t("删除图谱"));
  if (!ok) return;
  ev.target.disabled = true;
  try {
    await api(`/kbs/${encodeURIComponent(kb.kb_id)}/graph`, { method: "DELETE" });
    toast(t("「{0}」的知识图谱已删除", kb.dir));
    state.cfgKey = null;   // re-fetch the config: graph_enabled is cleared, the switch goes back to grey
  } catch (e) { toast(t("删除失败: ") + t(e.message)); }
  await refreshOverview(true);
});

$("#cfg-reparse").addEventListener("click", async () => {
  const kb = selectedEntry();
  if (!kb || kb.state !== "active") return;
  if (!(await confirmDialog(t("确认对「{0}」重新解析?", kb.dir), t("确认重解析")))) return;
  try {
    const r = await api(`/kbs/${encodeURIComponent(kb.kb_id)}/reparse`, { method: "POST", body: {} });
    toast(t("已重排 {0} 个文件", r.requeued));
    setTab("files");
  } catch (e) { toast(t("失败: ") + t(e.message)); }
});

// A permanent delete runs synchronously on the server and takes minutes for a large KB (the graph store deletes
// nodes in batches). pendingDeletes: delete requests this page sent that have not come back yet; deletingSeen: KBs
// still being deleted in the previous overview round (kb_id -> directory name). One no longer being deleted in this
// round has an outcome -- requests this page sent itself are reported by the click handler, those sent elsewhere
// (the page was reloaded meanwhile, another window clicked) are reported here, otherwise minutes of waiting would
// end without a word
const pendingDeletes = new Set();
const deletingSeen = {};
function noteDeleteOutcomes(kbs) {
  for (const k of kbs) {
    if (k.kb_id && pendingDeletes.has(k.kb_id)) k.state = "deleting";     // also the moment after the request is sent, before the server has registered it
    if (k.kb_id && k.state === "deleting") deletingSeen[k.kb_id] = k.dir;
  }
  for (const [id, dir] of Object.entries(deletingSeen)) {
    const now = kbs.find(k => k.kb_id === id);
    if (now && now.state === "deleting") continue;
    delete deletingSeen[id];
    if (pendingDeletes.has(id)) continue;
    toast(now && now.state === "delete_failed" ? t("「{0}」删除未完成,系统会在下次维护时接着删", dir) : t("「{0}」已彻底删除", dir));
  }
}

$("#cfg-delete").addEventListener("click", async () => {
  const kb = selectedEntry();
  if (!kb || !kb.kb_id || pendingDeletes.has(kb.kb_id)) return;
  const busy = kb.jobs_active > 0 || kb.graph_status === "running";
  const ok = await confirmDialog(
    t("彻底删除「{0}」?\n\n此操作立即生效,不可恢复", kb.dir)
    + (busy ? t("\n\n正在运行的解析/建图任务会被终止,已入库内容与缓存一并清除") : ""),
    t("彻底删除"));
  if (!ok) return;
  pendingDeletes.add(kb.kb_id);
  toast(t("正在彻底删除「{0}」,数据多的要几分钟", kb.dir));
  kb.state = "deleting";          // do not wait for the next poll: the sidebar and this page show the delete at once
  renderKbNav();
  renderConfig();
  try {
    await api(`/kbs/${encodeURIComponent(kb.kb_id)}`, { method: "DELETE" });
    toast(t("「{0}」已彻底删除", kb.dir));
  } catch (e) { toast(t("删除失败: ") + t(e.message)); }
  pendingDeletes.delete(kb.kb_id);
  delete deletingSeen[kb.kb_id];
  // The directory returns to the unenrolled state and the switch turns off with it. During the minutes of waiting
  // the user may have moved on to edit another KB: refill the form only if this KB is still the one selected
  await refreshOverview(state.selected === kb.dir);
});

/* ── files tab: table + filter / sort / pagination ─────────────── */
const FILES_PAGE = 50;
const FILE_SORTS = {
  path: (a, b) => a.rel_path.localeCompare(b.rel_path, "zh-CN"),
  size: (a, b) => (a.size || 0) - (b.size || 0),
  chunks: (a, b) => (a.chunks || 0) - (b.chunks || 0),
  time: (a, b) => (a.indexed_at || 0) - (b.indexed_at || 0),
  status: (a, b) => fileStatusRank(a) - fileStatusRank(b),
};

function fileStatusRank(f) {
  if (f.dot === "red") return 0;
  if (f.job_status === "running") return 1;
  if (f.job_status === "queued" || f.job_status === "retry") return 2;
  if (f.dot === "green") return 4;
  return 3;
}

function fileMatches(f, q, status) {
  if (q && !f.rel_path.toLowerCase().includes(q)) return false;
  switch (status) {
    case "indexed": return f.dot === "green";
    case "pending": return f.dot !== "green" && f.job_status !== "running" && f.dot !== "red";
    case "running": return f.job_status === "running";
    case "failed": return f.dot === "red";
    case "diag": return !!(f.chunk_diag && !f.chunk_diag.ok);
    default: return true;
  }
}

async function refreshFilesTab(force = false) {
  const kb = selectedEntry();
  if (!kb || kb.state !== "active") return;
  const fresh = state.files.kbId === kb.kb_id && Date.now() - state.files.at < (kb.jobs_active ? 2000 : 8000);
  if (!force && fresh) return;
  try {
    // Send the last ETag: unchanged content gets a 304 from the server and the whole table is neither sent
    // nor redrawn (every 2 s while parsing; health check D5)
    const sameKb = state.files.kbId === kb.kb_id;
    const load = () => fetch(`/api/kbs/${encodeURIComponent(kb.kb_id)}/files`,
      { headers: { ...authHeaders(), ...(sameKb && state.files.etag ? { "If-None-Match": state.files.etag } : {}) } });
    const sentWith = authHeaders().Authorization || "";
    let res = await load();
    if (res.status === 401 && ((authHeaders().Authorization || "") !== sentWith || await askToken())) res = await load();
    if (selectedEntry()?.kb_id !== kb.kb_id) return;   // the KB was switched meanwhile
    if (res.status === 304 && sameKb) { state.files.at = Date.now(); return; }
    if (!res.ok) throw new Error(res.statusText || `HTTP ${res.status}`);
    const rows = await res.json();
    if (!sameKb) state.files.page = 1;
    state.files = { ...state.files, kbId: kb.kb_id, rows, at: Date.now(), etag: res.headers.get("ETag") || null };
    renderFilesTable();
  } catch (e) {
    setHtml($("#fl-wrap"), `<div class="empty-state">${t("载入失败: ")}${esc(t(e.message))}</div>`);
  }
}

// The dot at the start of a row: pulsing yellow while a job runs (same as the sidebar KB dot), static
// yellow when queued; otherwise by indexed state. The server's dot only checks "indexed version == current
// version", so a file being re-parsed stays green throughout and the dot alone does not show it is running
function fileDot(f) {
  if (f.job_status === "running") return "yellow pulse";
  if (f.job_status === "queued" || f.job_status === "retry") return "yellow";
  return f.dot;
}

function fileStatusCell(f) {
  if (f.job_status === "running") return `<span class="stage">${esc(t(f.stage || "解析中"))}</span>${f.dot === "green" ? `<div class="hint" style="margin:2px 0 0">${t("重新解析中,旧切片仍可用")}</div>` : ""}`;
  if (f.job_status === "queued" || f.job_status === "retry") return `<span class="badge">${t("排队中")}</span>${f.dot === "green" ? `<div class="hint" style="margin:2px 0 0">${t("等待重新解析")}</div>` : f.stage ? `<div class="hint" style="margin:2px 0 0">${t("上次到:")}${esc(t(f.stage))}</div>` : ""}`;
  if (f.dot === "green") return `<span class="badge ok">${t("已入库")}</span>`;
  if (f.dot === "red") return `<span class="badge bad">${t("失败")}</span>${f.error ? `<div class="err" style="margin-top:3px">${esc(t(f.error).slice(0, 160))}</div>` : ""}`;
  return `<span class="badge">${t("待解析")}</span>`;
}

function renderFilesTable() {
  const wrap = $("#fl-wrap");
  const all = state.files.rows || [];
  const q = $("#fl-q").value.trim().toLowerCase();
  const status = $("#fl-status").value;
  let rows = all.filter(f => fileMatches(f, q, status));
  const cmp = FILE_SORTS[state.files.sort] || FILE_SORTS.path;
  rows.sort((a, b) => cmp(a, b) * state.files.dir);
  const pages = Math.max(1, Math.ceil(rows.length / FILES_PAGE));
  if (state.files.page > pages) state.files.page = pages;
  const start = (state.files.page - 1) * FILES_PAGE;
  const page = rows.slice(start, start + FILES_PAGE);
  const diagCount = all.filter(f => f.chunk_diag && !f.chunk_diag.ok).length;
  $("#fl-count").textContent = t("{0} 个文件", `${rows.length}${rows.length !== all.length ? ` / ${all.length}` : ""}`) + (diagCount ? t(" · {0} 个有切块提示", diagCount) : "");
  if (!all.length) { setHtml(wrap, `<div class="empty-state">${t("暂无文件(等待同步或扫描)")}</div>`); setHtml($("#fl-pager"), ""); return; }
  if (!rows.length) { setHtml(wrap, `<div class="empty-state">${t("没有符合筛选条件的文件")}</div>`); setHtml($("#fl-pager"), ""); return; }
  const th = (key, label, cls = "") => {
    const on = state.files.sort === key;
    return `<th class="sortable ${cls}" data-sort="${key}">${label}${on ? `<span class="arrow">${state.files.dir > 0 ? "▲" : "▼"}</span>` : ""}</th>`;
  };
  const html = `<table><thead><tr>
      <th style="width:22px"></th>
      ${th("path", t("文件"))}
      ${th("size", t("大小"), "num")}
      ${th("chunks", t("切片"), "num")}
      <th>${t("切块诊断")}</th>
      ${th("time", t("入库时间"))}
      ${th("status", t("状态"))}
      <th></th></tr></thead><tbody>` +
    page.map(f => {
      const diag = f.chunk_diag;
      const diagCell = !diag ? `<span class="faint">—</span>`
        : diag.ok ? `<span class="badge ok">${t("通过")}</span>`
        : `<span class="badge warn" title="${esc(diag.reasons.map(x => t(x)).join("\n"))}">⚠ ${esc(t(diag.reasons[0] || "提示"))}</span>`;
      return `<tr>
        <td><span class="dot ${fileDot(f)}"></span></td>
        <td class="path">${f.job_id ? `<span class="fn-link" data-job="${esc(f.job_id)}" title="${t("查看任务时间线")}">${esc(f.rel_path)}</span>` : esc(f.rel_path)}
          ${f.indexed_profile ? `<div class="hint" style="margin:1px 0 0">${esc(f.indexed_profile)}${diag && diag.tokens_mean != null ? t(" · 均长 {0} token", diag.tokens_mean) : ""}</div>` : ""}</td>
        <td class="num">${fmtSize(f.size)}</td>
        <td class="num">${f.chunks != null ? f.chunks : "—"}</td>
        <td>${diagCell}</td>
        <td class="num">${f.indexed_at ? fmtTsShort(f.indexed_at) : "—"}</td>
        <td>${fileStatusCell(f)}</td>
        <td><span class="act">
          ${f.dot === "green" ? `<button class="sm ghost" data-preview="${esc(f.file_id)}">${t("切块预览")}</button>` : ""}
          ${f.dot === "red" || f.dot === "green" ? `<button class="sm ghost" data-retry="${esc(f.file_id)}">${t("重新解析")}</button>` : ""}
        </span></td>
      </tr>`;
    }).join("") + `</tbody></table>`;
  if (setHtml(wrap, html)) {
    $$("th.sortable", wrap).forEach(h => h.addEventListener("click", () => {
      const key = h.dataset.sort;
      if (state.files.sort === key) state.files.dir = -state.files.dir; else { state.files.sort = key; state.files.dir = 1; }
      renderFilesTable();
    }));
    $$("[data-job]", wrap).forEach(el => el.addEventListener("click", () => openJob(el.dataset.job)));
    $$("[data-preview]", wrap).forEach(b => b.addEventListener("click", () => openChunkDrawer(b.dataset.preview)));
    $$("[data-retry]", wrap).forEach(b => b.addEventListener("click", async ev => {
      ev.target.disabled = true;
      try {
        const r = await api("/files/retry", { method: "POST", body: { file_id: ev.target.dataset.retry } });
        toast(r.already_active ? t("已有任务在飞,不重复排队") : t("已重新排队"));
        await refreshFilesTab(true);
      } catch (e) { toast(t("失败: ") + t(e.message)); ev.target.disabled = false; }
    }));
  }
  const pagerHtml = pages > 1
    ? `<button class="sm ghost" id="fl-prev" ${state.files.page <= 1 ? "disabled" : ""}>${t("上一页")}</button>
       <span>${state.files.page} / ${pages}</span>
       <button class="sm ghost" id="fl-next" ${state.files.page >= pages ? "disabled" : ""}>${t("下一页")}</button>`
    : "";
  if (setHtml($("#fl-pager"), pagerHtml)) {
    $("#fl-prev")?.addEventListener("click", () => { state.files.page--; renderFilesTable(); });
    $("#fl-next")?.addEventListener("click", () => { state.files.page++; renderFilesTable(); });
  }
}
$("#fl-q").addEventListener("input", () => { state.files.page = 1; renderFilesTable(); });
$("#fl-status").addEventListener("change", () => { state.files.page = 1; renderFilesTable(); });

/* ── chunk preview drawer: the chunks currently stored in the KB (no re-chunking; to see the effect of new
   rules / parameters use "Re-parse") ── */
function openSide(kind, title) {
  state.side = { ...state.side, kind };
  $("#side-title").textContent = title;
  setPanel("side");
}

async function openChunkDrawer(fileId) {
  const kb = selectedEntry();
  if (!kb || kb.state !== "active") return;
  state.side = { kind: "chunks", fileId };
  openSide("chunks", t("切块预览"));
  const out = $("#side-body");
  setHtml(out, `<div class="empty-state">${t("读取切片中…")}</div>`);
  try {
    const r = await api(`/kbs/${encodeURIComponent(kb.kb_id)}/files/${encodeURIComponent(fileId)}/chunks`);
    if (state.side.kind !== "chunks" || state.side.fileId !== fileId) return;
    state.preview = { r, filter: "all" };
    renderChunkPreview();
  } catch (e) {
    if (state.side.kind !== "chunks" || state.side.fileId !== fileId) return;
    setHtml(out, `<div class="empty-state" style="color:var(--red-text)">${esc(t(e.message))}</div>`);
  }
}

// Merged table cells (F02): table blocks flagged with high confidence; those split after screenshot check
// show ✓, unverified ones show ⚠ (those values are excluded from trusted parameters)
function tableFlagsOf(c) { return (c.table_flags || []).filter(f => f.confidence === "high"); }
function tableBadge(c) {
  const flags = tableFlagsOf(c);
  if (!flags.length) return "";
  const raw = flags.map(f => f.raw).join(", ");
  if (c.table_repair && c.table_repair.status === "verified")
    return `<span class="pv-badge ok" title="${t("截图核验后已拆开:")}${esc(raw)}">✓ ${t("截图核验")}</span>`;
  return `<span class="pv-badge warn" title="${t("未核验的粘连值:")}${esc(raw)}">⚠ ${t("表格粘连")} ${flags.length}</span>`;
}

// Page label: a table / paragraph merged across pages shows the range "p11–12" (Codex review F09); a
// single page stays "p11"
function pageLabel(c) {
  if (c.page_idx == null) return "";
  const end = c.page_end != null && c.page_end !== c.page_idx ? `–${esc(String(c.page_end))}` : "";
  return ` · p${esc(String(c.page_idx))}${end}`;
}

/* ── rendered chunk view. markdown-it draws tables / code / headings, KaTeX the parser's "EQUATION:" lines
   and $$ blocks, and the pipeline's own marker lines (TITLE / CAPTION / VISUAL SUMMARY / ...) become labels.
   Document text never turns into HTML: markdown-it runs with html:false (tags are escaped) and KaTeX with
   trust:false. Inline $...$ is left alone: in these documents it is money, not math. ── */
const PV_MARKER_RE = /^(TITLE|CAPTION|VISUAL SUMMARY|FACTS|ENTITIES|KEYWORDS|FOOTNOTE|IMAGE): ?(.*)$/;
let pvMd;
function pvMarkdown() {
  if (pvMd === undefined) pvMd = typeof markdownit === "function" ? markdownit({ html: false, linkify: false, breaks: true }) : null;
  return pvMd;
}
function pvMath(latex) {
  if (typeof katex === "undefined") return `<code>${esc(latex)}</code>`;
  try { return katex.renderToString(latex, { displayMode: true, throwOnError: false, trust: false, strict: "ignore" }); }
  catch (e) { return `<code>${esc(latex)}</code>`; }
}
function chunkHtml(text) {
  const md = pvMarkdown();
  if (!md) return `<pre>${esc(text)}</pre>`;
  const lines = String(text || "").split("\n"), out = [];
  let buf = [], inFence = false;
  const flush = () => { if (buf.length) { out.push(md.render(buf.join("\n"))); buf = []; } };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/^```/.test(line)) inFence = !inFence;
    let m;
    if (inFence || /^```/.test(line)) { buf.push(line); continue; }
    if ((m = /^EQUATION: ?(.*)$/.exec(line))) { flush(); out.push(`<div class="pv-eq">${pvMath(m[1])}</div>`); continue; }
    if (/^\$\$/.test(line)) {                                   // display math on one line or as a $$ ... $$ block
      const parts = [line.slice(2)];
      while (!/\$\$\s*$/.test(parts[parts.length - 1]) && i + 1 < lines.length) parts.push(lines[++i]);
      flush(); out.push(`<div class="pv-eq">${pvMath(parts.join("\n").replace(/\$\$\s*$/, ""))}</div>`); continue;
    }
    if ((m = PV_MARKER_RE.exec(line))) { flush(); out.push(`<div class="pv-label"><b>${esc(m[1])}</b>${md.renderInline(m[2])}</div>`); continue; }
    buf.push(line);
  }
  flush();
  return out.join("");
}
function pvView() {
  try { return localStorage.getItem("kb.pvview") === "raw" ? "raw" : "md"; } catch (e) { return "md"; }
}

function renderChunkPreview() {
  const { r, filter } = state.preview;
  const view = state.preview.view || pvView();
  const d = r.diagnostics, s = (d && d.stats) || null;
  const budget = r.max_tokens;
  const isVisual = c => ["table", "image", "chart", "vision"].includes(c.block_type);
  const tinyOf = c => !!s && c.tokens < s.tiny_threshold && !isVisual(c);
  const overOf = c => budget != null && c.tokens > budget * ((s && s.over_tolerance) || 1) && !isVisual(c);
  const params = [budget != null ? `max_tokens ${budget}` : "", r.overlap_tokens != null ? `overlap ${r.overlap_tokens}` : "",
                  r.file.parser_profile || ""].filter(Boolean).join(" / ");
  const head = `<div class="pv-file"><b>${esc(r.file.rel_path)}</b><span class="muted">${esc(params)}${r.file.chunks_total != null ? t(" · {0} 片", r.file.chunks_total) : ""}</span></div>`;
  const verdict = !d ? `<div class="hint" style="margin-bottom:10px">${t("这份文件入库时还没有切块诊断")}</div>`
    : d.ok ? `<div class="pv-verdict ok">✓ ${t("切块验收通过")}</div>`
    : `<div class="pv-verdict warn">${d.reasons.map(x => "⚠ " + esc(t(x.message))).join("<br>")}</div>`;
  const depths = s ? Object.entries(s.section_depths || {}).map(([k, v]) => t("{0} 层×{1}", k, v)).join(" · ") : "";
  const stats = s ? `<div class="pv-stats">
    <div><b>${s.chunks}</b><span>${t("切片 / {0} 个块", s.blocks ?? "-")}</span></div>
    <div><b>${s.tokens_mean}</b><span>${t("均长 token · σ {0}", s.tokens_stddev)}</span></div>
    <div><b>${s.tokens_min}–${s.tokens_max}</b><span>${t("最短 – 最长")}</span></div>
    <div><b>${s.tiny_count}</b><span>${t("碎片")} &lt;${s.tiny_threshold} token</span></div>
    <div><b>${s.over_count}</b><span>${t("超预算")}</span></div>
    <div><b>${s.headings ?? "-"}</b><span>${t("标题 · 补认 {0}", s.inferred_headings ?? 0)}${depths ? t(" · 深度 {0}", esc(depths)) : ""}</span></div>
  </div>` : "";
  const types = s ? Object.entries(s.by_block_type || {}).map(([k, v]) => [k, v.chunks])
    : Object.entries(r.chunks.reduce((m, c) => { m[c.block_type] = (m[c.block_type] || 0) + 1; return m; }, {}));
  const chip = (key, label, n) => `<span class="chip ${filter === key ? "on" : ""}" data-f="${esc(key)}">${esc(label)}${n != null ? ` ${n}` : ""}</span>`;
  const ambiguous = r.chunks.filter(c => tableFlagsOf(c).length > 0).length;
  const filters = `<div class="pv-filters">
    ${chip("all", t("全部"), r.chunks.length)}
    ${s ? chip("tiny", t("碎片"), s.tiny_count) : ""}
    ${ambiguous ? chip("ambiguous", t("表格粘连"), ambiguous) : ""}
    ${types.map(([k, n]) => chip("type:" + k, k, n)).join("")}
    <span class="pv-view"><span class="chip ${view === "md" ? "on" : ""}" data-v="md">${t("渲染")}</span><span class="chip ${view === "raw" ? "on" : ""}" data-v="raw">${t("原文")}</span></span>
  </div>`;
  const show = r.chunks.filter(c => filter === "all" ? true : filter === "tiny" ? tinyOf(c)
    : filter === "ambiguous" ? tableFlagsOf(c).length > 0 : c.block_type === filter.slice(5));
  const cards = show.map(c => {
    const tiny = tinyOf(c), over = overOf(c);
    return `<details class="pv-chunk${tiny ? " tiny" : ""}${over ? " over" : ""}">
      <summary><span class="t">#${c.chunk_index}</span><span class="tok">${c.tokens} tok</span>
        <span class="bt">${esc(c.block_type)}${pageLabel(c)}</span>${tableBadge(c)}
        <span class="sp" title="${esc(c.block_id)}">${esc((c.section_path || []).join(" › ") || c.block_id)}</span>
        <span class="pv-peek">${esc(c.text.slice(0, 80).replace(/\s+/g, " "))}</span></summary>
      ${view === "raw" ? `<pre>${esc(c.text)}</pre>` : `<div class="pv-md">${chunkHtml(c.text)}</div>`}</details>`;
  }).join("");
  const html = head + verdict + stats + filters + (cards || `<div class="empty-state">${t("没有符合筛选的切片")}</div>`) +
    (r.truncated ? `<div class="hint">${t("只显示前 {0} 片", r.chunks.length)}</div>` : "");
  if (!setHtml($("#side-body"), html)) return;
  $$("#side-body .pv-filters .chip[data-f]").forEach(el => el.addEventListener("click", () => { state.preview.filter = el.dataset.f; renderChunkPreview(); }));
  $$("#side-body .pv-filters .chip[data-v]").forEach(el => el.addEventListener("click", () => {
    state.preview.view = el.dataset.v;
    try { localStorage.setItem("kb.pvview", el.dataset.v); } catch (e) { /* private mode: not remembered */ }
    renderChunkPreview();
  }));
}

/* ── merges drawer: resolution_log / resolution_rejected from the current version's graph.json -- every
   merged pair (merged into whom, route, evidence category) and every blocked pair (reason). Merges with
   grounds and auditable is the discipline set by the generality review of 2026-09-08; this is where the
   ledger is inspected; the text box filters by entity name. ── */
const MERGE_SOURCE_LABEL = { auto: "规则", lexical: "字面", embedding: "向量", replay: "回放" };
const MERGE_CATEGORY_LABEL = { identifier: "记号相等", type_words: "去类型 / 后缀词相同", paren: "括号别名", abbreviation: "缩写", spelling: "拼写", translation: "翻译", alias: "别名", unspecified: "未给依据", prior: "上一版" };
const MERGE_REASON_LABEL = { polarity: "极性相反", recheck: "第二道判定" };
// The second-pass check answers the relation of "entity" to "merged into": broader (the entity is the
// larger class) / narrower (the entity is a kind, part or manifestation of it) / different
const MERGE_VERDICT_LABEL = { broader: "上位", narrower: "下位", different: "不同", "": "未答" };
const rejectReason = (m) => t(m.reason === "recheck" ? (MERGE_VERDICT_LABEL[m.verdict || ""] || "未答") : (MERGE_REASON_LABEL[m.reason] || m.reason));

async function openMergesDrawer() {
  const kb = selectedEntry();
  if (!kb || kb.state !== "active") return;
  state.side = { kind: "merges", fileId: null };
  openSide("merges", t("实体归并审计"));
  const out = $("#side-body");
  setHtml(out, `<div class="empty-state">${t("读取归并日志中…")}</div>`);
  // The merge log of a large KB takes a dozen seconds or more to read, long enough to switch to another KB and open
  // the drawer again: a response that is not the latest request's, or arrives after the KB changed, is dropped
  const seq = ++mergesReqSeq;
  const stale = () => seq !== mergesReqSeq || state.side.kind !== "merges" || selectedEntry()?.kb_id !== kb.kb_id;
  try {
    const r = await api(`/kbs/${encodeURIComponent(kb.kb_id)}/graph_merges`);
    if (stale()) return;
    state.merges = { r, q: "" };
    renderMerges();
  } catch (e) {
    if (stale()) return;
    setHtml(out, `<div class="empty-state" style="color:var(--red-text)">${esc(t(e.message))}</div>`);
  }
}
let mergesReqSeq = 0;

function renderMerges() {
  const { r, q } = state.merges || {};
  const host = $("#side-body");
  if (!r || !r.version) { setHtml(host, `<div class="empty-state">${t("还没有建成的图谱版本")}</div>`); return; }
  const needle = (q || "").trim().toLowerCase();
  const hit = (...names) => !needle || names.some(n => String(n || "").toLowerCase().includes(needle));
  const merges = (r.merges || []).filter(m => hit(m.kept_title, m.merged_title));
  const rejected = (r.rejected || []).filter(m => hit(m.a_title, m.b_title));
  const st = r.stats || {};
  const bySource = {}, byCat = {};
  for (const m of r.merges || []) { bySource[m.source] = (bySource[m.source] || 0) + 1; byCat[m.category] = (byCat[m.category] || 0) + 1; }
  const chips = Object.entries(bySource).map(([k, v]) => `<span class="chip">${esc(t(MERGE_SOURCE_LABEL[k] || k))} ${v}</span>`).join("")
    + Object.entries(byCat).map(([k, v]) => `<span class="chip">${esc(t(MERGE_CATEGORY_LABEL[k] || k))} ${v}</span>`).join("");
  // The first line keeps only the version (user, 2026-09-08: no merged / blocked / rechecked / entity count
  // string); per-item counts are in the chips below
  const head = `<div class="hint">${t("版本")} ${esc(r.version)}</div>`
    + `<div class="pv-filters">${chips}</div>`
    + `<div class="tool-row"><input type="text" id="merge-q" placeholder="${t("按实体名过滤")}" value="${esc(q || "")}"></div>`;
  // The merged / blocked sections share one table: the fields correspond one to one (entity / merged into /
  // route / grounds), so the columns line up naturally; cells do not wrap, overflow scrolls horizontally
  // in the outer .tbl-wrap
  // Fixed column widths (the table fills the drawer, and both sections are one table so they align); cell
  // content does not wrap, an overflowing cell scrolls in a loop inside its own cell (marqueeMergeCells,
  // like the long names in the model panel), no dragging needed
  const cell = (s) => `<td><div class="cell"><span class="cell-in"><span class="seg">${esc(s)}</span></span></div></td>`;
  const rows = merges.slice(0, 500).map(m =>
    `<tr>${cell(m.merged_title)}${cell(m.kept_title)}${cell(t(MERGE_SOURCE_LABEL[m.source] || m.source))}${cell(t(MERGE_CATEGORY_LABEL[m.category] || m.category))}</tr>`).join("");
  const rows2 = rejected.slice(0, 500).map(m =>
    `<tr>${cell(m.a_title)}${cell(m.b_title)}${cell(t(MERGE_SOURCE_LABEL[m.source] || m.source))}${cell(rejectReason(m))}</tr>`).join("");
  const none = `<tr><td colspan="4" class="merge-none">${t("没有")}</td></tr>`;
  const html = head
    + `<div class="tbl-wrap merge-wrap"><table class="merge-tbl">`
    + `<colgroup><col><col><col style="width:64px"><col style="width:150px"></colgroup>`
    + `<thead><tr><th>${t("实体")}</th><th>${t("并入实体")}</th><th>${t("来源")}</th><th>${t("依据")}</th></tr></thead><tbody>`
    + `<tr class="merge-sec"><td colspan="4">${t("合并实体")} <span class="n">${merges.length}</span></td></tr>${rows || none}`
    + `<tr class="merge-sec"><td colspan="4">${t("未合并实体")} <span class="n">${rejected.length}</span></td></tr>${rows2 || none}`
    + `</tbody></table></div>`;
  if (!setHtml(host, html)) return;
  const input = $("#merge-q");
  input?.addEventListener("input", () => { state.merges.q = input.value; const pos = input.selectionStart; renderMerges(); const el = $("#merge-q"); el?.focus(); try { el?.setSelectionRange(pos, pos); } catch (e) {} });
  marqueeMergeCells(host);
}

// Overflowing cells scroll in a loop (same approach as applyNameMarquee): a copy follows the content, and
// after shifting by "one width + gap" the copy lands exactly on the original, seamless end to end; constant
// speed, so longer text scrolls longer; cells that fit stay perfectly still.
// Measure twice: right after innerHTML is set nothing is laid out yet; the drawer has an entrance
// animation, so the first measurement may land mid-animation.
function marqueeMergeCells(host) {
  const GAP = 36;
  const apply = () => {
    for (const cell of host.querySelectorAll(".merge-tbl td .cell")) {
      const inner = cell.querySelector(".cell-in"), seg = inner && inner.querySelector(".seg");
      if (!seg) continue;
      inner.querySelectorAll(".seg-clone").forEach(node => node.remove());
      cell.classList.remove("scroll");
      cell.style.removeProperty("--shift"); cell.style.removeProperty("--dur");
      const textWidth = Math.ceil(seg.getBoundingClientRect().width);
      if (textWidth - Math.floor(cell.clientWidth) <= 2) continue;
      const clone = seg.cloneNode(true);
      clone.classList.add("seg-clone"); clone.setAttribute("aria-hidden", "true");
      inner.appendChild(clone);
      const shift = textWidth + GAP;
      cell.style.setProperty("--shift", shift + "px");
      cell.style.setProperty("--dur", Math.max(6, shift / 26) + "s");
      cell.classList.add("scroll");
    }
  };
  apply();
  setTimeout(apply, 60);
  setTimeout(apply, 400);
}

/* ── job timeline drawer: opened by clicking a file name in the file table; shows this parse's phases,
   durations and errors; can cancel and retry ── */
const JOB_STATUS_ZH = { queued: "排队中", retry: "等待重试", running: "解析中", done: "已完成", failed: "失败", cancelled: "已取消" };

function openJob(jobId) {
  state.jobs.selected = jobId;
  state.side = { kind: "job", fileId: null };
  openSide("job", t("任务时间线"));
  setHtml($("#side-body"), `<div class="empty-state">${t("载入中…")}</div>`);
  refreshJobDetail();
}

async function refreshJobDetail() {
  const jobId = state.jobs.selected;
  const host = $("#side-body");
  if (!jobId || state.side.kind !== "job") return;
  let d;
  try { d = await api(`/jobs/${encodeURIComponent(jobId)}`); }
  catch (e) { setHtml(host, `<div class="empty-state" style="color:var(--red-text)">${esc(t(e.message))}</div>`); return; }
  if (state.jobs.selected !== jobId) return;
  const j = d.job, f = d.file || {};
  const end = j.finished_at || (j.status === "running" ? Math.floor(Date.now() / 1000) : null);
  const meta = [
    [t("状态"), t(JOB_STATUS_ZH[j.status] || j.status) + (j.cancel_requested ? t(" · 已请求取消") : "")],
    [t("当前阶段"), t(j.stage) || "—"],
    [t("开始"), fmtTs(j.started_at)], [t("结束"), fmtTs(j.finished_at)],
    [t("时长"), j.started_at && end ? fmtDur(end - j.started_at) : "—"],
    [t("重试次数"), j.retry_count ?? 0],
    [t("解析 profile"), j.parser_profile || "—"],
    [t("文件大小"), f.size != null ? fmtSize(f.size) : "—"],
    [t("任务 id"), j.job_id],
  ];
  const ev = d.events || [];
  const timeline = ev.length ? ev.map((e, i) => {
    const next = ev[i + 1];
    const dur = e.kind === "stage" ? fmtDur((next ? next.ts : (end || e.ts)) - e.ts) : "";
    const ts = new Date(e.ts * 1000).toLocaleTimeString(I18N.locale, { hour12: false });
    const tag = e.kind === "error" ? "[ERROR] " : e.kind === "retry" ? "[RETRY] " : "";
    return `<div class="${esc(e.kind)}${e.synthesized ? " synth" : ""}"><span class="ts">${ts}</span><span class="dur">${dur}</span><span class="tx">${tag}${esc(t(e.text))}</span></div>`;
  }).join("") : `<div class="muted">${t("还没有时间线记录")}</div>`;
  const canCancel = ["queued", "retry", "running"].includes(j.status) && !j.cancel_requested;
  const canRetry = ["failed", "cancelled", "done"].includes(j.status) && f.file_id;
  const html = `
    <div style="display:flex;align-items:baseline;gap:10px">
      <b style="flex:1;min-width:0;word-break:break-all;font-size:14px">${esc(f.rel_path || j.file_id || jobId)}</b>
      <span class="badge ${j.status === "failed" ? "bad" : j.status === "done" ? "ok" : j.status === "running" ? "info" : ""}">${esc(t(JOB_STATUS_ZH[j.status] || j.status))}</span>
    </div>
    <div class="job-meta">${meta.map(([k, v]) => `<div><span>${k}</span><b>${esc(v)}</b></div>`).join("")}</div>
    ${d.synthesized ? `<div class="hint" style="margin:0 0 6px">${t("这条任务早于时间线记录,下面是按开始/结束时间合成的概要")}</div>` : ""}
    <div class="timeline">${timeline}</div>
    <div class="row-actions" style="justify-content:flex-end;margin-top:12px">
      ${canCancel ? `<button class="danger sm" id="job-cancel">${t("取消任务")}</button>` : ""}
      ${canRetry ? `<button class="sm" id="job-retry">${t("重新解析")}</button>` : ""}
      ${f.file_id && f.status !== "deleted" ? `<button class="sm ghost" id="job-preview">${t("切块预览")}</button>` : ""}
    </div>`;
  if (!setHtml(host, html)) return;   // unchanged: do not redraw, or the "Cancel job" about to be clicked gets replaced
  $("#job-cancel")?.addEventListener("click", async () => {
    try {
      const r = await api(`/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
      toast(r.outcome === "signalled" ? t("已请求取消,任务会在下一个阶段边界退出") : r.outcome === "cancelled" ? t("已取消") : t("任务已结束,无需取消"));
      await refreshFilesTab(true);
      refreshJobDetail();
    } catch (e) { toast(t("取消失败: ") + t(e.message)); }
  });
  $("#job-retry")?.addEventListener("click", async ev2 => {
    ev2.target.disabled = true;
    try {
      const r = await api("/files/retry", { method: "POST", body: { file_id: f.file_id } });
      toast(r.already_active ? t("已有任务在飞,不重复排队") : t("已重新排队"));
      await refreshFilesTab(true);
      if (r.job_id) { state.jobs.selected = r.job_id; refreshJobDetail(); }
    } catch (e) { toast(t("失败: ") + t(e.message)); ev2.target.disabled = false; }
  });
  $("#job-preview")?.addEventListener("click", () => openChunkDrawer(f.file_id));
}

/* ── graph tab: status card + build records ──────────────────────── */
/* Parse progress does not jump in four steps by file count: a file being parsed is interpolated to a
   0~1 fraction by pipeline phase. Phase weights are roughly tuned by real time share (MinerU and the VLM
   dominate), and within the VLM phase a second interpolation follows n/m images, so an image-heavy PDF
   shows progress climbing even in the slowest step. */
const STAGE_SPANS = [
  ["Parsing document",  0.00, 0.35],
  ["Describing images", 0.35, 0.70],
  ["Chunking",          0.70, 0.73],
  ["Text embedding",    0.73, 0.83],
  ["Visual embedding",  0.83, 0.93],
  ["Writing vectors",   0.93, 0.97],
  ["Keyword indexing",  0.97, 1.00],
];

/* Assumed phase durations (seconds) when there is no history: only used to convert the fixed table into
   the same "seconds per span" measure. With history (overview.kbs[].parse_stage_weights, the median
   per-phase duration of this KB's recent parses) the history is used.
   Measured 2026-09-08 on a 44 MB PDF: document parsing 104 s, image description 40 s, everything else
   4 s, while the fixed table said 35 : 35 : 30 -- the bar sat at 9% for over a hundred seconds and then
   jumped to 70%. */
const STAGE_DEFAULT_SECONDS = { "Parsing document": 60, "Describing images": 40, "Chunking": 2, "Text embedding": 8, "Visual embedding": 5, "Writing vectors": 3, "Keyword indexing": 2 };

function parseStageSpans(weights) {
  const secs = STAGE_SPANS.map(([name]) => Math.max(1, Number((weights && weights[name]) ?? STAGE_DEFAULT_SECONDS[name] ?? 3)));
  const total = secs.reduce((a, b) => a + b, 0);
  let at = 0;
  return STAGE_SPANS.map(([name], i) => { const row = [name, at, at + secs[i] / total, secs[i]]; at += secs[i] / total; return row; });
}

// job: the entry from overview.active_jobs (with stage_elapsed: seconds spent in the current phase,
// computed server-side); weights: this KB's phase duration table. Interpolate within the phase by elapsed
// time, capped at 95%, and move on when the phase changes -- image description follows n/m when available.
function stageFraction(stage, job, weights) {
  if (!stage) return 0.02;
  if (stage === "Done") return 1;
  const spans = (job || weights) ? parseStageSpans(weights) : STAGE_SPANS.map(s => [...s, STAGE_DEFAULT_SECONDS[s[0]] || 3]);
  for (const [name, lo, hi, secs] of spans) {
    if (stage.startsWith(name)) {
      if (name === "Describing images") {
        const m = stage.match(/VLM (\d+)\/(\d+)/);
        if (m && +m[2] > 0) return lo + (hi - lo) * (+m[1] / +m[2]);
      }
      const elapsed = job && Number.isFinite(Number(job.stage_elapsed)) ? Number(job.stage_elapsed) : null;
      if (elapsed != null && secs > 0) return lo + (hi - lo) * Math.min(0.95, elapsed / secs);
      return lo + (hi - lo) * 0.5;
    }
  }
  return 0.02;
}

/* Stage name -> progress span. The names must match, letter for letter, the prefixes written by stage() /
   progress() in build.py / vectors.py; a regression test guards this
   (test_stage_labels_are_declared_in_the_build). Weights follow measured kb_003 durations: entity
   extraction and description summaries are the two big ones, everything else takes seconds.
   The fourth item lists the sub-segments: "Description summaries · entities 23/511" then "Description
   summaries · relations 8/315"; each takes half of the span and counts its denominator from zero. */
const GRAPH_STAGES = [
  ["Preparing corpus",                       0.00, 0.02],
  ["Entity extraction",                      0.02, 0.45],
  ["Entity resolution",                      0.45, 0.50],
  ["Description summaries",                  0.50, 0.68, ["entities", "relations"]],
  ["Structured facts",                       0.68, 0.82],
  ["Compiling view pages",                   0.82, 0.87, ["subjects", "timelines", "sources", "narration"]],
  ["Writing vectors",                        0.87, 0.94, ["entities", "relations", "facts", "pages"]],
  ["Graph database import",                  0.94, 0.98],
  ["Switching version aliases",              0.98, 0.99],
  ["Cleaning up old versions",               0.99, 1.00],
];

/* When the position inside a span is unknown, take the lower bound, never the midpoint: the midpoint is
   an overestimate, while N/M of the same step climbs from the lower bound. Mixing both estimates into one
   monotonic clamp (graphPctFloor) lets the earlier overestimate lock the bar -- the previous console was
   measured stuck at 42%, and only a reload showed the real progress. */
/* When the last completed build has actual per-phase durations (graph_build.stage_weights, seconds by
   phase name) the spans follow them; without them (first build, resume skipped too much) fall back to the
   fixed table above (health check D8). The merge / resolution phase is split entity resolution 2 :
   description summaries 8 -- description summaries are LLM calls, resolution is just a few verdicts. */
function graphStagesFor(weights) {
  if (!weights || typeof weights !== "object") return GRAPH_STAGES;
  const w = k => Math.max(0, Number(weights[k] || 0));
  const total = w("prepare_input") + w("extract") + w("merge") + w("facts") + w("compile") + w("enrich") + w("neo4j_import");
  if (!(total > 0)) return GRAPH_STAGES;
  const span = 0.98 / total;
  let at = 0;
  const seg = (name, width, parts) => { const row = [name, at, at + width]; if (parts) row.push(parts); at += width; return row; };
  const merge = w("merge") * span;
  return [
    seg("Preparing corpus", w("prepare_input") * span),
    seg("Entity extraction", w("extract") * span),
    seg("Entity resolution", merge * 0.2),
    seg("Description summaries", merge * 0.8, ["entities", "relations"]),
    seg("Structured facts", w("facts") * span),
    seg("Compiling view pages", w("compile") * span, ["subjects", "timelines", "sources", "narration"]),
    seg("Writing vectors", w("enrich") * span, ["entities", "relations", "facts", "pages"]),
    seg("Graph database import", w("neo4j_import") * span),
    seg("Switching version aliases", 0.01),
    seg("Cleaning up old versions", 0.01),
  ];
}

function graphStagePct(stg, weights) {
  if (!stg) return 2;
  if (stg === "Done") return 100;
  const row = graphStagesFor(weights).find(([n]) => stg.startsWith(n));
  if (!row) return 2;
  let [name, lo, hi] = row;
  const parts = row[3];
  const rest = stg.slice(name.length);
  if (parts) {
    const k = parts.findIndex(p => rest.includes("· " + p));
    if (k >= 0) { const w = (hi - lo) / parts.length; lo = lo + w * k; hi = lo + w; }
  }
  const m = rest.match(/(\d+)\s*\/\s*(\d+)/);
  const frac = m && +m[2] > 0 ? Math.min(+m[1] / +m[2], 1) : 0;
  return Math.round((lo + (hi - lo) * frac) * 100);
}

/* The progress bar may only move forward. Phases skipped on resume make the stage name jump ahead, and
   computing honestly would jump back. Record the maximum seen per "this build"; reset only when the build
   changes (started_at differs). */
const graphFloors = {};

function graphPctFloor(kbId, startedAt, pct) {
  const key = String(kbId);
  const prev = graphFloors[key];
  if (!prev || prev.startedAt !== startedAt) {
    graphFloors[key] = { startedAt, pct };
    return pct;
  }
  prev.pct = Math.max(prev.pct, pct);
  return prev.pct;
}

// Document-level delta of an incremental append, in one sentence
function deltaText(d) {
  return t("新增 {0} 份 · 变更 {1} 份 · 删除 {2} 份 · 新切片 {3}", (d.added_docs || []).length, (d.modified_docs || []).length, (d.removed_docs || []).length, fmtNum(d.new_chunks || 0));
}
// Current status card: progress, phase, resumability note. graph_status is the backend's reading of the
// most recent build.
function renderGraphPanel() {
  const host = $("#graph-status");
  const kb = selectedEntry();
  if (!kb || kb.state !== "active") return;
  const b = kb.graph_build || {};
  const st = kb.graph_status;
  if (!st || st === "disabled") {
    setHtml(host, `<div class="gstat"><span class="dot gray"></span><span class="big">${t("未开启知识图谱")}</span>
      <span class="muted">${t("在「配置管理」里打开「开启知识图谱」、选好模型后保存,在那里建图")}</span></div>`);
    return;
  }
  // The progress bar is drawn only under the KB label on the left (since 2026-09-06); this card only states
  // status, phase and the resumability note
  let text = "", extra = "";
  const fail = st === "failed";
  if (st === "running") {
    text = t(b.stage || "构建中");
    extra = b.started_at ? t("已运行 {0}", fmtDur(Math.floor(Date.now() / 1000) - b.started_at)) : "";
  } else if (st === "ok") {
    text = "";                                    // a built graph speaks only through the three cards below
  } else if (st === "failed") {
    text = t("失败于「{0}」", t(b.stage || "未知步骤"));
    extra = b.error ? t(String(b.error)).slice(0, 220) : "";
  } else if (st === "stopped") {
    // There are two kinds of stopped, and auto-rebuild treats them oppositely, so the UI must tell them
    // apart: "Pause build" was clicked = persistent intent, the 00:00 round skips it; stopped for another
    // reason (graph turned off, process killed) is not intent, and the 00:00 round resumes it per policy
    // -- which is also right.
    text = t(b.paused ? "已暂停于「{0}」" : "已停止于「{0}」", t(b.stage || "未知步骤"));
    extra = [b.cache_entries ? t("LLM 缓存 {0} 条可复用", fmtNum(b.cache_entries)) : "",
             b.extraction_entries ? t("已存抽取记录 {0} 个单元", fmtNum(b.extraction_entries)) : "",
             b.paused ? t("自动重建已跳过") : ""].filter(Boolean).join(" · ");
  } else {
    text = t("待构建");
    extra = t("首次开启无条件构建(每 2 小时检查一次,解析忙时让路);之后新增内容自动并入,达到重建条件整库重来");
  }
  const phases = (b.phases_done || []);
  // Size of the current graph: three cards for entities / relations / units plus an "updated" line; while
  // building, failed or paused a status line appears above
  const c = b.counts || null;
  const cards = c ? `<div class="pv-stats gstats">
      <div><b>${fmtNum(c.entities ?? 0)}</b><span>${t("实体")}</span></div>
      <div><b>${fmtNum(c.relations ?? 0)}</b><span>${t("关系")}</span></div>
      <div><b>${fmtNum(c.text_units ?? 0)}</b><span>${t("单元")}</span></div>
    </div>${b.counts_at ? `<div class="gstats-at">${t("更新于")} ${esc(fmtTs(b.counts_at))}</div>` : ""}` : "";
  // Units whose extraction failed (at <=5% the build still completes) used to be invisible in the UI: say
  // how many and how many documents are involved (health check R3)
  const failedUnits = b.extract_failed_units
    ? `<div class="hint" style="margin-top:6px">${t("{0} 个单元抽取失败", fmtNum(b.extract_failed_units))}${(b.extract_failed_documents || []).length ? t("(涉及 {0} 份文档)", b.extract_failed_documents.length) : ""}${t(",已成功的都在图里;下次建图只补这些")}</div>`
    : "";
  // The number of units the facts phase still could not finish after split retries stays only in the API
  // field facts_partial_units; the status card no longer shows that note (2026-09-12, user decision)
  setHtml(host, `
    ${text ? `<div class="gstat"><span class="dot ${graphDot(kb)}"></span><span class="big">${esc(text)}</span></div>` : ""}
    ${extra ? `<div class="${fail ? "err" : "muted"} small" style="word-break:break-all">${esc(extra)}</div>` : ""}
    ${phases.length ? `<div class="phases">${phases.map(p => `<span class="phase done">✓ ${esc(t(p))}</span>`).join("")}<span class="muted small" style="align-self:center">${t("续跑时跳过")}</span></div>` : ""}
    ${cards || (st === "ok" ? `<div class="muted small">${t("已建成")}</div>` : "")}${failedUnits}`);
}

/* ── graph preview: entities and relations of the current version, drawn as a force-directed graph ───────
   Data comes from the graph.json build artifact: by default the top N entities by degree (the old "whole
   graph" mode that took everything, limit=0, is gone).
   The layout is computed in the browser: repulsion + springs + centering; with many nodes (> 250) the
   repulsion uses a Barnes-Hut quadtree, and the simulation runs in frames, drawing as it goes, so
   thousands of nodes settle in two or three seconds.
   Levels: every change of view (search / click an entity / legend filter) is one level, and the top-left
   ← → jump between levels; the canvas supports wheel zoom, drag on empty space to pan, drag an entity to
   move it, double-click empty space to reset. */
const GP_COLORS = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#0ea5e9", "#a855f7", "#ec4899", "#14b8a6", "#64748b"];
const GP_BH_MIN = 250;                                   // above this many nodes, repulsion switches to the quadtree approximation

function gpColor(upper, palette) {
  const i = palette.indexOf(upper);
  return GP_COLORS[(i >= 0 ? i : palette.length) % GP_COLORS.length];
}
// Chinese names of the upper ontology classes: legend, cluster labels and tooltip all use "Chinese +
// English name"; the English name matches the parent types on the config page
const UPPER_ZH = { entity: "事物", part: "部件", property: "属性", process: "过程", standard: "标准", document: "文档", substance: "物质" };
function gpUpperZh(u) { return UPPER_ZH[u] || ""; }
// The English UI shows only the English class name (capitalized), without repeating the English code
function gpUpperText(u) { const zh = gpUpperZh(u); if (!u) return t("未分类"); if (!zh) return u; return I18N.lang === "en" ? t(zh) : `${zh} ${u}`; }
// Same-name entities (document-scoped entities exist once per document) carry a document label on the
// canvas: "Urinalysis · 2024-07-26"
function gpLabelOf(n, dupTitles) {
  const t = n.title.length > 18 ? n.title.slice(0, 17) + "…" : n.title;
  return n.doc && dupTitles.has(n.title) ? `${t} · ${n.doc}` : t;
}

// ── levels (browsing history): hist is a list of views, cur points at the current level ──
function gpView() {
  const g = state.gp;
  if (!g.hist.length) { g.hist = [{ q: "", key: "", upper: "" }]; g.cur = 0; }
  return g.hist[g.cur];
}
function gpSameView(a, b) { return !!a && !!b && a.q === b.q && a.key === b.key && a.upper === b.upper; }
function gpGo(view) {
  const g = state.gp;
  const v = { q: "", key: "", upper: "", ...view };
  if (!gpSameView(v, gpView())) {                        // a new level: drop the branch under "next level"
    g.hist = g.hist.slice(0, g.cur + 1);
    g.hist.push(v); g.cur = g.hist.length - 1;
  }
  gpLoadView();
}
function gpStep(delta) {
  const g = state.gp, to = g.cur + delta;
  if (to < 0 || to >= g.hist.length) return;
  g.cur = to;
  gpLoadView();
}
function gpLoadView() {
  const v = gpView();
  $("#gp-q").value = v.q || "";
  state.gp.data = null; state.gp.error = null;
  renderGraphNav();
  refreshGraphPreview(true);
}
function renderGraphNav() {
  const g = state.gp;
  $("#gp-back").disabled = g.cur <= 0;
  $("#gp-fwd").disabled = g.cur >= g.hist.length - 1;
}

async function refreshGraphPreview(force = false) {
  const kb = selectedEntry();
  if (!kb || kb.state !== "active") return;
  if (state.gp.kbId && state.gp.kbId !== kb.kb_id) { state.gp.hist = []; state.gp.cur = -1; }   // KB switched, levels start over
  // The preview draws the most recently completed version (active_graph_version), not the new version
  // number of a build in progress: during a build it must not re-fetch, re-layout and reset the camera on
  // every poll (2026-09-06 health check B3). Old backends without the field fall back to graph_version.
  const gb0 = kb.graph_build || {};
  const version = gb0.active_graph_version || (kb.graph_status === "ok" ? gb0.graph_version : null) || null;
  // When the latest build failed or was paused, the version built before is still there and search keeps using it,
  // so the preview draws it as well; only a KB without the graph turned on draws nothing
  const ok = !!version && !!kb.graph_status && kb.graph_status !== "disabled";
  if (!ok) {
    state.gp = { ...state.gp, kbId: kb.kb_id, version: null, viewKey: null, data: null, pos: null, sim: null, loading: null };
    renderGraphPreview();
    return;
  }
  // The request identity = KB + version + full view parameters (query, focus entity, class, count): a
  // different view is not "the same data", and the 5-second throttle applies only within one view. Every
  // request carries a sequence number and a response whose number is no longer the latest is dropped --
  // clicking two views in quick succession used to let the earlier slow response overwrite the later one,
  // with B selected in the UI but A drawn (Codex review F08)
  const view = gpView();
  const viewKey = JSON.stringify([kb.kb_id, version, view.q || "", view.key || "", view.upper || "", $("#gp-limit").value]);
  const same = state.gp.viewKey === viewKey && !!state.gp.data;
  if (same && !force) return;
  if (same && force && Date.now() - state.gp.at < 5000) return;
  // No new request while one for the same view is on its way: a large KB takes a dozen seconds or more the first
  // time and polling comes round every few seconds, so every round used to send another identical request
  if (state.gp.loading === viewKey && Date.now() - state.gp.loadingAt < GP_LOAD_PATIENCE_MS) return;
  const seq = ++gpReqSeq;
  state.gp.loading = viewKey; state.gp.loadingAt = Date.now();
  if (!same) renderGraphPreview();                          // until the new view's data arrives draw "Loading...", not the previous view's graph
  try {
    const params = new URLSearchParams({ limit: $("#gp-limit").value, q: view.q || "", key: view.key || "", upper: view.upper || "" });
    const d = await api(`/kbs/${encodeURIComponent(kb.kb_id)}/graph_preview?${params}`);
    if (seq !== gpReqSeq) return;                          // stale response: a newer request was sent since
    if (selectedEntry()?.kb_id !== kb.kb_id) return;
    state.gp.loading = null;
    state.gp = { ...state.gp, kbId: kb.kb_id, version: d.version, viewKey, data: d, pos: null, sim: null, at: Date.now(), hover: null,
                 error: null, cam: { k: 1, tx: 0, ty: 0 } };
    renderGraphPreview();
  } catch (e) {
    if (seq !== gpReqSeq || selectedEntry()?.kb_id !== kb.kb_id) return;
    state.gp.loading = null;
    state.gp.error = t("载入失败: ") + t(e.message);
    renderGraphPreview();
  }
}
let gpReqSeq = 0;
const GP_PENDING = "pending";            // the KB was just switched and the preview request has not been sent yet
const GP_LOAD_PATIENCE_MS = 60000;       // a request in flight that has not come back after this long is sent again by the next poll

// Barnes-Hut quadtree: each cell records mass and coordinate sums; a distant cell counts as one point for
// repulsion
function gpQuadTree(pos, n) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    const p = pos[i];
    if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
  }
  const mk = (x, y, s) => ({ x, y, s, kids: null, pt: -1, m: 0, sx: 0, sy: 0 });
  const root = mk(minX, minY, Math.max(maxX - minX, maxY - minY) + 1e-3);
  const quad = (c, p) => (p.x >= c.x + c.s / 2 ? 1 : 0) + (p.y >= c.y + c.s / 2 ? 2 : 0);
  const insert = (c, i, depth) => {
    const p = pos[i];
    c.m++; c.sx += p.x; c.sy += p.y;
    if (c.kids === null) {
      if (c.pt < 0) { c.pt = i; return; }
      if (depth > 24) return;                            // coincident points: record the mass only, no further subdivision
      const old = c.pt, h = c.s / 2;
      c.pt = -1;
      c.kids = [mk(c.x, c.y, h), mk(c.x + h, c.y, h), mk(c.x, c.y + h, h), mk(c.x + h, c.y + h, h)];
      insert(c.kids[quad(c, pos[old])], old, depth + 1);
    }
    insert(c.kids[quad(c, p)], i, depth + 1);
  };
  for (let i = 0; i < n; i++) insert(root, i, 0);
  return root;
}

// Cluster anchors: one anchor per upper ontology class, arranged in a ring; bigger classes get wider
// sectors (angle by sqrt(node count)). The legend order is the order around the ring; with a single class
// there is no clustering (single) and it falls back to plain force-directed layout
function gpGroupAnchors(nodes, palette, k) {
  const counts = new Map();
  for (const nd of nodes) { const u = nd.upper || ""; counts.set(u, (counts.get(u) || 0) + 1); }
  const order = [...palette.filter(u => counts.has(u)), ...[...counts.keys()].filter(u => !palette.includes(u))];
  const byUpper = new Map();
  const weights = order.map(u => Math.sqrt(counts.get(u)));
  const total = weights.reduce((s, w) => s + w, 0) || 1;
  const Rc = k * Math.sqrt(Math.max(nodes.length, 1)) * 1.05;
  let acc = 0;
  order.forEach((u, i) => {
    const a = -Math.PI / 2 + ((acc + weights[i] / 2) / total) * Math.PI * 2;
    acc += weights[i];
    byUpper.set(u, { upper: u, x: order.length > 1 ? Rc * Math.cos(a) : 0, y: order.length > 1 ? Rc * Math.sin(a) : 0, n: 0, count: counts.get(u) });
  });
  return { order, byUpper, list: [...byUpper.values()], single: order.length < 2 };
}

// Force-directed layout: returns a simulation object; tick() runs one round, run(ms) runs several within a
// time budget, fit(w, h) scales the current coordinates into the canvas.
// opts.cluster: cluster by upper ontology -- nodes start near their own anchor and an extra pull towards
// the anchor acts during iteration, so each class forms a cluster and classes stay apart, making the
// structure visible at a glance (2026-09-07; the 3D proposal was rejected, this is the replacement)
function layoutGraph(nodes, edges, opts = {}) {
  const n = nodes.length;
  const k = 40;                                            // ideal edge length (layout-space unit, scaled at the end)
  const R = k * Math.sqrt(Math.max(n, 1)) * 0.5;           // initial radius grows with the node count, so thousands of nodes do not start as one blob
  const groups = opts.cluster ? gpGroupAnchors(nodes, opts.palette || [], k) : null;
  const anchorOf = groups && !groups.single ? nodes.map(nd => groups.byUpper.get(nd.upper || "")) : null;
  const pos = nodes.map((_, i) => {
    if (anchorOf) {
      const g = anchorOf[i], j = g.n++;                    // index within the group: golden-angle spiral out from the anchor
      const rr = k * 0.45 * Math.sqrt(j + 1), a = j * 2.399963;
      return { x: g.x + rr * Math.cos(a), y: g.y + rr * Math.sin(a), vx: 0, vy: 0 };
    }
    const a = (i / Math.max(n, 1)) * Math.PI * 2 + (i % 3) * 0.01, r = R * (0.55 + 0.45 * ((i % 7) / 6));
    return { x: r * Math.cos(a), y: r * Math.sin(a), vx: 0, vy: 0 };
  });
  const idx = new Map(nodes.map((nd, i) => [nd.key, i]));
  const links = edges.map(e => [idx.get(e.source), idx.get(e.target)]).filter(([a, b]) => a != null && b != null && a !== b);
  const deg = new Array(n).fill(0);
  for (const [a, b] of links) { deg[a]++; deg[b]++; }
  const iters = n > 600 ? 300 : n > 120 ? 260 : 320;
  const k2 = k * k, theta2 = 0.8 * 0.8;
  let it = 0;
  const sim = {
    nodes, pos, groups, done: n === 0,
    tick() {
      if (this.done) return;
      const t = 1 - it / iters;
      const tree = n > GP_BH_MIN ? gpQuadTree(pos, n) : null;
      for (let i = 0; i < n; i++) {
        const p = pos[i];
        let fx = 0, fy = 0;
        if (tree) {
          const walk = c => {
            if (c.m === 0 || (c.kids === null && c.pt === i)) return;
            const cx = c.sx / c.m, cy = c.sy / c.m;
            const dx = p.x - cx, dy = p.y - cy, d2 = dx * dx + dy * dy + 0.01;
            if (c.kids === null || c.s * c.s < theta2 * d2) {
              const f = c.m * k2 / d2;                    // repulsion, inverse square; the whole cell as one point
              fx += dx * f; fy += dy * f;
            } else for (const kid of c.kids) walk(kid);
          };
          walk(tree);
        } else {
          for (let j = 0; j < n; j++) {
            if (i === j) continue;
            const dx = p.x - pos[j].x, dy = p.y - pos[j].y;
            const d2 = dx * dx + dy * dy + 0.01;
            const f = k2 / d2;
            fx += dx * f; fy += dy * f;
          }
        }
        if (anchorOf) {
          // Clustering: pull towards the group's own anchor, plus a very weak global centering (so the whole
          // ring does not drift away slowly)
          // The higher the degree the tighter the pull: hub entities connect several classes, and without
          // the tighter pull cross-class edges drag them into the middle into one blob
          const g = anchorOf[i], gc = 0.06 + Math.min(deg[i], 30) * 0.003;
          fx += (g.x - p.x) * gc; fy += (g.y - p.y) * gc;
          fx -= p.x * 0.003; fy -= p.y * 0.003;
        } else {
          // Centering: the farther from the center the stronger the pull; high-degree nodes sit closer to
          // the middle
          const g = 0.015 + Math.min(deg[i], 20) * 0.002;
          fx -= p.x * g; fy -= p.y * g;
        }
        p.vx = fx; p.vy = fy;
      }
      for (const [a, b] of links) {
        const dx = pos[b].x - pos[a].x, dy = pos[b].y - pos[a].y;
        const d = Math.sqrt(dx * dx + dy * dy) || 0.01;
        const cross = anchorOf && anchorOf[a] !== anchorOf[b];
        const f = (d - k) * (cross ? 0.02 : 0.06);           // spring back to the ideal length; cross-class edges are much weaker, so two clusters are not pulled together
        // When clustering, normalize by degree: the combined pull of a hub's dozens of edges must not exceed
        // the pull of its own anchor
        const fa = anchorOf ? f / Math.sqrt(deg[a] || 1) : f, fb = anchorOf ? f / Math.sqrt(deg[b] || 1) : f;
        pos[a].vx += dx / d * fa; pos[a].vy += dy / d * fa;
        pos[b].vx -= dx / d * fb; pos[b].vy -= dy / d * fb;
      }
      const step = 10 * t + 0.5;
      for (const p of pos) {
        const v = Math.sqrt(p.vx * p.vx + p.vy * p.vy) || 1;
        const s = Math.min(v, step) / v;
        p.x += p.vx * s; p.y += p.vy * s;
      }
      if (++it >= iters) this.done = true;
    },
    run(budgetMs) {
      const t0 = performance.now();
      while (!this.done && performance.now() - t0 < budgetMs) this.tick();
    },
    // Scale into the canvas: the bounding box uses the 1% / 99% quantiles, so one far-off isolated node
    // does not squeeze the whole graph into the middle; nodes beyond the quantiles land outside the canvas
    // and become visible by dragging / zooming out. Scale x and y separately, leaving room on the right
    // for labels
    fit(w, h) {
      const pad = 36, padR = 140;
      const q = (arr, f) => { const a = [...arr].sort((x, y) => x - y); return a[Math.min(a.length - 1, Math.max(0, Math.round((a.length - 1) * f)))]; };
      const xs = pos.map(p => p.x), ys = pos.map(p => p.y);
      const minX = q(xs, 0.01), maxX = q(xs, 0.99), minY = q(ys, 0.01), maxY = q(ys, 0.99);
      const sx = (w - pad - padR) / Math.max(maxX - minX, 1), sy = (h - 2 * pad) / Math.max(maxY - minY, 1);
      const out = {};
      nodes.forEach((nd, i) => { out[nd.key] = { x: pad + (pos[i].x - minX) * sx, y: pad + (pos[i].y - minY) * sy }; });
      return out;
    },
  };
  return sim;
}

// Dot radius: the bigger the graph the smaller the base (a whole version has thousands of nodes), higher
// degree bigger; follows zoom by square root, so 4x zoom only doubles the dot
function gpRadius(n, degMax, k, total) {
  const base = total > 600 ? 2.4 : total > 250 ? 3.2 : 4, bonus = total > 600 ? 7 : 10;
  return Math.max(1.5, Math.min(22, (base + Math.sqrt(n.degree / degMax) * bonus) * Math.sqrt(k)));
}

// Cluster labels: each class writes its name right above its own cluster (counts are in the legend), same
// colour as the legend, drawn beneath edges and nodes
function gpDrawGroupLabels(ctx, sim, pos, cam, palette, w, h) {
  if (!sim.groups || sim.groups.single) return;
  ctx.font = "600 13px -apple-system, 'PingFang SC', 'Helvetica Neue', sans-serif";
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  // The label sits outside its own cluster along the class's sector direction: cluster center + direction
  // × (cluster radius + a margin), so classes do not overlap
  const all = sim.nodes.map(nd => pos[nd.key]).filter(Boolean);
  const gx = all.reduce((t, p) => t + p.x, 0) / (all.length || 1), gy = all.reduce((t, p) => t + p.y, 0) / (all.length || 1);
  for (const g of sim.groups.list) {
    const pts = sim.nodes.filter(nd => (nd.upper || "") === g.upper).map(nd => pos[nd.key]).filter(Boolean);
    if (pts.length < 2) continue;
    const cx = pts.reduce((t, p) => t + p.x, 0) / pts.length, cy = pts.reduce((t, p) => t + p.y, 0) / pts.length;
    const rad = Math.max(...pts.map(p => Math.hypot(p.x - cx, p.y - cy)));
    let ux = cx - gx, uy = cy - gy; const len = Math.hypot(ux, uy) || 1; ux /= len; uy /= len;
    ctx.globalAlpha = 0.6; ctx.fillStyle = gpColor(g.upper, palette);
    // Touching the canvas edge gets clipped: clamp back by text width (a fixed 30px used to cut "process"
    // on the left to "proc")
    const text = gpUpperText(g.upper);
    const half = ctx.measureText(text).width / 2 + 6;
    const lx = Math.max(half, Math.min(w - half, (cx + ux * rad) * cam.k + cam.tx + ux * 22));
    const ly = Math.max(10, Math.min(h - 10, (cy + uy * rad) * cam.k + cam.ty + uy * 18));
    ctx.fillText(text, lx, ly);
  }
  ctx.globalAlpha = 1; ctx.textAlign = "start";
}

// The cluster switch is remembered in the browser, on by default; off is the original plain force-directed
// layout
function gpClusterOn() { try { return localStorage.getItem("gp.cluster") !== "0"; } catch (e) { return true; } }
function gpSyncClusterButton() { $("#gp-cluster")?.classList.toggle("on", gpClusterOn()); }

function gpCanvasText(ctx, w, h, text) {
  ctx.font = "13px -apple-system, 'PingFang SC', 'Helvetica Neue', sans-serif";
  ctx.fillStyle = "#8a8f9c"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(text, w / 2, h / 2);
  ctx.textAlign = "start";
}

function renderGraphPreview() {
  const canvas = $("#gp-canvas"), legend = $("#gp-legend");
  if (!canvas) return;
  const kb = selectedEntry();
  const d = state.gp.data;
  const ctx = canvas.getContext("2d");
  const cssW = canvas.clientWidth || 800, cssH = canvas.clientHeight || 540, dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
    canvas.width = Math.round(cssW * dpr); canvas.height = Math.round(cssH * dpr);
    state.gp.pos = null;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  renderGraphNav();
  if (state.gp.error) { gpCanvasText(ctx, cssW, cssH, state.gp.error); setHtml(legend, ""); return; }
  if (!d || !d.nodes || !d.nodes.length) {
    gpCanvasText(ctx, cssW, cssH, !kb || !kb.graph_status || kb.graph_status === "disabled" ? t("开启并建成图谱后,这里画出现行版本的实体与关系")
      : state.gp.loading ? t("载入中…")
      : d && d.version ? t("这一层没有可画的实体") : t("还没有建成的版本"));
    setHtml(legend, "");
    return;
  }
  const palette = Object.keys(d.upper_counts || {});
  if (!state.gp.sim) { state.gp.sim = layoutGraph(d.nodes, d.edges, { cluster: gpClusterOn(), palette }); state.gp.sim.run(40); state.gp.pos = null; }
  const sim = state.gp.sim;
  if (!state.gp.pos) state.gp.pos = sim.fit(cssW, cssH);
  const pos = state.gp.pos;
  const { k, tx, ty } = state.gp.cam;
  gpDrawGroupLabels(ctx, sim, pos, state.gp.cam, palette, cssW, cssH);
  const degMax = Math.max(1, ...d.nodes.map(n => n.degree));
  const hover = state.gp.hover;
  const near = new Set();
  if (hover) for (const e of d.edges) { if (e.source === hover) near.add(e.target); if (e.target === hover) near.add(e.source); }
  // Edges: the denser the fainter (with hundreds of lines stacked 0.28 is a grey blob); on hover only the
  // connected edges light up
  const baseAlpha = d.edges.length > 900 ? 0.12 : d.edges.length > 400 ? 0.18 : 0.28;
  for (const e of d.edges) {
    const a = pos[e.source], b = pos[e.target];
    if (!a || !b) continue;
    const lit = hover && (e.source === hover || e.target === hover);
    ctx.strokeStyle = lit ? "rgba(99,102,241,.85)" : (hover ? "rgba(120,120,140,.08)" : `rgba(120,120,140,${baseAlpha})`);
    ctx.lineWidth = lit ? 1.6 : 1;
    ctx.beginPath(); ctx.moveTo(a.x * k + tx, a.y * k + ty); ctx.lineTo(b.x * k + tx, b.y * k + ty); ctx.stroke();
  }
  // Draw all dots first and the labels afterwards: no label gets covered by a later dot
  const drawn = [];
  for (const n of d.nodes) {
    const p = pos[n.key]; if (!p) continue;
    const x = p.x * k + tx, y = p.y * k + ty;
    if (x < -40 || x > cssW + 40 || y < -40 || y > cssH + 40) continue;
    const r = gpRadius(n, degMax, k, d.nodes.length);
    const dim = hover && n.key !== hover && !near.has(n.key);
    ctx.globalAlpha = dim ? 0.25 : (n.boilerplate && n.key !== hover ? 0.45 : 1);   // boilerplate entities are drawn faded
    ctx.fillStyle = gpColor(n.upper, palette);
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    if (n.key === d.focus || n.key === hover) { ctx.strokeStyle = "#111"; ctx.lineWidth = 1.5; ctx.stroke(); }
    drawn.push({ n, x, y, r, dim });
  }
  ctx.globalAlpha = 1;
  // Labels: small graphs label everything; big graphs label only the highest-degree nodes in the viewport,
  // more when zoomed in. The focus entity, the hovered entity and its neighbours are always labelled.
  // Same-name entities (one per document) carry a document label; overlapping labels are skipped (higher
  // degree first) so they do not smear into one
  const inView = drawn.filter(o => o.x > -20 && o.x < cssW + 20 && o.y > -12 && o.y < cssH + 12).map(o => o.n);
  const cap = d.nodes.length <= 90 ? d.nodes.length : Math.round(45 * k * k);
  const labelKeys = new Set(inView.sort((a, b) => b.degree - a.degree).slice(0, cap).map(n => n.key));
  const titleCount = new Map();
  for (const n of d.nodes) titleCount.set(n.title, (titleCount.get(n.title) || 0) + 1);
  const dupTitles = new Set([...titleCount].filter(([, c]) => c > 1).map(([t]) => t));
  ctx.font = "11.5px -apple-system, 'PingFang SC', 'Helvetica Neue', sans-serif";
  ctx.textBaseline = "middle";
  const boxes = [];
  const overlaps = (bx) => boxes.some(o => bx.x < o.x + o.w && bx.x + bx.w > o.x && bx.y < o.y + o.h && bx.y + bx.h > o.y);
  const order = drawn.filter(o => labelKeys.has(o.n.key) || o.n.key === hover || o.n.key === d.focus || near.has(o.n.key))
    .sort((a, b) => (b.n.key === hover || b.n.key === d.focus) - (a.n.key === hover || a.n.key === d.focus) || b.n.degree - a.n.degree);
  for (const o of order) {
    const label = gpLabelOf(o.n, dupTitles);
    const tw = ctx.measureText(label).width;
    const bx = { x: o.x + o.r + 3, y: o.y - 8, w: tw + 4, h: 16 };
    const must = o.n.key === hover || o.n.key === d.focus;
    if (!must && overlaps(bx)) continue;
    boxes.push(bx);
    ctx.globalAlpha = o.dim ? 0.35 : 1;
    ctx.fillStyle = must ? "rgba(255,255,255,.95)" : "rgba(255,255,255,.75)";
    ctx.fillRect(bx.x, bx.y, bx.w, bx.h);
    ctx.fillStyle = "#222";
    if (must) ctx.font = "600 11.5px -apple-system, 'PingFang SC', 'Helvetica Neue', sans-serif";
    ctx.fillText(label, bx.x + 2, o.y);
    if (must) ctx.font = "11.5px -apple-system, 'PingFang SC', 'Helvetica Neue', sans-serif";
  }
  ctx.globalAlpha = 1;
  // Legend (click a class to see only it, click again to clear): Chinese + English name, the English name
  // matches the parent types on the config page
  const view = gpView();
  const chips = palette.map(u => `<span class="chip ${view.upper === u ? "on" : ""}" data-upper="${esc(u)}" title="${esc(u || t("未分类"))}"><i style="background:${gpColor(u, palette)}"></i>${esc(t(gpUpperZh(u)) || u || t("未分类"))}${u && gpUpperZh(u) && I18N.lang !== "en" ? ` <span class="en">${esc(u)}</span>` : ""} ${d.upper_counts[u]}</span>`).join("");
  if (setHtml(legend, chips)) $$("#gp-legend .chip").forEach(el => el.addEventListener("click", () => {
    const cur = gpView();
    gpGo({ upper: cur.upper === el.dataset.upper ? "" : el.dataset.upper });
  }));
  if (kb && (kb.graph_status === "failed" || kb.graph_status === "stopped")) {
    ctx.font = "12px -apple-system, 'PingFang SC', 'Helvetica Neue', sans-serif";
    ctx.fillStyle = "#8a8f9c"; ctx.textBaseline = "alphabetic";
    ctx.fillText(t("画的是现行版本;最近一次建图没有完成"), 12, cssH - 12);
  }
  // Layout not settled: keep computing and drawing next frame
  if (!sim.done && !state.gp.raf) state.gp.raf = requestAnimationFrame(() => {
    state.gp.raf = 0;
    if (state.gp.sim !== sim) return;
    if (!sim.done) { sim.run(14); state.gp.pos = null; }
    renderGraphPreview();
  });
}

function gpNodeAt(x, y) {
  const d = state.gp.data, pos = state.gp.pos;
  if (!d || !pos) return null;
  const { k, tx, ty } = state.gp.cam;
  const degMax = Math.max(1, ...d.nodes.map(n => n.degree));
  let best = null, bestD = 1e9;
  for (const n of d.nodes) {
    const p = pos[n.key]; if (!p) continue;
    const r = gpRadius(n, degMax, k, d.nodes.length) + 4;
    const dd = (p.x * k + tx - x) ** 2 + (p.y * k + ty - y) ** 2;
    if (dd <= r * r && dd < bestD) { best = n; bestD = dd; }
  }
  return best;
}

const gpCanvas = $("#gp-canvas");
const gpPoint = ev => { const rect = gpCanvas.getBoundingClientRect(); return { x: ev.clientX - rect.left, y: ev.clientY - rect.top, rect }; };

gpCanvas.addEventListener("mousemove", ev => {
  const { x, y, rect } = gpPoint(ev);
  const n = state.gp.drag ? null : gpNodeAt(x, y);
  const tip = $("#gp-tip");
  const key = n ? n.key : null;
  gpCanvas.style.cursor = state.gp.drag ? (state.gp.drag.node ? "move" : "grabbing") : (n ? "pointer" : "grab");
  if (key !== state.gp.hover) { state.gp.hover = key; renderGraphPreview(); }
  if (!n) { tip.style.display = "none"; return; }
  const d = state.gp.data;
  const byKey = new Map(d.nodes.map(x => [x.key, x]));
  const rels = d.edges.filter(e => e.source === n.key || e.target === n.key).slice(0, 6).map(e => {
    const other = byKey.get(e.source === n.key ? e.target : e.source);
    return `${e.source === n.key ? "→" : "←"} ${esc(e.predicate)} ${esc(other ? other.title : "")}`;
  });
  const total = d.edges.filter(e => e.source === n.key || e.target === n.key).length;
  const sameName = d.nodes.filter(x => x.title === n.title && x.key !== n.key).length;
  const where = n.scope
    ? `<span class="muted">${t("文档内实体 · 来自 {0}", esc(n.doc || t("这份文档")))}${sameName ? t(" · 另有 {0} 个同名实体来自其他文档", sameName) : ""}</span>`
    : `<span class="muted">${t("全局实体 · 出现在 {0} 份文档", n.docs)}</span>`;
  tip.innerHTML = `<b>${esc(n.title)}</b><span class="muted">${esc(n.type)}${n.upper && n.upper !== n.type ? ` · ${esc(gpUpperText(n.upper))}` : ""} · ${t("{0} 条关系 · 出现 {1} 次", n.degree, n.frequency)}${n.boilerplate ? t(" · 样板实体") : ""}</span>`
    + where
    + (n.description ? `<div style="margin-top:4px">${esc(n.description)}</div>` : "")
    + (rels.length ? `<div class="muted" style="margin-top:4px">${rels.join("<br>")}${total > rels.length ? `<br>… ${t("共 {0} 条,点它看全部", total)}` : ""}</div>` : "");
  tip.style.display = "block";
  const tx = Math.min(x + 14, rect.width - 350), ty = Math.min(y + 14, rect.height - 140);
  tip.style.left = Math.max(0, tx) + "px"; tip.style.top = Math.max(0, ty) + "px";
});
gpCanvas.addEventListener("mouseleave", () => { $("#gp-tip").style.display = "none"; if (state.gp.hover) { state.gp.hover = null; renderGraphPreview(); } });

// Mouse down: remember the start point and the entity under it; moving more than 4px is a drag (an entity
// moves, empty space pans), releasing without moving is a click (enter the next level centred on it)
gpCanvas.addEventListener("mousedown", ev => {
  if (ev.button !== 0 || !state.gp.data) return;
  const { x, y } = gpPoint(ev);
  const n = gpNodeAt(x, y);
  state.gp.drag = { x, y, x0: x, y0: y, node: n ? n.key : null, moved: false };
  ev.preventDefault();
});
window.addEventListener("mousemove", ev => {
  const dg = state.gp.drag;
  if (!dg) return;
  const { x, y } = gpPoint(ev);
  const dx = x - dg.x, dy = y - dg.y;
  dg.x = x; dg.y = y;
  if (!dg.moved && Math.hypot(x - dg.x0, y - dg.y0) < 4) return;
  dg.moved = true;
  $("#gp-tip").style.display = "none";
  if (dg.node) {
    if (state.gp.sim) state.gp.sim.done = true;          // once moved by hand, stop the automatic layout
    const p = state.gp.pos && state.gp.pos[dg.node];
    if (p) { p.x += dx / state.gp.cam.k; p.y += dy / state.gp.cam.k; }
  } else { state.gp.cam.tx += dx; state.gp.cam.ty += dy; }
  renderGraphPreview();
});
window.addEventListener("mouseup", () => {
  const dg = state.gp.drag;
  if (!dg) return;
  state.gp.drag = null;
  gpCanvas.style.cursor = "grab";
  if (dg.moved || !dg.node || !state.gp.data) return;
  if (Date.now() - (state.gp.clickAt || 0) < 350) return;   // a double-click is not two level entries
  state.gp.clickAt = Date.now();
  const n = state.gp.data.nodes.find(x => x.key === dg.node);
  if (n) gpGo({ q: n.title, key: n.key });
});
gpCanvas.addEventListener("dblclick", ev => {
  const { x, y } = gpPoint(ev);
  if (gpNodeAt(x, y)) return;
  state.gp.cam = { k: 1, tx: 0, ty: 0 };                  // double-click on empty space: reset zoom and pan
  renderGraphPreview();
});
gpCanvas.addEventListener("wheel", ev => {
  if (!state.gp.data || !state.gp.pos) return;
  ev.preventDefault();
  const { x, y } = gpPoint(ev);
  const cam = state.gp.cam;
  const delta = ev.deltaMode === 1 ? ev.deltaY * 16 : ev.deltaY;
  const k2 = Math.max(0.15, Math.min(12, cam.k * Math.exp(-delta * 0.0015)));
  const f = k2 / cam.k;
  cam.tx = x - (x - cam.tx) * f; cam.ty = y - (y - cam.ty) * f; cam.k = k2;   // zoom about the mouse position
  $("#gp-tip").style.display = "none";
  renderGraphPreview();
}, { passive: false });

// ── Touch (2026-09-11): one finger drags an entity / pans on empty space, two fingers pinch to zoom (about
//    the midpoint, and moving the midpoint pans too), a tap on an entity enters the next level, a double
//    tap on empty space resets. CSS touch-action:none stops the browser from zooming the whole page;
//    preventDefault in touchstart also suppresses the synthesized mouse events that would follow, so one
//    gesture never runs both code paths.
let gpTouch = null;
const gpTouchPoint = tch => { const rect = gpCanvas.getBoundingClientRect(); return { x: tch.clientX - rect.left, y: tch.clientY - rect.top }; };
gpCanvas.addEventListener("touchstart", ev => {
  if (!state.gp.data) return;
  ev.preventDefault();
  $("#gp-tip").style.display = "none";
  if (ev.touches.length === 1) {
    const { x, y } = gpTouchPoint(ev.touches[0]);
    const n = gpNodeAt(x, y);
    gpTouch = { mode: "drag", x, y, x0: x, y0: y, node: n ? n.key : null, moved: false };
  } else if (ev.touches.length >= 2) {
    const a = gpTouchPoint(ev.touches[0]), b = gpTouchPoint(ev.touches[1]);
    gpTouch = { mode: "pinch", dist: Math.hypot(a.x - b.x, a.y - b.y), cx: (a.x + b.x) / 2, cy: (a.y + b.y) / 2, moved: true };
  }
}, { passive: false });
gpCanvas.addEventListener("touchmove", ev => {
  if (!gpTouch || !state.gp.pos) return;
  ev.preventDefault();
  const cam = state.gp.cam;
  if (gpTouch.mode === "pinch" && ev.touches.length >= 2) {
    const a = gpTouchPoint(ev.touches[0]), b = gpTouchPoint(ev.touches[1]);
    const dist = Math.hypot(a.x - b.x, a.y - b.y), cx = (a.x + b.x) / 2, cy = (a.y + b.y) / 2;
    cam.tx += cx - gpTouch.cx; cam.ty += cy - gpTouch.cy;                       // pan the graph by as much as the midpoint moved
    const k2 = Math.max(0.15, Math.min(12, cam.k * (dist / Math.max(1, gpTouch.dist))));
    const f = k2 / cam.k;
    cam.tx = cx - (cx - cam.tx) * f; cam.ty = cy - (cy - cam.ty) * f; cam.k = k2;   // zoom about the midpoint between the two fingers
    gpTouch.dist = dist; gpTouch.cx = cx; gpTouch.cy = cy;
    renderGraphPreview();
    return;
  }
  if (gpTouch.mode !== "drag" || ev.touches.length !== 1) return;
  const { x, y } = gpTouchPoint(ev.touches[0]);
  const dx = x - gpTouch.x, dy = y - gpTouch.y;
  gpTouch.x = x; gpTouch.y = y;
  if (!gpTouch.moved && Math.hypot(x - gpTouch.x0, y - gpTouch.y0) < 6) return;
  gpTouch.moved = true;
  if (gpTouch.node) {
    if (state.gp.sim) state.gp.sim.done = true;
    const p = state.gp.pos[gpTouch.node];
    if (p) { p.x += dx / cam.k; p.y += dy / cam.k; }
  } else { cam.tx += dx; cam.ty += dy; }
  renderGraphPreview();
}, { passive: false });
gpCanvas.addEventListener("touchend", ev => {
  const tc = gpTouch;
  if (!tc) return;
  if (ev.touches.length > 0) {                               // one finger lifted, the other still down: continue as a single-finger drag
    const { x, y } = gpTouchPoint(ev.touches[0]);
    gpTouch = { mode: "drag", x, y, x0: x, y0: y, node: null, moved: true };
    return;
  }
  gpTouch = null;
  if (tc.mode !== "drag" || tc.moved || !state.gp.data) return;
  const now = Date.now();
  if (!tc.node) {                                            // double tap on empty space: reset zoom and pan
    if (now - (state.gp.tapAt || 0) < 350) { state.gp.cam = { k: 1, tx: 0, ty: 0 }; state.gp.tapAt = 0; renderGraphPreview(); }
    else state.gp.tapAt = now;
    return;
  }
  state.gp.tapAt = 0;
  if (now - (state.gp.clickAt || 0) < 350) return;
  state.gp.clickAt = now;
  const n = state.gp.data.nodes.find(x => x.key === tc.node);
  if (n) gpGo({ q: n.title, key: n.key });
}, { passive: false });
gpCanvas.addEventListener("touchcancel", () => { gpTouch = null; });

$("#gp-q").addEventListener("keydown", ev => { if (ev.key === "Enter") gpGo({ q: ev.target.value.trim() }); });
$("#gp-limit").addEventListener("change", () => gpLoadView());     // count changed: re-fetch the same level in place (no "whole graph" state since 2026-09-12)
$("#gp-merges")?.addEventListener("click", () => openMergesDrawer());
$("#gp-cluster")?.addEventListener("click", () => {
  try { localStorage.setItem("gp.cluster", gpClusterOn() ? "0" : "1"); } catch (e) { /* no storage in private mode: this session uses the default */ }
  gpSyncClusterButton();
  state.gp.sim = null; state.gp.pos = null; state.gp.cam = { k: 1, tx: 0, ty: 0 };   // lay out again
  renderGraphPreview();
});
gpSyncClusterButton();
$("#gp-back").addEventListener("click", () => gpStep(-1));
$("#gp-fwd").addEventListener("click", () => gpStep(1));
window.addEventListener("resize", () => { if (state.tab === "graph" && state.gp.data) { state.gp.pos = null; renderGraphPreview(); } });

/* ── service status drawer ─────────────────────────────────────── */
const SERVICES = [
  ["数据库服务",     "database",         h => h.qdrant && h.opensearch && h.neo4j !== false],   // vector store + keyword index + graph store (null when not configured, not counted as broken)
  ["解析服务",       "mineru",           h => h.mineru],
  ["文本向量服务",   "embedding",        h => h.embedding],
  ["跨模态向量服务", "visual_embedding", h => h.visual_embedding],  // null when disabled, the row is filtered out
  ["多模态服务",     "vlm",              h => h.vlm],
  // Text rerank + cross-modal rerank, used only by retrieval; null when neither address is configured, the
  // whole row hidden
  ["重排序服务",     "reranker",         h => h.reranker == null && h.visual_reranker == null ? null
                                              : h.reranker !== false && h.visual_reranker !== false],
];

let healthInFlight = false;
let chunkLimits = null;   // most recent /limits; the overlap bound is recomputed live as max_tokens changes

function setChunkHint(lim) {
  chunkLimits = lim || null;
  const el = $("#cfg-max-hint");
  if (el) {
    el.textContent = lim ? `${lim.max_tokens_min}–${lim.max_tokens_cap}` : "";
  }
  setOverlapHint();
}

// Same rules as limits.normalize_entity_types on the backend: case-insensitive dedup (keeping the first
// spelling seen). The backend normalizes again -- this only keeps the count shown in the UI equal to what
// gets saved.
function normalizeTypes(list) {
  const out = [], seen = new Set();
  for (const raw of list || []) {
    const name = String(raw).trim();
    if (!name) continue;
    const key = name.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(name);
  }
  return out;
}

function setSchemaView(view, { note = "", dirty = false, kbId = "", active = "" } = {}) {
  const versions = ((view && view.versions) || []).map(v => ({
    ...v, entity_types: normalizeTypes(v.entity_types),
    predicates: Array.isArray(v.predicates) ? v.predicates : [],
    parent_types: v.parent_types && typeof v.parent_types === "object" ? v.parent_types : {},
  }));
  // The active argument wins over the view's: when loading the config it preserves the "just extracted,
  // not yet saved" selection.
  const wanted = String(active || (view && view.active) || "");
  state.graphSchema = {
    versions,
    active: versions.some(v => String(v.id) === wanted) ? wanted
      : (versions.length ? String(versions[0].id) : ""),
    // The version the server considers in effect. It does not follow the dropdown -- picking another option
    // is only "intending to switch", builds keep using this version until saved, so deletability depends
    // on it, not on active.
    savedActive: String((view && view.active) || ""),
    note: note || "",
    dirty,
    kbId: String(kbId || ""),
  };
  renderGraphSchema();
}

function activeVersion() {
  const { versions, active } = state.graphSchema;
  return versions.find(v => String(v.id) === active) || null;
}

function versionLabel(v) {
  // Date · model · sample · label count. The fallback entry of a legacy KB has no metadata; say unknown
  // honestly.
  if (v.legacy) return t("当前(无版本信息) · {0} 个标签", v.entity_types.length);
  const when = v.created_at
    ? new Date(v.created_at * 1000).toLocaleString(I18N.locale,
        { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false })
    : t("时间未知");
  const parts = [when, v.model || t("模型未知")];
  if (v.sample_size != null) parts.push(t("采样 {0}", v.sample_size));
  parts.push(t("{0} 个标签", v.entity_types.length));
  if (v.predicates && v.predicates.length) parts.push(t("{0} 个谓词", v.predicates.length));
  // Mark versions the pipeline extracted itself: before a blank graph's first build, or re-extracted before
  // a threshold-triggered full rebuild (based on the previous version + endpoint ledger)
  if (v.origin === "auto_blank") parts.push(t("自动·首次建图"));
  else if (v.origin === "auto_rebuild") parts.push(t("自动·重建前重抽"));
  return parts.join(" · ");
}

// Output language, entity labels, parent types and predicates are all products of "Extract labels", shown
// read-only: to change them, extract again. Switching the version = switching the whole set (labels +
// language + parent types + predicates), so all of them follow the dropdown.
function renderGraphSchema() {
  const { versions, active, note } = state.graphSchema;
  const current = activeVersion();
  const types = current ? current.entity_types : [];

  const sel = $("#cfg-gver");
  if (sel) {
    sel.innerHTML = "";
    if (!versions.length) {
      const opt = document.createElement("option");
      opt.value = ""; opt.textContent = t("尚未抽取");
      sel.appendChild(opt);
      sel.disabled = true;
    } else {
      sel.disabled = false;
      for (const v of versions) {
        const opt = document.createElement("option");
        opt.value = String(v.id);
        opt.textContent = versionLabel(v);
        if (String(v.id) === active) opt.selected = true;
        sel.appendChild(opt);
      }
    }
  }

  const del = $("#cfg-gvdel");
  if (del) {
    // The synthesized "current (no version info)" entry is never persisted and cannot be deleted; the
    // version in effect cannot be deleted either.
    const chosen = versions.find(v => String(v.id) === active);
    const blocked = !chosen || chosen.legacy
      || String(chosen.id) === state.graphSchema.savedActive;
    del.disabled = blocked;
    del.title = !chosen ? t("尚未抽取标签")
      : chosen.legacy ? t("这一条只是当前生效标签的只读视图,不是保存下来的版本")
      : blocked ? t("正在生效的那一版不能删:图就是按它建的")
      : t("删除选中的这一版标签");
  }

  const lang = $("#cfg-glang");
  if (lang) {
    const value = current ? String(current.language || "") : "";
    lang.textContent = value || t("尚未抽取");
    lang.classList.toggle("empty", !value);
  }

  const box = $("#cfg-get");
  if (box) {
    box.innerHTML = "";
    if (!types.length) {
      const ph = document.createElement("span");
      ph.className = "empty";
      ph.textContent = t("尚未抽取,建图将使用全局默认标签");
      box.appendChild(ph);
    } else {
      for (const ty of types) {
        const chip = document.createElement("span");
        chip.className = "chip-t";
        chip.textContent = ty;
        box.appendChild(chip);
      }
    }
  }

  const hint = $("#cfg-get-hint");
  if (hint) {
    const bits = [];
    if (types.length) bits.push(t("{0} 个标签", types.length));
    if (note) bits.push(note);
    if (state.graphSchema.dirty) bits.push(t("未保存:点「保存配置」后下次建图按这一版"));
    hint.textContent = bits.join(" · ");
  }

  // Parent types: the server stores {type: parent} (the limits.normalize_parent_types shape); shown here
  // grouped by parent as "parent -> subtypes...". Old versions without parent types show empty, and the
  // endpoint gating does nothing at build time.
  const parents = $("#cfg-gparents");
  if (parents) {
    parents.innerHTML = "";
    const groups = groupParentTypes(current ? current.parent_types : {});
    const names = Object.keys(groups);
    if (!names.length) {
      const ph = document.createElement("span");
      ph.className = "empty";
      ph.textContent = types.length ? t("这一版没有父类信息(重抽一次会补上);没有父类就不做端点类型检查") : t("尚未抽取");
      parents.appendChild(ph);
    } else {
      for (const name of names) {
        const chip = document.createElement("span");
        chip.className = "chip-p";
        const b = document.createElement("b"); b.textContent = name;
        const sp = document.createElement("span"); sp.textContent = groups[name].join(t("、"));
        chip.append(b, sp);
        parents.appendChild(chip);
      }
    }
  }

  // Predicate table: name / source parents / target parents / description
  const preds = $("#cfg-gpreds");
  if (preds) {
    const list = current ? current.predicates : [];
    if (!list.length) {
      preds.innerHTML = `<div class="empty-state" style="padding:12px">${types.length ? t("这一版没有谓词表:关系一律记为 related_to(重抽一次会补上)") : t("尚未抽取")}</div>`;
    } else {
      const ends = v => Array.isArray(v) ? v.join(t("、")) : String(v || "");
      // Endpoint ledger (written back into this version when the build finished): endpoint combinations
      // with at least 20 edges and 5% of the predicate. Passed directly at the next resolution and preserved
      // by rule at the next label extraction, no manual edits needed.
      const observed = (current && current.observed) || {};
      const confirmed = {};
      for (const row of (observed.confirmed || [])) {
        (confirmed[row.predicate] ||= []).push(`${row.source_parent}→${row.target_parent} ×${row.count}`);
      }
      const edges = observed.edges || {};
      const account = pr => {
        const items = confirmed[pr.name] || [];
        const used = edges[pr.name] != null ? t("{0} 条", edges[pr.name]) : "";
        if (!items.length) return used ? `<span class="muted">${esc(used)}</span>` : "";
        return `${used ? `<span class="muted">${esc(used)} · </span>` : ""}${items.map(x => esc(x).replace("→", "→<wbr>")).join("<br>")}`;   // on phones, allow a line break after the arrow
      };
      const accountHead = observed.graph_version
        ? `<th title="${t("建图 {0} 结束时写回的端点账:这个谓词实际用了多少条边;够 20 条且占 5% 以上的端点组合列在这里,下次归并直接放行、重抽标签时规则保住", esc(observed.graph_version))}">${t("数据放行")}</th>`
        : `<th title="${t("这一版还没建过图,没有端点账")}">${t("数据放行")}</th>`;
      preds.innerHTML = `<table class="pred-table"><thead><tr><th>${t("谓词")}</th><th>${t("起点父类")}</th><th>${t("终点父类")}</th>${accountHead}<th>${t("说明")}</th></tr></thead><tbody>`
        + list.map(pr => `<tr><td class="mono">${esc(pr.name)}</td><td class="muted">${esc(ends(pr.source_parents) || t("任意"))}</td><td class="muted">${esc(ends(pr.target_parents) || t("任意"))}</td><td class="muted">${account(pr)}</td><td>${esc(pr.description || "")}</td></tr>`).join("")
        + `</tbody></table>`;
    }
  }
}
// {type: parent} -> {parent: [types...]} (parents in order of first appearance). The old shape with array
// values is accepted too.
function groupParentTypes(map) {
  const groups = {};
  for (const [key, value] of Object.entries(map || {})) {
    if (Array.isArray(value)) {
      (groups[key] ||= []).push(...value.map(String));
    } else if (value != null && String(value).trim()) {
      (groups[String(value)] ||= []).push(key);
    }
  }
  return groups;
}

function domainSummary(text, max) {
  const line = String(text || "").split(/\r?\n/).find(l => l.trim()) || "";
  const clean = line.replace(/[*_`#]/g, "").replace(/^\s*Domain\s*[::]\s*/i, "").trim();
  return clean.length > max ? clean.slice(0, max) + "…" : clean;
}

// The hint under the sample size: shows how many chunks this KB actually has to sample from. Range and
// default are not written here -- the input carries its own min/max, and the prefilled value is the
// default 8 when empty.
// Fetched only while the knowledge graph is on: one computation feeds the whole KB's text to the
// tokenizer, and the server caches by directory fingerprint.
async function refreshCorpusHint(kb) {
  const el = $("#cfg-gss-hint");
  if (!el) return;
  if (!kb || kb.kb_id == null || kb.state !== "active" || !$("#cfg-graph").checked) {
    el.textContent = "";
    return;
  }
  const key = state.cfgKey;
  el.textContent = t("统计中…");
  try {
    const r = await api(`/kbs/${encodeURIComponent(kb.kb_id)}/graph_corpus`);
    if (state.cfgKey !== key) return;   // the KB was switched meanwhile; do not write KB A's numbers under KB B's name
    el.textContent = r.documents
      ? t("{0} 份文档共 {1} 个采样片段", r.documents, fmtNum(r.chunks))
      : t("该库还没有入库切片");
  } catch (e) {
    if (state.cfgKey === key) el.textContent = "";
  }
}

function setOverlapHint() {
  // The overlap bound is not a fixed number: it is constrained both by the model's remaining budget and by
  // "overlap must not exceed half", and both vary with max_tokens. A hard-coded "must be below half" would
  // claim more than is actually allowed when max_tokens approaches the model cap, so it is recomputed
  // live from the input.
  const el = $("#cfg-ov-hint");
  if (!el) return;
  const maxTokens = parseInt($("#cfg-max")?.value, 10);
  // With no KB selected, the 400 in the form is just index.html's placeholder value, not any KB's real
  // config; a bound computed from it would be fake, so nothing is shown in that state.
  if (!state.cfgKey || !chunkLimits || !Number.isFinite(maxTokens) || maxTokens <= 0) {
    el.textContent = "";
    return;
  }
  const served = chunkLimits.effective_max_model_len ?? 4096;
  const cap = Math.max(0, Math.min(served - maxTokens, Math.floor(maxTokens / 2)));
  el.textContent = t("小于 {0}", cap);
}

async function refreshHealth() {
  if (healthInFlight) return;      // the 5-second panel interval and the permanent 30-second interval overlap
  healthInFlight = true;
  try { return await _refreshHealth(); } finally { healthInFlight = false; }
}

async function _refreshHealth() {
  const body = $("#health-body");
  try {
    const h = await api("/health");
    state.health = h;      // lets the restart button tell whether the service has turned green
    // Only managed rows (KB_CONSOLE_SERVICES) get a restart button; for the four unmanaged model rows the
    // backend returns null and the row does not appear
    const managed = new Set(Array.isArray(h.managed) ? h.managed : []);
    const rows = SERVICES
      .map(([name, key, fn]) => [name, key, fn(h)])
      .filter(([, , ok]) => ok !== null && ok !== undefined);
    // Timer jobs (scan / parse / graph check / three GCs): only the last result and last time are shown.
    // A failed unit used to be visible only in the journal (health check R6); a job whose last run failed
    // also turns the top-bar dot red
    const timers = Array.isArray(h.timers) ? h.timers : [];
    const allOk = rows.every(([, , ok]) => ok) && timers.every(tm => tm.ok !== false);
    $("#health-dot").className = "dot " + (allOk ? "green" : "red");
    // Timer jobs show only the result, not the execution: dot = last result (failed red / fine green /
    // unknown grey), text = when it last ran. The next time follows from the period, and nobody needs to
    // watch whether it finished or how many rounds it yielded -- all of that is noise; when yielding
    // exceeds the cap the unit turns red by itself, and that is what deserves a look.
    const timerDot = tm => tm.ok === false ? "red" : tm.ok ? "green" : "gray";
    const timerMeta = tm => [tm.last ? t("上次 {0}", tm.last) : "",
                             tm.ok === false ? t("上次失败({0})", tm.result || tm.state || t("未知")) : "",
                             tm.timer_active === false ? t("定时器未启用") : ""].filter(Boolean).join(" · ");
    const busy = !!svcBulk;
    const bulk = managed.size ? `<span class="svc-bulk">
        <button class="sm ghost" id="svc-restart-all" ${busy ? "disabled" : ""}>${t("全部重启")}</button>
        <button class="sm ghost danger" id="svc-stop-all" ${busy ? "disabled" : ""}>${t("全部关闭")}</button></span>` : "";
    body.innerHTML = `<div class="svc-head svc-head-row"><span>${t("系统服务")}</span>${bulk}</div>`
      + rows.map(([name, key, ok]) =>
      `<div class="svc-row"><span class="dot ${ok ? "green" : "red"}"></span>
        <span class="svc-name">${t(name)}</span>`
        // The parse service row carries the backend mode (vlm-engine / pipeline, chosen by the container from
        // the hardware); it is data, not translated
        + (key === "mineru" && h.mineru_backend ? `<span class="svc-meta" title="${esc(h.mineru_backend_source || "")}">${esc(h.mineru_backend)}</span>` : "")
        + (managed.has(key) ? `<button class="svc-restart" data-svc="${key}" data-name="${name}" ${busy || svcRestarting.has(key) ? "disabled" : ""}>${t("重启")}</button>` : "")
        + `</div>`).join("")
      + (timers.length ? `<div class="svc-head">${t("定时任务")}</div>` + timers.map(tm =>
        `<div class="svc-row"><span class="dot ${timerDot(tm)}"></span>
          <span class="svc-name">${esc(t(tm.label))}</span>
          <span class="svc-meta" title="${esc(tm.unit || "")}">${esc(timerMeta(tm))}</span></div>`).join("") : "");
    $$(".svc-restart", body).forEach(b => b.addEventListener("click", onRestartService));
    $("#svc-restart-all")?.addEventListener("click", () => onBulkService("restart"));
    $("#svc-stop-all")?.addEventListener("click", () => onBulkService("stop"));
  } catch {
    $("#health-dot").className = "dot red";
    body.innerHTML = `<div class="svc-row"><span class="dot red"></span><span class="svc-name">${t("控制台后端不可达")}</span></div>`;
  }
}

// "Restart all / Stop all": after sending the command the buttons stay disabled until the health probe
// shows everything settled (restart = all green, stop = all red) or a 2-minute fallback; the panel redraws
// every 5 seconds, so the disabled state lives in svcBulk rather than in the DOM.
let svcBulk = null;
async function onBulkService(action) {
  if (svcBulk) return;
  const parsing = state.overview?.parsing_active;
  // The confirmation text names only managed rows: unmanaged model services are not stopped
  const managedKeys = state.health?.managed || [];
  const managedNames = SERVICES.filter(([, key]) => managedKeys.includes(key)).map(([name]) => t(name))
    .join(I18N.lang === "en" ? ", " : "、");
  const text = action === "stop"
    ? t("关闭全部系统服务?\n\n{0}都会停止;解析队列会等服务回来再继续", managedNames) + (parsing ? t(",进行中的解析步骤会失败并自动重试") : "") + t("。重新打开用「全部重启」。")
    : t("重启全部系统服务?\n\n重启期间各服务短暂不可用") + (parsing ? t(",进行中的解析步骤会失败并自动重试") : "") + t(";状态点全部转绿即恢复完成。");
  const label = t(action === "stop" ? "全部关闭" : "全部重启");
  if (!(await confirmDialog(text, label))) return;
  svcBulk = { action, until: Date.now() + 120000 };
  refreshHealth();
  const path = action === "stop" ? "/services/stop_all" : "/services/restart_all";
  try {
    const r = await api(path, { method: "POST", body: {} });
    toast(t("{0}指令已发出: ", label) + (r.stopping || r.restarting || []).join(", "));
  } catch (e) {
    const again = await confirmDialog(t(e.message) + t("\n\n仍要强制{0}吗?", label), t("强制{0}", label));
    if (!again) { svcBulk = null; refreshHealth(); return; }
    try {
      const r = await api(path, { method: "POST", body: { force: true } });
      toast(t("已强制{0}: ", label) + (r.stopping || r.restarting || []).join(", "));
    } catch (e2) { toast(t("{0}失败: ", label) + t(e2.message)); svcBulk = null; refreshHealth(); return; }
  }
  const settled = () => {
    const h = state.health || {};
    const states = SERVICES.map(([, , fn]) => fn(h)).filter(v => v !== null && v !== undefined);
    return action === "stop" ? states.every(v => v === false) : states.every(v => v === true);
  };
  const poll = async () => {
    await refreshHealth();
    if (settled() || Date.now() > svcBulk.until) { svcBulk = null; refreshHealth(); return; }
    setTimeout(poll, 3000);
  };
  setTimeout(poll, 3000);
}

// Marks for single services being restarted: key -> { until, wentDown }. The panel is redrawn wholesale every few
// seconds, so the buttons' disabled state is kept here rather than in the DOM
const svcRestarting = new Map();
async function onRestartService(ev) {
  const key = ev.target.dataset.svc, name = ev.target.dataset.name;
  if (svcRestarting.has(key)) return;
  const parsing = state.overview?.parsing_active;
  const ok = await confirmDialog(
    t("确认重启「{0}」?\n\n重启期间该服务短暂不可用", t(name)) +
    (parsing ? t(",进行中的解析步骤会失败并自动重试") : "") +
    t(";状态点转绿即恢复完成。"), t("确认重启"));
  if (!ok) return;
  const mark = { until: Date.now() + 30000, wentDown: false };
  svcRestarting.set(key, mark);
  ev.target.disabled = true;
  let sent = false;
  try {
    const r = await api(`/services/${encodeURIComponent(key)}/restart`, { method: "POST", body: {} });
    toast(t("重启指令已发出: ") + r.restarting.join(", "));
    sent = true;
  } catch (e) {
    // The backend refuses once while work (parse or build) is running; show the real reason and let the
    // user decide
    const again = await confirmDialog(t(e.message) + t("\n\n仍要强制重启「{0}」吗?", t(name)), t("强制重启"));
    if (again) {
      try {
        const r = await api(`/services/${encodeURIComponent(key)}/restart`, { method: "POST", body: { force: true } });
        toast(t("已强制重启: ") + r.restarting.join(", "));
        sent = true;
      } catch (e2) { toast(t("重启失败: ") + t(e2.message)); }
    }
  }
  if (!sent) { svcRestarting.delete(key); refreshHealth(); return; }
  // Keep the button disabled until this service is seen going down and turning green again (or a 30-second
  // fallback): docker restart can take tens of seconds, and for the first seconds after the command the probe is
  // still green, so looking only at "is it green now" would release it at once and allow repeated commands.
  // Whether it turned green is judged by this row's own check (the database and rerank rows each cover several services)
  const healthy = (SERVICES.find(([, k]) => k === key) || [])[2] || (() => true);
  const poll = async () => {
    await refreshHealth();
    const up = healthy(state.health || {}) === true;
    if (!up) mark.wentDown = true;
    if ((mark.wentDown && up) || Date.now() > mark.until) { svcRestarting.delete(key); refreshHealth(); return; }
    setTimeout(poll, 3000);
  };
  setTimeout(poll, 2000);
}

/* ── model registry (cards: view mode + edit mode) ─────────────────── */
function llmCard(m) {
  const div = document.createElement("div");
  div.className = "llm-card";
  if (m) renderLlmView(div, m);
  else renderLlmForm(div, null);
  return div;
}

function scheduleNameMarquee(scope) {
  // Measure immediately (reading geometry forces layout, so the result is accurate), then again after a
  // delay: the panel has an entrance animation and the first measurement may land mid-animation. No
  // requestAnimationFrame -- it never fires while the tab is in the background, and the name would never
  // start scrolling.
  applyNameMarquee(scope);
  setTimeout(() => applyNameMarquee(scope), 60);
  setTimeout(() => applyNameMarquee(scope), 400);
}

function applyNameMarquee(scope) {
  const GAP = 36;   // matches --gap in the CSS: the gap between the two copies of the text
  (scope || document).querySelectorAll(".llm-title").forEach(box => {
    const inner = box.querySelector(".llm-title-inner");
    const seg = inner && inner.querySelector(".llm-seg");
    if (!seg) return;
    // Restore the static state before measuring, so the previous clone does not skew this decision
    inner.querySelectorAll(".llm-seg-clone").forEach(node => node.remove());
    box.classList.remove("marquee");
    box.style.removeProperty("--shift");

    // Measure the real rendered width with getBoundingClientRect: seg has flex:0 0 auto, so it is not
    // squeezed by the window and this value is the natural width of the text.
    const textWidth = Math.ceil(seg.getBoundingClientRect().width);
    const overflow = textWidth - Math.floor(box.clientWidth);
    if (overflow <= 2) return;          // fits: no scrolling, stay perfectly still

    // Append a copy: after shifting by "one width + gap" the second copy lands exactly on the original,
    // seamless loop
    const clone = seg.cloneNode(true);
    clone.classList.add("llm-seg-clone");
    clone.setAttribute("aria-hidden", "true");
    inner.appendChild(clone);

    const shift = textWidth + GAP;
    box.style.setProperty("--shift", shift + "px");
    box.style.setProperty("--dur", Math.max(6, shift / 26) + "s");   // constant speed, longer names take longer
    box.classList.add("marquee");
  });
}

function renderLlmView(div, m) {
  div.innerHTML = `
    <div class="llm-top">
      <span class="llm-title" title="${esc(m.name)}"><span class="llm-title-inner"><span class="llm-seg">${esc(m.name)}</span></span></span>
      <span class="llm-actions">
        ${m.has_api_key ? `<span class="llm-chip key">${t("Key 已配置")}</span>` : ""}
        <button class="sm ghost c-edit">${t("编辑")}</button>
        <button class="sm ghost c-del" style="color:var(--red-text)">${t("删除")}</button>
      </span>
    </div>
    <div class="llm-meta">${esc(m.model_id || "")}<br>${esc(m.base_url || "")}${m.protocol === "anthropic" ? " · anthropic" : ""}</div>`;
  scheduleNameMarquee(div);
  div.querySelector(".c-edit").addEventListener("click", () => {
    renderLlmForm(div, m);
    div.querySelector(".c-base").focus();
  });
  div.querySelector(".c-del").addEventListener("click", async () => {
    // Deletion is no longer refused when KBs reference it, but which fields go empty must be spelled out
    // first -- decoupled does not mean silent.
    const used = m.used_by || [];
    const detail = used.length
      ? t("\n\n以下知识图谱在引用它,删除后对应栏位会清空,需要重新选:\n")
        + used.map(u => `· ${u.source_root} — ${(u.steps || []).join(t("、"))}`).join("\n")
      : "";
    if (!(await confirmDialog(t("删除模型「{0}」?", m.name) + detail, t("确认删除")))) return;
    try {
      const r = await api(`/llms/${encodeURIComponent(m.name)}`, { method: "DELETE" });
      const n = (r.cleared || []).length;
      toast(n ? t("已删除;{0} 个知识图谱的对应栏位已清空", n) : t("已删除"));
      refreshLlms();
      renderConfig();          // the model dropdowns in the config section must be redrawn too
    } catch (e) { toast(t("删除失败: ") + t(e.message)); }
  });
}

function renderLlmForm(div, m) {
  const isNew = !m;
  div.innerHTML = `
    <div class="fld"><label class="f">${t("名称")}${isNew ? t("(唯一,保存后不可改)") : ""}</label>
      <input type="text" class="c-name" placeholder="${t("如 deepseek-v3")}" value="${m ? esc(m.name) : ""}" ${isNew ? "" : "readonly"}></div>
    <div class="fld"><label class="f">${t("接口协议")}</label>
      <select class="c-proto">
        <option value="openai" ${!m || m.protocol !== "anthropic" ? "selected" : ""}>${t("OpenAI 兼容(/chat/completions)")}</option>
        <option value="anthropic" ${m && m.protocol === "anthropic" ? "selected" : ""}>${t("Anthropic(/v1/messages)")}</option>
      </select></div>
    <div class="fld"><label class="f">Base URL</label>
      <input type="text" class="c-base mono" placeholder="http://127.0.0.1:8000/v1" value="${m ? esc(m.base_url) : ""}"></div>
    <div class="fld"><label class="f">Model ID</label>
      <input type="text" class="c-model mono" placeholder="${t("如 deepseek-chat")}" value="${m ? esc(m.model_id) : ""}"></div>
    <div class="fld"><label class="f">API Key</label>
      <input class="c-key mono" type="password" autocomplete="new-password" placeholder="${m && m.has_api_key ? t("已配置,留空保持不变") : t("本地服务可留空")}"></div>
    <div class="c-actions">
      <button class="c-cancel">${t("取消")}</button>
      <button class="c-save primary">${t("保存")}</button>
    </div>`;
  const protoSel = div.querySelector(".c-proto"), baseInput = div.querySelector(".c-base");
  const protoHint = () => {
    baseInput.placeholder = protoSel.value === "anthropic"
      ? t("https://api.anthropic.com(不带 /v1,自动补 /v1/messages)")
      : "http://127.0.0.1:8000/v1";
  };
  protoSel.addEventListener("change", protoHint);
  protoHint();
  div.querySelector(".c-cancel").addEventListener("click", () => {
    if (isNew) div.remove();
    else renderLlmView(div, m);
  });
  div.querySelector(".c-save").addEventListener("click", async ev => {
    ev.target.disabled = true;
    ev.target.textContent = t("测试连通性…");
    try {
      await api("/llms", { method: "POST", body: {
        name: div.querySelector(".c-name").value,
        base_url: div.querySelector(".c-base").value,
        model_id: div.querySelector(".c-model").value,
        api_key: div.querySelector(".c-key").value,
        protocol: div.querySelector(".c-proto").value,
      } });
      toast(t("连通性测试通过,已保存"));
      refreshLlms();
    } catch (e) {
      toast(t("保存失败: ") + t(e.message));
      ev.target.disabled = false;
      ev.target.textContent = t("保存");
    }
  });
}

async function refreshLlms() {
  try {
    state.llms = await api("/llms");
    const host = $("#llm-cards");
    host.innerHTML = "";
    if (!state.llms.length) {
      host.innerHTML = `<div class="llm-empty">${t("还没有注册模型。点击下方「添加模型」创建;开启建图的知识库至少需要选择一个。")}</div>`;
    }
    state.llms.forEach(m => host.appendChild(llmCard(m)));
    scheduleNameMarquee(host);
    // The model dropdowns in the config card are rendered from state.llms and rebuilt only when cfgKey
    // changes: a newly registered model would not appear, and a just-deleted one would still show and fail
    // on save with "not in the registry".
    ["#cfg-ge", "#cfg-gs", "#cfg-gt"].forEach(sel => {
      const el = document.querySelector(sel);
      if (!el) return;
      const cur = el.value;
      el.innerHTML = llmOptions(cur, { fallback: true });
      // cur may be the model just deleted: forcing it back would replace the default llmOptions picked
      // with empty
      if ([...el.options].some(o => o.value === cur)) el.value = cur;
    });
  } catch (e) { /* retried the next time the panel opens */ }
}

$("#llm-add").addEventListener("click", () => {
  const host = $("#llm-cards");
  const card = llmCard(null);
  host.appendChild(card);
  card.querySelector(".c-name").focus();
});

/* ── polling ─────────────────────────────────────────────── */
// No need to keep hitting the backend every 2.5 s while the tab is in the background / the screen is
// locked (each poll rebuilds settings, runs a few COUNTs, and while parsing kicks the worker via systemctl
// every 10 s). Catch up immediately on returning to the foreground.
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    // Returning to the foreground does more than one catch-up fetch: the polling chain is rescheduled too.
    // Previously only one fetch was made while the chain's 60-second timer kept running, so the progress
    // bar stalled for up to a minute after coming back (2026-09-08: switch windows and back, the panel did
    // not refresh for ages). Chrome on macOS also counts a page fully covered by another window as hidden.
    tick();
    if (state.activePanel === "health") refreshHealth();
  }
});
let overviewSeq = 0;
async function refreshOverview(forceConfig = false) {
  const seq = ++overviewSeq;
  try {
    const data = await api("/overview");
    // An older response arriving late must not overwrite newer state (button actions and polling run
    // concurrently)
    if (seq !== overviewSeq) return;
    noteDeleteOutcomes(data.kbs);
    state.overview = data;
    state.offlineStreak = 0;
    if (!state.selected && data.kbs.length) {
      // On first load select a KB right away: the one last viewed, else the first enabled one. An empty
      // workspace is pointless.
      const wanted = recall("selected", null);
      const pick = data.kbs.find(k => k.dir === wanted) || data.kbs.find(k => k.state === "active") || data.kbs[0];
      state.selected = pick.dir;
    }
    document.body.classList.remove("backend-offline");
    const parseOff = state.overview.parse_enabled === false;
    document.body.classList.toggle("parse-disabled", parseOff);
    if (forceConfig) state.cfgKey = null;
    renderKbNav();
    renderConfig();
    refreshTab();
  } catch (e) {
    // An unreachable backend used to be completely silent (the UI froze on the last frame); users only found
    // out by clicking a button.
    state.offlineStreak = (state.offlineStreak || 0) + 1;
    if (state.offlineStreak >= 2) document.body.classList.add("backend-offline");
  }
}

let firstTick = true;
let tickTimer = null;
let tickBusy = false;
async function tick() {
  if (tickBusy) return;            // keep a single polling chain: a visibility change reruns this one instead of stacking a second
  tickBusy = true;
  clearTimeout(tickTimer);
  try {
    // The first round loads unconditionally: a tab opened in the background needs data too, or it is blank
    // when switched to
    if (firstTick || !document.hidden) await refreshOverview();
  } finally {
    tickBusy = false;
  }
  firstTick = false;
  const kb = selectedEntry();
  const busy = state.overview?.parsing_active || (kb && kb.graph_status === "running") || state.activePanel !== null
    || (state.overview?.kbs || []).some(k => k.state === "deleting");
  // Stretch to 60 seconds while the page is hidden; on returning to the foreground visibilitychange
  // restarts the chain without waiting for this timer
  clearTimeout(tickTimer);
  tickTimer = setTimeout(tick, document.hidden ? 60000 : (busy ? 2500 : 8000));
}

// Re-measure after a window resize whether names still need to scroll (mobile rotation, window dragging)
let marqueeResizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(marqueeResizeTimer);
  marqueeResizeTimer = setTimeout(() => applyNameMarquee($("#llm-win")), 200);
}, { passive: true });

$("#cfg-max")?.addEventListener("input", setOverlapHint);
refreshHealth();
setInterval(refreshHealth, 30000);
// Load the range hint at startup: the config template stays the same whether nothing is selected / not
// enabled / enabled
api("/limits").then(lim => {
  setChunkHint(lim);
}).catch(() => {});
tick();
