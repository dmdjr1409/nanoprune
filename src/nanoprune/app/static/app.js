"use strict";

// Document contents are always inserted as text nodes (never innerHTML):
// indexed files are untrusted and must not be able to run script in this page.

const TEXT_EXTENSIONS = [".txt", ".md", ".markdown", ".json", ".csv", ".tsv", ".log", ".rst"];
const BINARY_EXTENSIONS = [".docx", ".pdf"];
const MAX_FILE_BYTES = 25 * 1024 * 1024;
const MAX_UPLOAD_BYTES = 60 * 1024 * 1024;
const PAGE_SIZE = 10;
const MAX_RESULTS = 50;
const FILTERS = { 0.3: "Large", 0.5: "Normal", 0.7: "Strict" };
const SAMPLE_QUERIES = [
  { label: "Allergie de M. Dupont", query: "Allergie pénicilline Dupont" },
  { label: "Contre-indication AINS", query: "Contre-indication AINS Kardegic" },
  { label: "Sans allergie : Sophie Bernard", query: "Aucune allergie Sophie Bernard" },
];
// Questions worded differently from the documents, to show what semantic search adds.
const SEMANTIC_SAMPLES = [
  { label: "Que prendre pour une douleur au genou ?", query: "Que prendre pour une douleur au genou ?" },
  { label: "Qui ne doit pas prendre d'anti-inflammatoires ?", query: "Qui ne doit pas prendre d'anti-inflammatoires ?" },
];
const ICONS = {
  open: "M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5",
  copy: "M9 9h10v10H9zM5 15V5h10",
};

// Browser storage only keeps per-viewer conveniences (theme, filter, recent searches).
const prefs = {
  get(key, fallback) {
    try {
      const value = localStorage.getItem(key);
      return value === null ? fallback : JSON.parse(value);
    } catch (err) {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (err) {
      /* storage unavailable (private window): keep working without it */
    }
  },
};

// Apply the saved theme before the page is painted.
applyTheme(prefs.get("np.theme", null));

const state = {
  status: null,
  backend: "heuristic",
  supported: TEXT_EXTENSIONS.concat(BINARY_EXTENSIONS),
  threshold: 0.5,
  topK: PAGE_SIZE,
  searchTimer: null,
  searchSeq: 0,
  recentTimer: null,
  pollTimer: null,
  watchedJob: null,
  handledJob: null,
  phaseMark: null,
  clientSkipped: [],
  clientIgnored: 0,
  dragDepth: 0,
  toastTimer: null,
};
const cardData = new WeakMap();

const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function icon(name) {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("width", "16");
  svg.setAttribute("height", "16");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS(ns, "path");
  path.setAttribute("d", ICONS[name]);
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", "currentColor");
  path.setAttribute("stroke-width", "2");
  path.setAttribute("stroke-linecap", "round");
  path.setAttribute("stroke-linejoin", "round");
  svg.appendChild(path);
  return svg;
}

async function api(path, body) {
  const options = body === undefined
    ? { method: "GET" }
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(path, options);
  let data = {};
  try {
    data = await res.json();
  } catch (err) {
    data = {};
  }
  if (!res.ok || data.error) {
    const error = new Error(data.error || `Erreur HTTP ${res.status}`);
    error.status = res.status;
    throw error;
  }
  return data;
}

/* ------------------------------------------------------------------ formatting */

const numberFormat = new Intl.NumberFormat("fr-FR");

function plural(count, one, many) {
  return `${numberFormat.format(count)} ${count > 1 ? many : one}`;
}

function basename(path) {
  const parts = String(path || "").replace(/[\\/]+$/, "").split(/[\\/]/);
  return parts[parts.length - 1] || String(path || "");
}

function splitPath(relPath) {
  const index = relPath.lastIndexOf("/");
  return index < 0 ? { dir: "", name: relPath } : { dir: relPath.slice(0, index + 1), name: relPath.slice(index + 1) };
}

function percent(score) {
  return `${Math.round(score * 100)} %`;
}

function linesLabel(result) {
  return result.line_start === result.line_end
    ? `ligne ${result.line_start}`
    : `lignes ${result.line_start}–${result.line_end}`;
}

function extensionOf(name) {
  const dot = name.lastIndexOf(".");
  return dot >= 0 ? name.slice(dot).toLowerCase() : "";
}

function filterName() {
  return FILTERS[state.threshold] || `${Math.round(state.threshold * 100)} %`;
}

// Light Markdown clean-up: headings and list markers become plain text.
function cleanLines(text) {
  return String(text || "")
    .split("\n")
    .map((line) => line.replace(/^\s*#{1,6}\s+/, "").replace(/^\s*[-*]\s+/, "• "))
    .join("\n");
}

function plainText(text) {
  return cleanLines(text).replace(/\*\*/g, "").replace(/^• /gm, "").trim();
}

// Append text, wrapping the words listed in ``terms`` (lowercase) in <mark>.
function appendMarked(parent, text, terms) {
  if (!terms || terms.size === 0) {
    parent.appendChild(document.createTextNode(text));
    return;
  }
  const words = /[\p{L}\p{N}_]+/gu;
  let last = 0;
  let match;
  while ((match = words.exec(text)) !== null) {
    if (!terms.has(match[0].toLowerCase())) continue;
    if (match.index > last) parent.appendChild(document.createTextNode(text.slice(last, match.index)));
    parent.appendChild(el("mark", null, match[0]));
    last = match.index + match[0].length;
  }
  if (last < text.length) parent.appendChild(document.createTextNode(text.slice(last)));
}

// Render **bold** and query-word highlights as safe DOM nodes.
function appendRich(parent, text, terms) {
  cleanLines(text).split("**").forEach((part, index) => {
    if (!part) return;
    const target = index % 2 === 1 ? parent.appendChild(el("strong")) : parent;
    appendMarked(target, part, terms);
  });
}

/* ------------------------------------------------------------------ theme, toast, clipboard */

function applyTheme(theme) {
  if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
}

function currentTheme() {
  const explicit = document.documentElement.dataset.theme;
  if (explicit) return explicit;
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function toggleTheme() {
  const next = currentTheme() === "dark" ? "light" : "dark";
  applyTheme(next);
  prefs.set("np.theme", next);
}

function toast(message, isError) {
  const node = $("toast");
  node.textContent = message;
  node.className = isError ? "toast is-error" : "toast";
  node.hidden = false;
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => { node.hidden = true; }, isError ? 6000 : 3000);
}

async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const area = el("textarea", "sr-only");
  area.value = text;
  area.setAttribute("readonly", "");
  document.body.appendChild(area);
  area.select();
  const ok = document.execCommand("copy");
  area.remove();
  if (!ok) throw new Error("copie impossible");
}

/* ------------------------------------------------------------------ status and indexing jobs */

async function refreshStatus() {
  let st;
  try {
    st = await api("/api/status");
  } catch (err) {
    $("folderName").textContent = "Serveur indisponible";
    $("folderStats").textContent = "relancez « nanoprune app »";
    if (state.watchedJob) schedulePoll(2000);
    return null;
  }
  state.status = st;
  const model = st.model || {};
  state.backend = model.backend || "heuristic";
  if (Array.isArray(st.supported_formats) && st.supported_formats.length) state.supported = st.supported_formats;
  renderEngine(model);
  renderFolder(st);
  renderSuggestions();
  handleJob(st.job);
  return st;
}

function renderEngine(model) {
  const tag = $("engineTag");
  if (state.backend === "semantic") {
    tag.textContent = "Recherche sémantique";
    tag.className = "engine-tag is-semantic";
    tag.title = `Modèle ${model.model_name || ""} : trouve aussi les passages formulés autrement. Cliquez pour l'aide.`;
    $("scoreHelp").textContent =
      "Le pourcentage estime la probabilité que le passage réponde à la question. Il est calibré : sur notre " +
      "jeu de test, environ 85 % des décisions prises au seuil de 50 % sont justes.";
  } else if (state.backend === "heuristic") {
    tag.textContent = "Mots-clés";
    tag.className = "engine-tag is-heuristic";
    tag.title = "Aucun modèle chargé : les passages sont trouvés par leurs mots. Cliquez pour l'aide.";
    $("scoreHelp").textContent =
      "En mode mots-clés, le pourcentage mesure la présence des mots de la recherche dans le passage : " +
      "ce n'est pas une probabilité.";
  } else {
    tag.textContent = model.model_name || state.backend;
    tag.className = "engine-tag is-model";
    tag.title = `Modèle ${model.model_name || state.backend}. Cliquez pour l'aide.`;
    $("scoreHelp").textContent = "Score de pertinence du modèle NanoPrune, de 0 à 100 %.";
  }
  $("modeBanner").hidden = state.backend !== "heuristic";
}

function renderFolder(st) {
  const hasIndex = st.indexed_chunks > 0;
  const name = st.current_folder ? (st.folder_path ? basename(st.current_folder) : st.current_folder) : "";
  const firstImport = !hasIndex && st.job && st.job.state === "running" ? st.job : null;
  $("folderName").textContent = name || (firstImport ? firstImport.label : "Choisir un dossier");
  $("folderStats").textContent = hasIndex
    ? `${plural(st.files_indexed, "fichier", "fichiers")} · ${plural(st.indexed_chunks, "extrait", "extraits")}`
    : firstImport ? "indexation en cours…" : "";
  $("btnFolder").title = st.folder_path ? `${st.folder_path}\nCliquer pour changer de dossier` : "Changer de dossier";
  document.title = name ? `${name} — NanoPrune` : "NanoPrune — Recherche locale";
  $("formatsHint").textContent = `Formats lus : ${state.supported.join(", ")}`;
}

function schedulePoll(delay) {
  if (state.pollTimer) return;
  state.pollTimer = setTimeout(() => {
    state.pollTimer = null;
    refreshStatus();
  }, delay || 400);
}

function handleJob(job) {
  const card = $("jobCard");
  if (job && job.state === "running") {
    state.watchedJob = job.id;
    renderJob(job);
    card.hidden = false;
    schedulePoll();
    return;
  }
  card.hidden = true;
  if (!job || state.handledJob === job.id) return;
  state.handledJob = job.id;
  if (state.watchedJob === job.id) {
    state.watchedJob = null;
    jobFinished(job);
  } else if (job.state === "error") {
    showReport(`L'indexation a échoué : ${job.error}`, true);
    toggleImport(true);
  }
}

function remainingTime(job) {
  const now = performance.now();
  const key = `${job.id}:${job.phase}`;
  if (!state.phaseMark || state.phaseMark.key !== key) {
    state.phaseMark = { key, done: job.done, time: now };
    return "";
  }
  const seconds = (now - state.phaseMark.time) / 1000;
  const progressed = job.done - state.phaseMark.done;
  if (seconds < 3 || progressed <= 0 || !job.total) return "";
  const remaining = (job.total - job.done) / (progressed / seconds);
  if (remaining < 60) return " · moins d'une minute restante";
  const minutes = Math.round(remaining / 60);
  return ` · environ ${minutes} min restante${minutes > 1 ? "s" : ""}`;
}

function renderJob(job) {
  const label = job.kind === "folder" && job.label ? ` « ${job.label} »` : "";
  let title = "Indexation…";
  let detail = "";
  let fraction = null;
  if (job.cancel_requested) {
    title = "Annulation…";
  } else if (job.phase === "listing") {
    title = `Recherche des fichiers${label}…`;
  } else if (job.phase === "reading") {
    title = `Lecture des fichiers${label}`;
    if (job.total) {
      fraction = job.done / job.total;
      detail = `${numberFormat.format(job.done)} / ${plural(job.total, "fichier", "fichiers")}`;
    }
  } else if (job.phase === "embedding") {
    title = "Analyse sémantique des passages";
    if (job.total) {
      fraction = job.done / job.total;
      detail = `${numberFormat.format(job.done)} / ${plural(job.total, "extrait", "extraits")}${remainingTime(job)}`;
    }
  }
  if (!detail && state.status && state.status.indexed_chunks) {
    detail = "La recherche porte sur l'index précédent jusqu'à la fin.";
  }
  setProgress(title, detail, fraction);
  $("btnCancel").disabled = Boolean(job.cancel_requested);
  $("btnCancel").hidden = false;
}

function setProgress(title, detail, fraction) {
  $("jobTitle").textContent = title;
  $("jobDetail").textContent = detail || "";
  const progress = $("jobProgress");
  const bar = $("jobBar");
  progress.classList.toggle("is-indeterminate", fraction === null);
  if (fraction === null) {
    bar.style.transform = "";
    progress.removeAttribute("aria-valuenow");
  } else {
    const clamped = Math.max(0.02, Math.min(1, fraction));
    bar.style.transform = `scaleX(${clamped})`;
    progress.setAttribute("aria-valuenow", String(Math.round(clamped * 100)));
  }
}

function jobFinished(job) {
  if (job.state === "done") {
    const report = job.report || {};
    const counts = `${plural(report.files_indexed || 0, "fichier", "fichiers")}, ` +
      `${plural(report.total_chunks || 0, "extrait", "extraits")}`;
    toast(job.kind === "folder" ? `« ${basename(report.folder)} » est prêt : ${counts}.` : `Import terminé : ${counts}.`);
    const skipped = describeSkipped(report);
    if (skipped) {
      showReport(skipped, false);
    } else {
      $("importReport").hidden = true;
      if (report.total_chunks) toggleImport(false);
    }
    if (!report.total_chunks) {
      showReport("Aucun passage n'a pu être lu dans ces documents." + (skipped ? `\n${skipped}` : ""), true);
      toggleImport(true);
    }
    state.clientSkipped = [];
    state.clientIgnored = 0;
    state.topK = PAGE_SIZE;
    doSearch();
    if (report.total_chunks) $("q").focus();
    return;
  }
  const hasIndex = Boolean(state.status && state.status.indexed_chunks);
  if (job.state === "cancelled") {
    toast(hasIndex ? "Import annulé : l'index précédent est conservé." : "Import annulé.");
    if (!hasIndex) toggleImport(true);
  } else {
    showReport(`L'import a échoué : ${job.error}`, true);
    toggleImport(true);
    toast("L'import a échoué.", true);
  }
  doSearch(); // refresh the idle message or the results
}

function describeSkipped(report) {
  const skipped = (report.skipped || []).map((s) => `• ${basename(s.file)} : ${s.reason}`);
  const count = (report.skipped_count || 0) + state.clientSkipped.length;
  state.clientSkipped.forEach((name) => skipped.push(`• ${name} : trop volumineux`));
  const lines = [];
  if (count) {
    lines.push(`${plural(count, "fichier ignoré", "fichiers ignorés")} :`);
    lines.push(...skipped.slice(0, 8));
    if (count > 8) lines.push(`… et ${count - 8} autre(s)`);
  }
  if (state.clientIgnored) {
    lines.push(`${plural(state.clientIgnored, "fichier", "fichiers")} dans un format non pris en charge, non envoyé(s).`);
  }
  if (report.truncated) lines.push("Limite de 5 000 fichiers atteinte : le dossier n'a été indexé qu'en partie.");
  return lines.join("\n");
}

async function cancelJob() {
  $("btnCancel").disabled = true;
  try {
    await api("/api/cancel", {});
  } catch (err) {
    toast(`Annulation impossible : ${err.message}`, true);
  }
  refreshStatus();
}

/* ------------------------------------------------------------------ panels */

function toggleImport(force) {
  const panel = $("importPanel");
  const open = force !== undefined ? force : panel.hidden;
  panel.hidden = !open;
  $("btnFolder").setAttribute("aria-expanded", String(open));
}

function toggleHelp(force) {
  const panel = $("helpPanel");
  const open = force !== undefined ? force : panel.hidden;
  panel.hidden = !open;
  $("btnHelp").setAttribute("aria-expanded", String(open));
  $("engineTag").setAttribute("aria-expanded", String(open));
}

function showReport(text, isError) {
  const report = $("importReport");
  report.textContent = text;
  report.className = isError ? "import-report is-error" : "import-report";
  report.hidden = !text;
}

/* ------------------------------------------------------------------ suggestions and recent searches */

function recentSearches() {
  const list = prefs.get("np.recent", []);
  return Array.isArray(list) ? list.filter((q) => typeof q === "string" && q.trim()).slice(0, 6) : [];
}

function rememberSearch(query) {
  query = (query || "").trim();
  if (!query || !state.status || state.status.is_sample) return;
  const others = recentSearches().filter((q) => q.toLowerCase() !== query.toLowerCase());
  prefs.set("np.recent", [query].concat(others).slice(0, 6));
  renderSuggestions();
}

function renderSuggestions() {
  const box = $("suggestions");
  box.replaceChildren();
  const st = state.status;
  if (!st || !st.indexed_chunks) return;
  let items;
  let label;
  if (st.is_sample) {
    items = state.backend === "semantic" ? SEMANTIC_SAMPLES.concat(SAMPLE_QUERIES.slice(0, 1)) : SAMPLE_QUERIES;
    label = "Exemples";
  } else {
    items = recentSearches().map((q) => ({ label: q, query: q }));
    label = "Récentes";
  }
  if (!items.length) return;
  box.appendChild(el("span", "chips-label", label));
  items.forEach((item) => {
    const chip = el("button", "chip", item.label);
    chip.type = "button";
    chip.title = item.query;
    chip.addEventListener("click", () => {
      $("q").value = item.query;
      state.topK = PAGE_SIZE;
      doSearch();
      $("q").focus();
    });
    box.appendChild(chip);
  });
  if (!st.is_sample) {
    const clear = el("button", "chip is-clear", "effacer");
    clear.type = "button";
    clear.title = "Effacer les recherches récentes";
    clear.addEventListener("click", () => {
      prefs.set("np.recent", []);
      renderSuggestions();
    });
    box.appendChild(clear);
  }
}

/* ------------------------------------------------------------------ search */

function scheduleSearch() {
  clearTimeout(state.searchTimer);
  state.searchTimer = setTimeout(doSearch, 200);
}

async function doSearch() {
  clearTimeout(state.searchTimer);
  const query = $("q").value.trim();
  const seq = ++state.searchSeq;
  const st = state.status;
  if (!query || !st || !st.indexed_chunks) {
    $("searchSpinner").hidden = true;
    renderIdle();
    return;
  }
  const spinner = setTimeout(() => {
    if (seq === state.searchSeq) $("searchSpinner").hidden = false;
  }, 250);
  try {
    const data = await api("/api/search", { query, threshold: state.threshold, top_k: state.topK });
    if (seq !== state.searchSeq) return; // a newer search is in flight
    renderResults(data);
    clearTimeout(state.recentTimer);
    if (data.results.length) {
      // Remember searches the user settled on (still displayed, same folder).
      const folder = st.current_folder;
      state.recentTimer = setTimeout(() => {
        if ($("q").value.trim() === query && state.status && state.status.current_folder === folder) {
          rememberSearch(query);
        }
      }, 2500);
    }
  } catch (err) {
    if (seq === state.searchSeq) renderError(`La recherche a échoué : ${err.message}`);
  } finally {
    clearTimeout(spinner);
    if (seq === state.searchSeq) $("searchSpinner").hidden = true;
  }
}

function emptyBox(className, title, lines) {
  const box = el("div", className);
  box.appendChild(el("p", "empty-title", title));
  (lines || []).forEach((line) => box.appendChild(typeof line === "string" ? el("p", null, line) : line));
  return box;
}

function renderIdle() {
  $("summary").textContent = "";
  $("moreRow").hidden = true;
  const container = $("results");
  const st = state.status;
  if (!st) {
    container.replaceChildren();
    return;
  }
  const running = st.job && st.job.state === "running";
  if (!st.indexed_chunks) {
    if (running) {
      container.replaceChildren(emptyBox("empty", "Indexation en cours…",
        ["La recherche sera disponible dès la fin de l'analyse des documents."]));
      return;
    }
    const button = el("button", "btn btn-primary", "Choisir un dossier…");
    button.type = "button";
    button.addEventListener("click", () => {
      toggleImport(true);
      $("folderInput").click();
    });
    container.replaceChildren(emptyBox("empty", "Cherchez dans vos documents, sans rien envoyer sur Internet", [
      "Choisissez un dossier : NanoPrune lit les fichiers texte, Markdown, Word et PDF, " +
        "puis retrouve les passages qui répondent à vos questions.",
      button,
    ]));
    return;
  }
  const where = st.current_folder ? ` dans « ${st.folder_path ? basename(st.current_folder) : st.current_folder} »` : "";
  const hint = state.backend === "semantic"
    ? "Posez votre question avec vos mots : la recherche comprend aussi les formulations différentes."
    : state.backend === "heuristic"
      ? "Tapez des mots présents dans les documents (mode mots-clés)."
      : "Tapez une question ou des mots-clés.";
  const tip = el("p", "hint");
  tip.append("Astuce : ", el("kbd", null, "/"), " pour chercher, ", el("kbd", null, "?"), " pour l'aide.");
  container.replaceChildren(emptyBox("empty is-plain",
    `Prêt : ${plural(st.files_indexed, "fichier", "fichiers")}, ${plural(st.indexed_chunks, "extrait", "extraits")}${where}.`,
    [hint, tip]));
}

function renderError(message) {
  $("summary").textContent = "";
  $("moreRow").hidden = true;
  $("results").replaceChildren(emptyBox("empty is-error", "Oups", [message]));
}

function groupByFile(results) {
  const groups = new Map();
  results.forEach((result) => {
    const key = result.rel_path || result.file_path;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(result);
  });
  return Array.from(groups.values());
}

function renderResults(data) {
  const container = $("results");
  const results = data.results || [];
  const near = data.near_misses || [];
  const heuristic = data.backend === "heuristic";
  const latency = `${Math.round(data.latency_ms)} ms`;
  $("moreRow").hidden = !data.more_available || state.topK >= MAX_RESULTS;

  if (!results.length) {
    $("summary").textContent = `Aucun passage au-dessus du filtre « ${filterName()} » · ${latency}`;
    const tips = [];
    if (state.threshold > 0.3) tips.push("Essayez le filtre « Large » ou reformulez la recherche.");
    else tips.push("Reformulez la recherche ou vérifiez le dossier indexé.");
    if (heuristic) tips.push("En mode mots-clés, seuls les passages contenant vos mots sont trouvés.");
    const box = emptyBox("empty", "Aucun passage ne répond assez bien à cette recherche", tips);
    if (near.length) {
      const nearBox = el("div", "empty-near");
      nearBox.appendChild(el("p", null, near.length > 1
        ? "Les passages les plus proches, sous le filtre :"
        : "Le passage le plus proche, sous le filtre :"));
      const list = el("div", "results");
      groupByFile(near).forEach((group) => list.appendChild(renderGroup(group, heuristic, true)));
      nearBox.appendChild(list);
      box.appendChild(nearBox);
    }
    container.replaceChildren(box);
    return;
  }

  const groups = groupByFile(results);
  let summary = plural(results.length, "passage", "passages");
  if (results.length > 1) summary += ` dans ${plural(groups.length, "fichier", "fichiers")}`;
  if (data.more_available) summary += ` (${numberFormat.format(data.matches_total)} au total)`;
  $("summary").textContent = `${summary} · ${latency}`;

  const nodes = groups.map((group) => renderGroup(group, heuristic, false));
  if (near.length) {
    const details = el("details", "near-block");
    details.appendChild(el("summary", null,
      `${plural(near.length, "passage proche", "passages proches")}, sous le filtre « ${filterName()} »`));
    const list = el("div", "results");
    groupByFile(near).forEach((group) => list.appendChild(renderGroup(group, heuristic, true)));
    details.appendChild(list);
    nodes.push(details);
  }
  container.replaceChildren(...nodes);
}

function actionButton(name, label, onClick) {
  const button = el("button", "icon-btn small");
  button.type = "button";
  button.title = label;
  button.setAttribute("aria-label", label);
  button.appendChild(icon(name));
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    onClick();
  });
  return button;
}

function renderHead(result, heuristic, withFile) {
  const head = el("div", "result-head");
  const tier = result.score >= 0.8 ? "tier-high" : result.score >= 0.6 ? "tier-mid" : "tier-low";
  const score = el("span", `score ${heuristic ? "is-heuristic" : tier}`, percent(result.score));
  score.title = heuristic
    ? "Score mots-clés : présence des mots de la recherche (pas une probabilité)"
    : "Pertinence estimée du passage pour cette recherche";
  const source = el("div", "result-source");
  if (withFile) {
    const { dir, name } = splitPath(result.rel_path || result.file_name);
    source.appendChild(el("span", "file-name", name));
    if (dir) source.appendChild(el("span", "file-dir", dir));
  }
  source.appendChild(el("span", "lines", linesLabel(result)));
  const actions = el("div", "result-actions");
  if (result.can_open) actions.appendChild(actionButton("open", "Ouvrir le fichier (O)", () => openPassage(result)));
  actions.appendChild(actionButton("copy", "Copier la citation (C)", () => copyCitation(result)));
  head.append(score, source, actions);
  return head;
}

function renderPassage(parent, result) {
  const terms = new Set(result.highlight_terms || []);
  const quote = el("blockquote", "quote");
  // A one-line quote taken from a list item does not need its bullet.
  const highlight = result.highlight.includes("\n") ? result.highlight : result.highlight.replace(/^\s*[-*]\s+/, "");
  appendRich(quote, highlight, terms);
  parent.appendChild(quote);
  if (plainText(result.text) !== plainText(result.highlight)) {
    const details = el("details", "context");
    details.appendChild(el("summary", null, "Voir le passage complet"));
    const passage = el("div", "passage");
    appendRich(passage, result.text, terms);
    details.appendChild(passage);
    parent.appendChild(details);
  }
}

function renderGroup(passages, heuristic, isNear) {
  const [first, ...others] = passages;
  const card = el("article", isNear ? "result is-near" : "result");
  card.tabIndex = 0;
  cardData.set(card, first);
  card.appendChild(renderHead(first, heuristic, true));
  renderPassage(card, first);
  if (others.length) {
    const details = el("details", "others");
    details.appendChild(el("summary", null, others.length > 1
      ? `${others.length} autres passages dans ce fichier`
      : "1 autre passage dans ce fichier"));
    others.forEach((result) => {
      const sub = el("div", "sub-passage");
      sub.appendChild(renderHead(result, heuristic, false));
      renderPassage(sub, result);
      details.appendChild(sub);
    });
    card.appendChild(details);
  }
  return card;
}

async function openPassage(result) {
  try {
    await api("/api/open", { chunk_id: result.chunk_id });
    toast(`Ouverture de ${result.file_name}…`);
    rememberSearch($("q").value);
  } catch (err) {
    toast(err.message, true);
  }
}

async function copyCitation(result) {
  const citation = `« ${plainText(result.highlight)} »\n— ${result.rel_path || result.file_name}, ${linesLabel(result)}`;
  try {
    await copyText(citation);
    toast("Citation copiée dans le presse-papiers.");
    rememberSearch($("q").value);
  } catch (err) {
    toast("Impossible de copier la citation.", true);
  }
}

/* ------------------------------------------------------------------ import */

function toBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

async function uploadFiles(fileList) {
  const files = Array.from(fileList || []);
  const payload = [];
  const tooLarge = [];
  let total = 0;
  const candidates = files.filter((file) => !file.name.startsWith(".") && state.supported.includes(extensionOf(file.name)));
  state.clientIgnored = files.filter((file) => !file.name.startsWith(".")).length - candidates.length;
  if (candidates.length === 0) {
    showReport(`Aucun fichier au format pris en charge dans la sélection (${state.supported.join(", ")}).`, true);
    toggleImport(true);
    return;
  }
  $("jobCard").hidden = false;
  $("btnCancel").hidden = true;
  setProgress(`Préparation de ${plural(candidates.length, "fichier", "fichiers")}…`, "", null);
  try {
    for (const file of candidates) {
      if (file.size > MAX_FILE_BYTES || total + file.size > MAX_UPLOAD_BYTES) {
        tooLarge.push(file.name);
        continue;
      }
      total += file.size;
      if (BINARY_EXTENSIONS.includes(extensionOf(file.name))) {
        payload.push({ name: file.name, data_base64: toBase64(await file.arrayBuffer()) });
      } else {
        payload.push({ name: file.name, content: await file.text() });
      }
    }
    if (payload.length === 0) {
      $("jobCard").hidden = true;
      showReport("Les fichiers sélectionnés sont trop volumineux (25 Mo par fichier, 60 Mo au total). " +
        "Indiquez plutôt le chemin du dossier.", true);
      toggleImport(true);
      return;
    }
    setProgress(`Envoi de ${plural(payload.length, "fichier", "fichiers")} au serveur local…`, "", null);
    const data = await api("/api/index_direct", { files: payload });
    state.clientSkipped = tooLarge;
    state.watchedJob = data.job.id;
    showReport("", false);
    await refreshStatus();
  } catch (err) {
    $("jobCard").hidden = true;
    if (err.status === 409) {
      toast("Un import est déjà en cours.", true);
    } else {
      showReport(`Erreur d'import : ${err.message}`, true);
      toggleImport(true);
    }
    refreshStatus();
  }
}

// Collect files from a drop, walking dropped folders recursively.
async function filesFromDrop(dataTransfer) {
  const items = Array.from(dataTransfer.items || []);
  // Entries must be read synchronously, before the first await.
  const entries = items.map((item) => (item.webkitGetAsEntry ? item.webkitGetAsEntry() : null)).filter(Boolean);
  if (entries.length === 0) return Array.from(dataTransfer.files || []);

  const files = [];
  const readEntries = (reader) => new Promise((resolve, reject) => reader.readEntries(resolve, reject));
  const fileOf = (entry) => new Promise((resolve, reject) => entry.file(resolve, reject));
  async function walk(entry) {
    if (entry.name.startsWith(".")) return;
    if (entry.isFile) {
      files.push(await fileOf(entry));
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      let batch;
      do {
        batch = await readEntries(reader);
        for (const child of batch) await walk(child);
      } while (batch.length > 0);
    }
  }
  for (const entry of entries) await walk(entry);
  return files;
}

async function loadFolderFromPath(event) {
  event.preventDefault();
  const path = $("pathInput").value.trim();
  if (!path) {
    $("pathInput").focus();
    return;
  }
  try {
    const data = await api("/api/load_folder", { folder_path: path });
    prefs.set("np.lastPath", path);
    state.clientSkipped = [];
    state.clientIgnored = 0;
    state.watchedJob = data.job.id;
    showReport("", false);
    await refreshStatus();
  } catch (err) {
    if (err.status === 409) toast("Un import est déjà en cours.", true);
    else showReport(err.message, true);
  }
}

function hasFiles(event) {
  return Boolean(event.dataTransfer) && Array.from(event.dataTransfer.types || []).includes("Files");
}

function wireDragAndDrop() {
  const overlay = $("dropOverlay");
  window.addEventListener("dragenter", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    state.dragDepth += 1;
    overlay.hidden = false;
  });
  window.addEventListener("dragover", (event) => {
    if (hasFiles(event)) event.preventDefault();
  });
  window.addEventListener("dragleave", (event) => {
    if (!hasFiles(event)) return;
    state.dragDepth = Math.max(0, state.dragDepth - 1);
    if (state.dragDepth === 0) overlay.hidden = true;
  });
  window.addEventListener("drop", async (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    state.dragDepth = 0;
    overlay.hidden = true;
    uploadFiles(await filesFromDrop(event.dataTransfer));
  });
}

/* ------------------------------------------------------------------ filter */

function renderFilter() {
  document.querySelectorAll("#filter button").forEach((button) => {
    const checked = parseFloat(button.dataset.threshold) === state.threshold;
    button.setAttribute("aria-checked", String(checked));
    button.tabIndex = checked ? 0 : -1;
  });
}

function setThreshold(value) {
  state.threshold = value;
  prefs.set("np.threshold", value);
  renderFilter();
  state.topK = PAGE_SIZE;
  doSearch();
}

function wireFilter() {
  const buttons = Array.from(document.querySelectorAll("#filter button"));
  buttons.forEach((button, index) => {
    button.addEventListener("click", () => setThreshold(parseFloat(button.dataset.threshold)));
    button.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
      event.preventDefault();
      const next = buttons[(index + (event.key === "ArrowRight" ? 1 : buttons.length - 1)) % buttons.length];
      next.focus();
      setThreshold(parseFloat(next.dataset.threshold));
    });
  });
  const saved = parseFloat(prefs.get("np.threshold", 0.5));
  state.threshold = FILTERS[saved] ? saved : 0.5;
  renderFilter();
}

/* ------------------------------------------------------------------ keyboard */

function visibleCards() {
  return Array.from(document.querySelectorAll("#results .result")).filter((card) => card.offsetParent !== null);
}

function focusSearch() {
  const q = $("q");
  q.focus();
  q.select();
}

function onKeydown(event) {
  const target = event.target;
  const typing = target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement;
  if ((event.ctrlKey || event.metaKey) && (event.key === "k" || event.key === "K")) {
    event.preventDefault();
    focusSearch();
    return;
  }
  if (event.key === "Escape") {
    if (!$("helpPanel").hidden) {
      toggleHelp(false);
    } else if (!$("importPanel").hidden && state.status && state.status.indexed_chunks) {
      toggleImport(false);
    } else if (target === $("q")) {
      event.preventDefault();
      if ($("q").value) {
        $("q").value = "";
        doSearch();
      } else {
        $("q").blur();
      }
    } else if (target.closest && target.closest(".result")) {
      focusSearch();
    }
    return;
  }
  if (typing || event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.key === "/") {
    event.preventDefault();
    focusSearch();
    return;
  }
  if (event.key === "?") {
    event.preventDefault();
    toggleHelp();
    return;
  }
  const card = target.closest && target.closest(".result");
  if (!card) return;
  const cards = visibleCards();
  const index = cards.indexOf(card);
  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    if (event.key === "ArrowUp" && index <= 0) {
      focusSearch();
    } else {
      const next = cards[Math.max(0, Math.min(cards.length - 1, index + (event.key === "ArrowDown" ? 1 : -1)))];
      next.focus();
      next.scrollIntoView({ block: "nearest" });
    }
  } else if (event.key === "Enter" && target === card) {
    event.preventDefault();
    const details = card.querySelector("details.context") || card.querySelector("details.others");
    if (details) details.open = !details.open;
  } else if (event.key === "o" || event.key === "O") {
    const result = cardData.get(card);
    if (result && result.can_open) openPassage(result);
  } else if (event.key === "c" || event.key === "C") {
    const result = cardData.get(card);
    if (result) copyCitation(result);
  }
}

/* ------------------------------------------------------------------ wiring */

async function init() {
  $("btnFolder").addEventListener("click", () => toggleImport());
  $("btnCloseImport").addEventListener("click", () => toggleImport(false));
  $("btnHelp").addEventListener("click", () => toggleHelp());
  $("engineTag").addEventListener("click", () => toggleHelp());
  $("btnCloseHelp").addEventListener("click", () => toggleHelp(false));
  $("btnTheme").addEventListener("click", toggleTheme);
  $("btnCancel").addEventListener("click", cancelJob);
  $("btnPickFiles").addEventListener("click", () => $("fileInput").click());
  $("fileInput").addEventListener("change", (event) => {
    uploadFiles(event.target.files);
    event.target.value = "";
  });
  $("folderInput").addEventListener("change", (event) => {
    uploadFiles(event.target.files);
    event.target.value = "";
  });
  $("pathForm").addEventListener("submit", loadFolderFromPath);
  $("pathInput").value = prefs.get("np.lastPath", "") || "";
  $("btnMore").addEventListener("click", () => {
    state.topK = Math.min(MAX_RESULTS, state.topK + PAGE_SIZE);
    doSearch();
  });
  $("btnCopyInstall").addEventListener("click", async () => {
    try {
      await copyText($("installCommand").textContent);
      toast("Commande copiée : collez-la dans un terminal.");
    } catch (err) {
      toast("Impossible de copier la commande.", true);
    }
  });

  const dropzone = $("dropzone");
  dropzone.addEventListener("click", () => $("folderInput").click());
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      $("folderInput").click();
    }
  });
  wireDragAndDrop();
  wireFilter();

  const q = $("q");
  q.addEventListener("input", () => {
    state.topK = PAGE_SIZE;
    scheduleSearch();
  });
  q.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      doSearch();
      rememberSearch(q.value);
    } else if (event.key === "ArrowDown") {
      const cards = visibleCards();
      if (cards.length) {
        event.preventDefault();
        cards[0].focus();
      }
    }
  });
  document.addEventListener("keydown", onKeydown);

  const st = await refreshStatus();
  if (st && !st.indexed_chunks && !(st.job && st.job.state === "running")) toggleImport(true);
  renderIdle();
  if (st && st.indexed_chunks) q.focus();
}

document.addEventListener("DOMContentLoaded", init);
