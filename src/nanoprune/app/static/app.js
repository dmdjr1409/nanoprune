"use strict";

// Document contents are always inserted as text nodes (never innerHTML):
// indexed files are untrusted and must not be able to run script in this page.

const TEXT_EXTENSIONS = [".txt", ".md", ".markdown", ".json", ".csv", ".tsv", ".log", ".rst"];
const BINARY_EXTENSIONS = [".docx", ".pdf"];
const MAX_FILE_BYTES = 25 * 1024 * 1024;
const MAX_UPLOAD_BYTES = 60 * 1024 * 1024;

const state = {
  backend: "heuristic",
  supported: TEXT_EXTENSIONS.concat(BINARY_EXTENSIONS),
  searchTimer: null,
  searchSeq: 0,
};

const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
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
    throw new Error(data.error || `Erreur HTTP ${res.status}`);
  }
  return data;
}

// Render light Markdown (**bold**, list markers, headings) as safe DOM nodes.
function appendRichText(parent, text) {
  const cleaned = String(text || "")
    .split("\n")
    .map((line) => line.replace(/^\s*#{1,6}\s+/, "").replace(/^\s*[-*]\s+/, "• "))
    .join("\n");
  cleaned.split("**").forEach((part, index) => {
    if (!part) return;
    parent.appendChild(index % 2 === 1 ? el("strong", null, part) : document.createTextNode(part));
  });
}

function extensionOf(name) {
  const dot = name.lastIndexOf(".");
  return dot >= 0 ? name.slice(dot).toLowerCase() : "";
}

/* ------------------------------------------------------------------ status */

async function refreshStatus() {
  try {
    const st = await api("/api/status");
    const model = st.model || {};
    state.backend = model.backend || "heuristic";
    if (Array.isArray(st.supported_formats) && st.supported_formats.length) {
      state.supported = st.supported_formats;
    }
    $("statInfo").textContent = `${st.indexed_chunks} extraits indexés • 100 % local`;
    $("formatsHint").textContent =
      `Formats : ${state.supported.join(", ")} (PDF : nécessite pypdf côté serveur)`;

    const tag = $("modelTag");
    const banner = $("modeBanner");
    if (state.backend === "heuristic") {
      tag.textContent = "mode mots-clés";
      tag.className = "brand-tag is-heuristic";
      banner.textContent =
        "Aucun modèle NanoPrune n'est chargé : les scores viennent d'une heuristique par mots-clés, " +
        "pas du réseau de neurones. Ce ne sont pas des probabilités. " +
        "Voir « Getting the weights » dans le README pour installer un modèle.";
      banner.hidden = false;
    } else {
      const label = state.backend === "semantic" ? "sémantique" : state.backend;
      tag.textContent = `${model.model_name || "modèle"} · ${label}`;
      tag.className = "brand-tag is-model";
      banner.hidden = true;
    }
  } catch (err) {
    $("statInfo").textContent = `Serveur indisponible (${err.message})`;
  }
}

/* ------------------------------------------------------------------ search */

function renderMessage(text, isError) {
  const container = $("results");
  container.replaceChildren(el("div", isError ? "empty is-error" : "empty", text));
}

function renderResults(data) {
  const container = $("results");
  if (!data.results || data.results.length === 0) {
    renderMessage("Aucun passage au-dessus du seuil.");
    return;
  }
  const heuristic = data.backend === "heuristic";
  const items = data.results.map((r) => {
    const item = el("article", "result-item");
    const top = el("div", "result-top");
    const title = el("span", "result-title", `📄 ${r.file_name}`);
    const lines = r.line_start === r.line_end ? `ligne ${r.line_start}` : `lignes ${r.line_start}–${r.line_end}`;
    title.appendChild(el("span", "result-lines", lines));
    const pct = Math.round(r.score * 1000) / 10;
    const badge = el("span", heuristic ? "score-badge is-heuristic" : "score-badge",
      heuristic ? `mots-clés ${pct} %` : `pertinence ${pct} %`);
    badge.title = heuristic
      ? "Score de l'heuristique par mots-clés (pas une probabilité)"
      : "Score de pertinence du modèle (0 à 100 %)";
    top.append(title, badge);

    const quote = el("div", "proof-quote");
    appendRichText(quote, r.highlight);
    item.append(top, quote);
    if ((r.text || "").trim() !== (r.highlight || "").trim()) {
      const context = el("div", "full-context");
      appendRichText(context, r.text);
      item.appendChild(context);
    }
    return item;
  });
  container.replaceChildren(...items);
}

async function doSearch() {
  const query = $("q").value.trim();
  const seq = ++state.searchSeq;
  if (!query) {
    renderMessage("Tape une requête pour chercher dans les documents indexés.");
    $("latencyTag").textContent = "-- ms";
    return;
  }
  try {
    const data = await api("/api/search", {
      query,
      threshold: parseFloat($("threshold").value),
      top_k: 8,
    });
    if (seq !== state.searchSeq) return; // a newer search is in flight
    $("latencyTag").textContent = `${data.latency_ms} ms · ${data.matches_retained} trouvé(s)`;
    renderResults(data);
  } catch (err) {
    if (seq === state.searchSeq) renderMessage(`Erreur : ${err.message}`, true);
  }
}

function debounceSearch() {
  clearTimeout(state.searchTimer);
  state.searchTimer = setTimeout(doSearch, 150);
}

/* ------------------------------------------------------------------ import */

function toggleDrawer(force) {
  const drawer = $("importDrawer");
  const open = force !== undefined ? force : drawer.hidden;
  drawer.hidden = !open;
  $("btnImport").setAttribute("aria-expanded", String(open));
}

function showReport(text, isError) {
  const report = $("importReport");
  report.textContent = text;
  report.className = isError ? "import-report is-error" : "import-report";
  report.hidden = false;
}

function describeImport(data) {
  let text = `${data.files_indexed} fichier(s) indexé(s), ${data.total_chunks} extraits.`;
  if (data.skipped_count) {
    const details = (data.skipped || []).slice(0, 5).map((s) => `• ${s.file} : ${s.reason}`).join("\n");
    text += `\n${data.skipped_count} fichier(s) ignoré(s) :\n${details}`;
  }
  if (data.truncated) text += "\nLimite de fichiers atteinte : dossier indexé partiellement.";
  return text;
}

async function afterImport(data) {
  showReport(describeImport(data), false);
  await refreshStatus();
  doSearch();
}

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
  const skipped = [];
  let total = 0;
  for (const file of files) {
    const name = file.name;
    const ext = extensionOf(name);
    if (name.startsWith(".") || !state.supported.includes(ext)) continue;
    if (file.size > MAX_FILE_BYTES || total + file.size > MAX_UPLOAD_BYTES) {
      skipped.push(name);
      continue;
    }
    total += file.size;
    if (BINARY_EXTENSIONS.includes(ext)) {
      payload.push({ name, data_base64: toBase64(await file.arrayBuffer()) });
    } else {
      payload.push({ name, content: await file.text() });
    }
  }
  if (payload.length === 0) {
    showReport("Aucun fichier au format pris en charge dans la sélection.", true);
    return;
  }
  $("statInfo").textContent = `Indexation de ${payload.length} fichier(s)…`;
  try {
    const data = await api("/api/index_direct", { files: payload });
    if (skipped.length) {
      data.skipped_count = (data.skipped_count || 0) + skipped.length;
      data.skipped = (data.skipped || []).concat(skipped.map((file) => ({ file, reason: "trop volumineux" })));
    }
    await afterImport(data);
  } catch (err) {
    showReport(`Erreur d'import : ${err.message}`, true);
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
  if (!path) return;
  $("statInfo").textContent = "Indexation en cours…";
  try {
    const data = await api("/api/load_folder", { folder_path: path });
    await afterImport(data);
  } catch (err) {
    showReport(err.message, true);
    refreshStatus();
  }
}

/* ------------------------------------------------------------------ wiring */

function init() {
  $("btnImport").addEventListener("click", () => toggleDrawer());
  $("btnCloseDrawer").addEventListener("click", () => toggleDrawer(false));
  $("btnPickFiles").addEventListener("click", () => $("fileInput").click());
  $("fileInput").addEventListener("change", (e) => uploadFiles(e.target.files));
  $("folderInput").addEventListener("change", (e) => uploadFiles(e.target.files));
  $("pathForm").addEventListener("submit", loadFolderFromPath);

  const dz = $("dropzone");
  dz.addEventListener("click", () => $("folderInput").click());
  dz.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      $("folderInput").click();
    }
  });
  dz.addEventListener("dragover", (e) => {
    e.preventDefault();
    dz.classList.add("dragover");
  });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", async (e) => {
    e.preventDefault();
    dz.classList.remove("dragover");
    uploadFiles(await filesFromDrop(e.dataTransfer));
  });

  const q = $("q");
  q.addEventListener("input", debounceSearch);
  q.addEventListener("keydown", (e) => {
    if (e.key === "Enter") doSearch();
  });
  document.querySelectorAll(".filter-btn").forEach((button) => {
    button.addEventListener("click", () => {
      q.value = button.dataset.query;
      doSearch();
    });
  });
  const threshold = $("threshold");
  threshold.addEventListener("input", () => {
    $("thresholdValue").textContent = parseFloat(threshold.value).toFixed(2);
    debounceSearch();
  });

  refreshStatus();
}

document.addEventListener("DOMContentLoaded", init);
