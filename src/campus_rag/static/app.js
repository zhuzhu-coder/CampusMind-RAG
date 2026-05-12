const state = {
  ready: false,
  busy: false,
};

const elements = {
  serviceStatus: document.querySelector("#serviceStatus"),
  statusText: document.querySelector("#statusText"),
  warmupButton: document.querySelector("#warmupButton"),
  questionInput: document.querySelector("#questionInput"),
  traceToggle: document.querySelector("#traceToggle"),
  askButton: document.querySelector("#askButton"),
  errorPanel: document.querySelector("#errorPanel"),
  metaInfo: document.querySelector("#metaInfo"),
  answerText: document.querySelector("#answerText"),
  sourceCount: document.querySelector("#sourceCount"),
  sourcesList: document.querySelector("#sourcesList"),
  tracePanel: document.querySelector("#tracePanel"),
  traceContent: document.querySelector("#traceContent"),
};

function setBusy(isBusy) {
  state.busy = isBusy;
  elements.askButton.disabled = isBusy;
  elements.warmupButton.disabled = isBusy;
  elements.askButton.textContent = isBusy ? "处理中..." : "提交问题";
}

function setStatus(ready, text, variant = "muted") {
  state.ready = ready;
  elements.statusText.textContent = text;
  elements.serviceStatus.textContent = ready ? "Ready" : variant === "error" ? "Error" : "Not ready";
  elements.serviceStatus.className = `status-pill status-${variant}`;
}

function showError(message, requestId) {
  const suffix = requestId ? ` Request ID: ${requestId}` : "";
  elements.errorPanel.textContent = `${message}${suffix}`;
  elements.errorPanel.classList.remove("hidden");
}

function clearError() {
  elements.errorPanel.textContent = "";
  elements.errorPanel.classList.add("hidden");
}

async function parseError(response) {
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    return { message: `请求失败，HTTP ${response.status}` };
  }
  return payload.error || { message: `请求失败，HTTP ${response.status}` };
}

function updateMeta(response) {
  const requestId = response.headers.get("X-Request-ID") || "-";
  const processTime = response.headers.get("X-Process-Time-MS") || "-";
  elements.metaInfo.textContent = `request_id=${requestId} · ${processTime} ms`;
}

function renderAnswer(payload) {
  elements.answerText.textContent = payload.answer || "当前没有回答。";
  elements.answerText.classList.remove("empty-state");
}

function renderSources(sources) {
  const sourceItems = Array.isArray(sources) ? sources : [];
  elements.sourceCount.textContent = `${sourceItems.length} 条`;

  if (!sourceItems.length) {
    elements.sourcesList.textContent = "当前响应没有返回来源。";
    elements.sourcesList.className = "sources-list empty-state";
    return;
  }

  elements.sourcesList.className = "sources-list";
  elements.sourcesList.innerHTML = sourceItems
    .map((source) => {
      const page = source.page === null || source.page === undefined ? "-" : source.page;
      const score =
        source.rrf_score === null || source.rrf_score === undefined
          ? "-"
          : Number(source.rrf_score).toFixed(4);
      return `
        <article class="source-card">
          <div class="source-title">${escapeHtml(source.doc_title || "未知文档")}</div>
          <div class="source-meta">
            ${escapeHtml(source.doc_category || "未知")} · ${escapeHtml(source.department || "未注明")} ·
            ${escapeHtml(source.section || "正文")} · page=${page} · chunk=${source.chunk_index} · score=${score}
          </div>
          <p>${escapeHtml(source.snippet || "")}</p>
        </article>
      `;
    })
    .join("");
}

function renderTrace(trace) {
  if (!trace) {
    elements.tracePanel.classList.add("hidden");
    elements.traceContent.innerHTML = "";
    return;
  }

  const timings = trace.timings_ms || {};
  elements.tracePanel.classList.remove("hidden");
  elements.traceContent.innerHTML = Object.entries(timings)
    .map(([name, value]) => {
      return `
        <div class="trace-item">
          <span>${escapeHtml(name)}</span>
          <strong>${Number(value).toFixed(2)} ms</strong>
        </div>
      `;
    })
    .join("");
}

async function checkReady() {
  clearError();
  try {
    const response = await fetch("/ready");
    const payload = await response.json();
    if (payload.ready) {
      setStatus(true, `知识库已就绪：${payload.total_documents} 个文档，${payload.total_chunks} 个片段。`, "ready");
    } else if (payload.status === "error") {
      setStatus(false, payload.last_error || "知识库初始化失败。", "error");
    } else {
      setStatus(false, "知识库未初始化，请先预热。", "muted");
    }
  } catch {
    setStatus(false, "无法连接后端服务。", "error");
  }
}

async function warmup() {
  clearError();
  setBusy(true);
  setStatus(false, "正在预热知识库...", "muted");
  try {
    const response = await fetch("/warmup", { method: "POST" });
    if (!response.ok) {
      const error = await parseError(response);
      showError(error.message || "预热失败。", error.request_id);
      setStatus(false, "知识库预热失败。", "error");
      return;
    }
    await checkReady();
  } catch {
    showError("预热请求失败，请确认 API 服务仍在运行。");
    setStatus(false, "知识库预热失败。", "error");
  } finally {
    setBusy(false);
  }
}

async function askQuestion() {
  const question = elements.questionInput.value.trim();
  if (!question) {
    showError("请输入问题。");
    return;
  }

  clearError();
  setBusy(true);
  elements.answerText.textContent = "正在生成回答...";
  elements.answerText.classList.add("empty-state");

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        return_sources: true,
        return_trace: elements.traceToggle.checked,
      }),
    });

    updateMeta(response);

    if (!response.ok) {
      const error = await parseError(response);
      showError(error.message || "问答请求失败。", error.request_id);
      return;
    }

    const payload = await response.json();
    renderAnswer(payload);
    renderSources(payload.sources);
    renderTrace(payload.trace);
  } catch {
    showError("问答请求失败，请确认 API 服务仍在运行。");
  } finally {
    setBusy(false);
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

elements.warmupButton.addEventListener("click", warmup);
elements.askButton.addEventListener("click", askQuestion);
elements.questionInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    askQuestion();
  }
});

checkReady();
