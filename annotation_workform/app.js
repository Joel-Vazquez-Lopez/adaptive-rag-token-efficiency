(function () {
  const STORAGE_KEY = "rag-annotation-form-state-v2";
  const samples = window.ANNOTATION_SAMPLES || [];
  const state = {
    currentIndex: 0,
    filter: "all",
    search: "",
    annotations: loadState(),
  };

  const els = {
    exportCsvBtn: document.getElementById("exportCsvBtn"),
    exportJsonBtn: document.getElementById("exportJsonBtn"),
    importJsonInput: document.getElementById("importJsonInput"),
    filterSelect: document.getElementById("filterSelect"),
    searchInput: document.getElementById("searchInput"),
    sampleList: document.getElementById("sampleList"),
    prevBtn: document.getElementById("prevBtn"),
    nextBtn: document.getElementById("nextBtn"),
    positionText: document.getElementById("positionText"),
    sampleId: document.getElementById("sampleId"),
    sampleGroup: document.getElementById("sampleGroup"),
    sampleMode: document.getElementById("sampleMode"),
    queryText: document.getElementById("queryText"),
    referenceAnswer: document.getElementById("referenceAnswer"),
    modelAnswer: document.getElementById("modelAnswer"),
    evidenceText: document.getElementById("evidenceText"),
    metricStrip: document.getElementById("metricStrip"),
    annotationForm: document.getElementById("annotationForm"),
  };

  function loadState() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    } catch {
      return {};
    }
  }

  function saveState() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state.annotations));
  }

  function emptyAnnotation() {
    return {
      rating: "",
      notes: "",
    };
  }

  function getAnnotation(sample) {
    if (!state.annotations[sample.candidate_id]) {
      state.annotations[sample.candidate_id] = emptyAnnotation();
    }
    const existing = state.annotations[sample.candidate_id];
    state.annotations[sample.candidate_id] = {
      ...emptyAnnotation(),
      ...existing,
      rating: existing.rating || existing.annotator_1_rating || existing.final_rating || existing.annotator_1_label || existing.final_label || "",
      notes: existing.notes || existing.annotator_1_notes || existing.final_notes || "",
    };
    return state.annotations[sample.candidate_id];
  }

  function statusFor(sample) {
    return getAnnotation(sample).rating ? "complete" : "unstarted";
  }

  function filteredSamples() {
    const q = state.search.trim().toLowerCase();
    return samples.filter((sample) => {
      const status = statusFor(sample);
      if (state.filter === "unstarted" && status !== "unstarted") return false;
      if (state.filter === "complete" && status !== "complete") return false;
      if (state.filter === "worst" && !sample.selection_reason.startsWith("worst")) return false;
      if (state.filter === "best" && !sample.selection_reason.startsWith("best")) return false;
      if (!q) return true;
      const annotation = getAnnotation(sample);
      const haystack = [
        sample.candidate_id,
        sample.query_id,
        sample.query_text,
        sample.reference_answer,
        sample.model_answer,
        annotation.notes,
      ].join(" ").toLowerCase();
      return haystack.includes(q);
    });
  }

  function currentSample() {
    return samples[state.currentIndex] || samples[0];
  }

  function setCurrentById(id) {
    const index = samples.findIndex((sample) => sample.candidate_id === id);
    if (index >= 0) {
      state.currentIndex = index;
      render();
    }
  }

  function renderList() {
    const visible = filteredSamples();
    els.sampleList.innerHTML = "";
    for (const sample of visible) {
      const button = document.createElement("button");
      const status = statusFor(sample);
      button.type = "button";
      button.className = `sample-row ${sample.candidate_id === currentSample()?.candidate_id ? "active" : ""}`;
      button.dataset.id = sample.candidate_id;
      button.innerHTML = `
        <span>
          <strong>${escapeHtml(sample.candidate_id)} · ${escapeHtml(sample.query_id)}</strong>
          <span>${escapeHtml(sample.selection_reason)} · ${escapeHtml(status)}</span>
        </span>
        <i class="status-dot ${escapeHtml(status)}" aria-hidden="true"></i>
      `;
      button.addEventListener("click", () => setCurrentById(sample.candidate_id));
      els.sampleList.appendChild(button);
    }
  }

  function renderSample() {
    const sample = currentSample();
    if (!sample) return;
    const annotation = getAnnotation(sample);
    els.sampleId.textContent = sample.candidate_id;
    els.sampleGroup.textContent = sample.selection_reason;
    els.sampleMode.textContent = sample.method_name || sample.mode;
    els.queryText.textContent = sample.query_text;
    els.referenceAnswer.textContent = sample.reference_answer;
    els.modelAnswer.textContent = sample.model_answer;
    els.evidenceText.textContent = sample.selected_document_text || "No selected evidence text.";
    els.metricStrip.innerHTML = [
      ["F1", sample.answer_f1],
      ["Semantic", sample.semantic_similarity],
      ["nDCG@10", sample.ndcg_at_10],
      ["MRR@10", sample.mrr_at_10],
      ["Docs", sample.docs_used],
      ["Tokens", sample.total_tokens],
    ].map(([label, value]) => `<span class="metric-pill">${label}: ${escapeHtml(value || "")}</span>`).join("");

    els.annotationForm.elements.rating.value = annotation.rating || "";
    els.annotationForm.elements.notes.value = annotation.notes || "";
    els.positionText.textContent = `${state.currentIndex + 1} of ${samples.length}`;
  }

  function updateFromForm() {
    const sample = currentSample();
    const annotation = getAnnotation(sample);
    annotation.rating = els.annotationForm.elements.rating.value;
    annotation.notes = els.annotationForm.elements.notes.value;
    saveState();
    renderList();
  }

  function exportRows() {
    return samples.map((sample) => {
      const annotation = getAnnotation(sample);
      return {
        ...sample,
        rating: annotation.rating || "",
        notes: annotation.notes || "",
      };
    });
  }

  function download(filename, mimeType, content) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function exportJson() {
    const payload = {
      exported_at: new Date().toISOString(),
      rows: exportRows(),
    };
    download("rag_annotations_export.json", "application/json", JSON.stringify(payload, null, 2));
  }

  function exportCsv() {
    const rows = exportRows();
    const headers = Object.keys(rows[0] || {});
    const csv = [
      headers.join(","),
      ...rows.map((row) => headers.map((header) => csvEscape(row[header])).join(",")),
    ].join("\n");
    download("rag_annotations_export.csv", "text/csv", csv);
  }

  function importJson(file) {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = JSON.parse(String(reader.result || "{}"));
        const rows = Array.isArray(parsed.rows) ? parsed.rows : [];
        for (const row of rows) {
          if (!row.candidate_id) continue;
          state.annotations[row.candidate_id] = {
            ...emptyAnnotation(),
            rating: row.rating || row.annotator_1_rating || row.final_rating || row.annotator_1_label || row.final_label || "",
            notes: row.notes || row.annotator_1_notes || row.final_notes || "",
          };
        }
        saveState();
        render();
      } catch (error) {
        alert(`Import failed: ${error.message}`);
      }
    };
    reader.readAsText(file);
  }

  function csvEscape(value) {
    const text = String(value ?? "");
    if (/[",\n\r]/.test(text)) {
      return `"${text.replace(/"/g, '""')}"`;
    }
    return text;
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function render() {
    renderList();
    renderSample();
  }

  els.filterSelect.addEventListener("change", () => {
    state.filter = els.filterSelect.value;
    renderList();
  });
  els.searchInput.addEventListener("input", () => {
    state.search = els.searchInput.value;
    renderList();
  });
  els.prevBtn.addEventListener("click", () => {
    state.currentIndex = Math.max(0, state.currentIndex - 1);
    render();
  });
  els.nextBtn.addEventListener("click", () => {
    state.currentIndex = Math.min(samples.length - 1, state.currentIndex + 1);
    render();
  });
  els.annotationForm.addEventListener("input", updateFromForm);
  els.exportCsvBtn.addEventListener("click", exportCsv);
  els.exportJsonBtn.addEventListener("click", exportJson);
  els.importJsonInput.addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    if (file) importJson(file);
    event.target.value = "";
  });

  render();
})();
