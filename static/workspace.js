(function (global) {
  function safeList(values) {
    return Array.isArray(values) ? values.filter(Boolean) : [];
  }

  function setText(node, value, fallback) {
    if (!node) return;
    const next = value && String(value).trim() ? String(value).trim() : (fallback || "Not available");
    node.textContent = next;
  }

  function renderChips(container, values, emptyLabel) {
    if (!container) return;
    container.replaceChildren();
    const items = safeList(values);
    if (!items.length) {
      const chip = document.createElement("span");
      chip.className = "inline-flex items-center rounded-full bg-slate-200 dark:bg-slate-700 px-3 py-1 text-xs font-semibold text-slate-600 dark:text-slate-300";
      chip.textContent = emptyLabel;
      container.appendChild(chip);
      return;
    }
    items.forEach((value) => {
      const chip = document.createElement("span");
      chip.className = "inline-flex items-center rounded-full bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 px-3 py-1 text-xs font-semibold text-slate-700 dark:text-slate-200";
      chip.textContent = value;
      container.appendChild(chip);
    });
  }

  function formatPackAndManufacturer(med) {
    const parts = [med && med.pack_size, med && med.manufacturer].filter(Boolean);
    return parts.length ? parts.join(" • ") : "Not available";
  }

  function alternativeLabel(med, webLookupEnabled) {
        const lookup = med && med.alternatives_lookup;
        if (lookup && typeof lookup === "object") {
            if (lookup.local_count || lookup.skipped_reason === "local_candidates_present") {
                return "If this medicine is unavailable (local dataset)";
            }
            if (lookup.web_count) return "If this medicine is unavailable (web/model)";
            if (lookup.skipped_reason === "lookup_disabled") return "Local list empty. Web/model lookup is off.";
            if (lookup.skipped_reason === "no_validated_candidates") return "No validated web/model candidates";
        }
        const status = med && med.alternatives_status;
        if (status === "local_dataset") return "If this medicine is unavailable (local dataset)";
        if (status === "web_model") return "If this medicine is unavailable (web/model)";
        if (status === "web_empty") return "No validated web/model candidates";
        if (status === "web_disabled" || webLookupEnabled === false) {
            return "Local list empty. Web/model lookup is off.";
        }
        return "No dataset reference candidates";
    }

  function alternativeNames(med) {
    const local = safeList(med && med.substitutes);
    if (local.length) return local;
    return safeList(med && med.web_alternatives).map((item) => (item && item.name) || item).filter(Boolean);
  }

  function pipelineSummary(pipeline) {
    if (!pipeline || typeof pipeline !== "object") return "";
    const parts = [];
    if (pipeline.requested_provider && pipeline.used_provider && pipeline.requested_provider !== pipeline.used_provider) {
      parts.push("Fallback " + pipeline.requested_provider + " → " + pipeline.used_provider);
    } else if (pipeline.used_provider) parts.push("Provider: " + pipeline.used_provider);
    if (pipeline.degraded) parts.push("Degraded analysis");
    if (pipeline.error_code) parts.push("Code " + pipeline.error_code);
    const warnings = [...safeList(pipeline.warnings), ...safeList(pipeline.ocr_warnings)];
    if (warnings.length) parts.push(warnings.join(" "));
    return parts.join(". ");
  }

  function showBanner(node, message, kind) {
    if (!node) return;
    node.hidden = !message;
    node.textContent = message || "";
    node.className = kind === "error"
      ? "mt-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-semibold text-red-800"
      : "mt-4 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm font-semibold text-amber-900";
  }

  function errorMessage(payload, fallback) {
    if (!payload) return fallback;
    if (typeof payload.detail === "string") return payload.detail;
    if (payload.error) return payload.error;
    return fallback;
  }

  global.SimpliScribeUI = {
    safeList,
    setText,
    renderChips,
    formatPackAndManufacturer,
    alternativeLabel,
    alternativeNames,
    pipelineSummary,
    showBanner,
    errorMessage,
  };
})(window);
