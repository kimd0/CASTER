const SITE_TITLE = "CASTER: A Multimodal Dataset and Benchmark for Observation-Grounded StarCraft Commentary Generation";
const HF_DATASET_REPO = "kimd00/CASTER";
const HF_DATASET_REVISION = "main";
const HF_DATASET_BASE = `https://huggingface.co/datasets/${HF_DATASET_REPO}/resolve/${HF_DATASET_REVISION}`;
const TOTAL_MATCHES = 239;
const VALID_TABS = new Set(["overview", "dataset", "benchmark"]);
const TAG_DISTRIBUTION = [
  { tag: "EVALUATION", count: 6698 },
  { tag: "STATUS_REPORT", count: 5665 },
  { tag: "PREDICTION", count: 5398 },
  { tag: "ACTION_CALL", count: 3581 },
  { tag: "CAST_BANTER", count: 2649 },
  { tag: "CAUSAL_EXPLAIN", count: 2584 },
  { tag: "ALTERNATIVES", count: 1312 },
  { tag: "SUMMARY", count: 518 },
  { tag: "PRODUCTION_CHAT", count: 325 },
];
const RACE_DISTRIBUTION = [
  { label: "Terran", count: 104 },
  { label: "Protoss", count: 111 },
  { label: "Zerg", count: 263 },
];
const MATCHUP_DISTRIBUTION = [
  { label: "Protoss vs Zerg", count: 71 },
  { label: "Terran vs Zerg", count: 70 },
  { label: "Zerg vs Zerg", count: 61 },
  { label: "Protoss vs Terran", count: 28 },
  { label: "Protoss vs Protoss", count: 6 },
  { label: "Terran vs Terran", count: 3 },
];
const TASK1_RESULTS = [
  { model: "Qwen3.5", shortLabel: "Qwen", formatAcc: 96.07, unitF1: 0.133, countMae: 1.87, vpL2: 742.01, unitL2: 403.03, temporalIoU: 0.663 },
  { model: "Mimo-v2", shortLabel: "Mimo", formatAcc: 95.06, unitF1: 0.132, countMae: 1.85, vpL2: 749.12, unitL2: 489.33, temporalIoU: 0.632 },
  { model: "GPT-4o", shortLabel: "GPT-4o", formatAcc: 99.36, unitF1: 0.17, countMae: 1.55, vpL2: 739.36, unitL2: 394.67, temporalIoU: 0.655 },
  { model: "Gemini 3.1 Pro", shortLabel: "Gemini", formatAcc: 99.42, unitF1: 0.202, countMae: 1.04, vpL2: 676.29, unitL2: 325.83, temporalIoU: 0.732 },
];
const TASK1_MODEL_COLORS = ["#9aa8b7", "#73879c", "#48627a", "#22384d"];
const TASK1_METRICS = [
  { handleKey: "benchmarkTask1Format", canvasId: "benchmark-task1-format-chart", key: "formatAcc", label: "Format Acc", tickPrecision: 1, tooltipPrecision: 2 },
  { handleKey: "benchmarkTask1UnitF1", canvasId: "benchmark-task1-unitf1-chart", key: "unitF1", label: "Unit F1", tickPrecision: 3, tooltipPrecision: 3 },
  { handleKey: "benchmarkTask1TIoU", canvasId: "benchmark-task1-tiou-chart", key: "temporalIoU", label: "Temporal IoU", tickPrecision: 3, tooltipPrecision: 3 },
  { handleKey: "benchmarkTask1CountMae", canvasId: "benchmark-task1-countmae-chart", key: "countMae", label: "Count MAE", tickPrecision: 2, tooltipPrecision: 2 },
  { handleKey: "benchmarkTask1VpL2", canvasId: "benchmark-task1-vpl2-chart", key: "vpL2", label: "VP L2 Dist", tickPrecision: 0, tooltipPrecision: 2 },
  { handleKey: "benchmarkTask1UnitL2", canvasId: "benchmark-task1-unitl2-chart", key: "unitL2", label: "Unit L2 Dist", tickPrecision: 0, tooltipPrecision: 2 },
];
const TASK2_OVERALL = [
  { modality: "Multimodal", b4: 3.43, rl: 14.1, bs: 86.34, tc: 4.17, sc: 1.69, cn: 3.98 },
  { modality: "Video-Only", b4: 3.33, rl: 13.92, bs: 86.3, tc: 4.13, sc: 1.67, cn: 3.97 },
  { modality: "Text-Only", b4: 2.77, rl: 12.92, bs: 85.83, tc: 4.13, sc: 1.65, cn: 3.98 },
];
const TASK2_RESULTS = [
  { tag: "Overall", multimodal: { b4: 3.43, rl: 14.10, bs: 86.34, tc: 4.17, sc: 1.69, cn: 3.98 }, videoOnly: { b4: 3.33, rl: 13.92, bs: 86.30, tc: 4.13, sc: 1.67, cn: 3.97 }, textOnly: { b4: 2.77, rl: 12.92, bs: 85.83, tc: 4.13, sc: 1.65, cn: 3.98 } },
  { tag: "ACTION_CALL", multimodal: { b4: 3.47, rl: 15.46, bs: 86.65, tc: 4.23, sc: 0.41, cn: 4.01 }, videoOnly: { b4: 3.32, rl: 14.98, bs: 86.48, tc: 4.22, sc: 1.70, cn: 3.98 }, textOnly: { b4: 2.79, rl: 13.65, bs: 85.99, tc: 3.95, sc: 1.67, cn: 3.94 } },
  { tag: "ALTERNATIVES", multimodal: { b4: 3.08, rl: 13.64, bs: 86.29, tc: 2.64, sc: 1.71, cn: 3.91 }, videoOnly: { b4: 3.09, rl: 13.78, bs: 86.33, tc: 2.07, sc: 1.54, cn: 3.93 }, textOnly: { b4: 2.42, rl: 12.31, bs: 85.79, tc: 2.26, sc: 1.52, cn: 3.93 } },
  { tag: "CAST_BANTER", multimodal: { b4: 1.48, rl: 8.89, bs: 83.49, tc: 3.74, sc: 0.26, cn: 3.80 }, videoOnly: { b4: 1.47, rl: 9.08, bs: 83.48, tc: 3.84, sc: 1.58, cn: 3.91 }, textOnly: { b4: 1.27, rl: 8.73, bs: 83.30, tc: 4.01, sc: 1.64, cn: 3.94 } },
  { tag: "CAUSAL_EXPLAIN", multimodal: { b4: 3.98, rl: 14.92, bs: 86.75, tc: 3.67, sc: 1.82, cn: 3.99 }, videoOnly: { b4: 3.62, rl: 14.55, bs: 86.73, tc: 3.26, sc: 1.85, cn: 3.96 }, textOnly: { b4: 3.06, rl: 13.41, bs: 86.17, tc: 3.35, sc: 1.81, cn: 3.94 } },
  { tag: "EVALUATION", multimodal: { b4: 3.41, rl: 13.85, bs: 86.29, tc: 4.80, sc: 0.66, cn: 4.00 }, videoOnly: { b4: 3.32, rl: 13.64, bs: 86.24, tc: 4.84, sc: 1.69, cn: 4.00 }, textOnly: { b4: 2.71, rl: 12.83, bs: 85.76, tc: 4.82, sc: 1.67, cn: 4.02 } },
  { tag: "PREDICTION", multimodal: { b4: 3.85, rl: 15.23, bs: 87.13, tc: 4.40, sc: 1.83, cn: 4.00 }, videoOnly: { b4: 3.70, rl: 15.08, bs: 87.05, tc: 4.40, sc: 1.80, cn: 3.99 }, textOnly: { b4: 2.98, rl: 13.89, bs: 86.52, tc: 4.59, sc: 1.74, cn: 3.99 } },
  { tag: "PRODUCTION_CHAT", multimodal: { b4: 2.11, rl: 9.82, bs: 84.20, tc: 3.09, sc: 0.03, cn: 4.00 }, videoOnly: { b4: 1.85, rl: 9.85, bs: 83.97, tc: 2.64, sc: 1.82, cn: 3.91 }, textOnly: { b4: 1.76, rl: 10.42, bs: 84.13, tc: 3.07, sc: 1.73, cn: 4.13 } },
  { tag: "STATUS_REPORT", multimodal: { b4: 3.83, rl: 14.86, bs: 86.72, tc: 4.06, sc: 1.56, cn: 3.99 }, videoOnly: { b4: 3.84, rl: 14.63, bs: 86.76, tc: 4.07, sc: 1.50, cn: 3.98 }, textOnly: { b4: 3.28, rl: 13.65, bs: 86.28, tc: 3.98, sc: 1.54, cn: 3.98 } },
  { tag: "SUMMARY", multimodal: { b4: 3.41, rl: 13.28, bs: 86.45, tc: 3.53, sc: 0.05, cn: 4.00 }, videoOnly: { b4: 3.35, rl: 13.74, bs: 86.44, tc: 3.94, sc: 1.72, cn: 4.00 }, textOnly: { b4: 3.30, rl: 12.87, bs: 86.01, tc: 3.57, sc: 1.62, cn: 4.00 } },
];
const CHART_COLORS = {
  ink: "#2e4761",
  dark: "#587492",
  soft: "#9db1c6",
  highlight: "#c97432",
  highlightSoft: "#f0c8a6",
  grid: "#d5dae0",
  text: "#121418",
  muted: "#434b55",
  tooltip: "#17191d",
};

const state = {
  data: null,
  activeTab: "overview",
  activeMatch: 1,
  filteredTag: "ALL",
  activeSegmentId: 0,
};
const chartHandles = {
  overviewTag: null,
  overviewRace: null,
  overviewMatchup: null,
  benchmarkTask1Format: null,
  benchmarkTask1UnitF1: null,
  benchmarkTask1TIoU: null,
  benchmarkTask1CountMae: null,
  benchmarkTask1VpL2: null,
  benchmarkTask1UnitL2: null,
  benchmarkTask2B4: null,
  benchmarkTask2Rl: null,
  benchmarkTask2Bs: null,
  benchmarkTask2Tc: null,
  benchmarkTask2Sc: null,
  benchmarkTask2Cn: null,
};
let chartRenderFrame = 0;
let chartRefreshFrame = 0;
let chartRefreshTimeout = 0;
let playerAliasMap = new Map();
let activeMatchRequestId = 0;
const matchDataCache = new Map();

const matchSelectorEl = document.getElementById("match-selector");
const tagFilterEl = document.getElementById("tag-filter");
const segmentListEl = document.getElementById("segment-list");
const clipTitleEl = document.getElementById("clip-title");
const clipTagPillEl = document.getElementById("clip-tag-pill");
const clipPlayerEl = document.getElementById("clip-player");
const refinedCommentaryEl = document.getElementById("refined-commentary");
const stateGridEl = document.getElementById("state-grid");
const statePreviewEl = document.getElementById("state-preview");
const task2ResultsBodyEl = document.getElementById("task2-results-body");
const tabButtons = Array.from(document.querySelectorAll("[data-tab-trigger]"));
const tabPanels = Array.from(document.querySelectorAll("[data-tab-panel]"));
const chartFontFamily = getComputedStyle(document.documentElement).getPropertyValue("--ui-font").trim() || "sans-serif";

function formatNumber(value) {
  return new Intl.NumberFormat("en-US").format(value);
}

function formatMatchId(matchIndex) {
  return `match_${String(matchIndex).padStart(3, "0")}`;
}

function buildHfUrl(path) {
  return `${HF_DATASET_BASE}/${path}`;
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load ${url}`);
  }
  return response.json();
}

function formatBytes(bytes) {
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unitIndex = 0;

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function joinValues(values, fallback = "N/A") {
  if (!Array.isArray(values) || !values.length) {
    return fallback;
  }
  return values.join(", ");
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => {
    return {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[character];
  });
}

function truncateText(value, maxLength = 160) {
  const normalized = String(value ?? "").trim();
  if (normalized.length <= maxLength) {
    return normalized;
  }
  return `${normalized.slice(0, Math.max(0, maxLength - 1)).trimEnd()}…`;
}

function normalizePresentationText(value) {
  return String(value ?? "")
    .normalize("NFKC")
    .replace(/[^\x20-\x7E]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function selectAlignmentSpeechText(segment) {
  const cleanedRaw = normalizePresentationText(segment.commentary?.raw ?? "");
  const cleanedAligned = normalizePresentationText(segment.alignedCompact?.speech ?? "");
  const cleanedFallback = normalizePresentationText(segment.commentary?.refined ?? "");

  if (!cleanedRaw) {
    return cleanedAligned || cleanedFallback || "No transcript segment available.";
  }

  const rawLooksSuspicious =
    cleanedRaw.length < Math.max(24, Math.floor((cleanedAligned || cleanedFallback).length * 0.72)) ||
    !/^[A-Z"'(\[]/.test(cleanedRaw);

  if (rawLooksSuspicious) {
    return cleanedAligned || cleanedFallback || cleanedRaw;
  }

  return cleanedRaw;
}

function parseCompactUnitEntry(entry) {
  const match = typeof entry === "string" ? entry.match(/^([^*[\]]+)\*(\d+)/) : null;
  if (!match) {
    return { label: String(entry ?? "Unknown"), count: 1 };
  }

  return {
    label: match[1].trim(),
    count: Number(match[2]),
  };
}

function formatCompactViewport(value) {
  return String(value ?? "N/A").replace(/->/g, "→");
}

function formatCoordinatePoint(x, y) {
  return `[${x},${y}]`;
}

function formatAlignmentPlayerShort(playerName) {
  const masked = maskPlayerName(playerName);
  const match = String(masked).match(/\[PLAYER_(\d+)\]/);
  if (match) {
    return `P${match[1]}`;
  }
  return String(masked ?? "P");
}

function formatAlignmentCoordinate(x, y) {
  return `${x},${y}`;
}

function formatAlignmentUnit(unitName) {
  return simplifyUnitName(unitName).replace(/_/g, " ");
}

function formatFrameSpan(start, end) {
  return `${formatNumber(start)}-${formatNumber(end)}`;
}

function parseFrameWindow(value) {
  const [rawStart, rawEnd] = String(value ?? "")
    .split("~")
    .map((item) => Number(String(item).replace(/[^\d-]/g, "")));

  const start = Number.isFinite(rawStart) ? rawStart : 0;
  const end = Number.isFinite(rawEnd) ? rawEnd : start;

  return {
    start,
    end,
    exclusiveEnd: end,
    count: Math.max(0, end - start),
  };
}

function replaceDatasetPlaceholders(text, metadata) {
  const mapName = metadata?.game_info?.map ?? "[MAP_NAME]";
  return String(text ?? "").replace(/\[MAP_NAME\]/g, mapName);
}

function buildTagCounts(records) {
  const counter = new Map();

  for (const record of records) {
    const tag = record.speech_tag || "UNTAGGED";
    counter.set(tag, (counter.get(tag) ?? 0) + 1);
  }

  return [...counter.entries()]
    .map(([tag, count]) => ({ tag, count }))
    .sort((left, right) => right.count - left.count);
}

function buildClipUrl(matchId, clipPath, segmentId) {
  if (clipPath) {
    const normalized = clipPath.endsWith(".mp4") ? clipPath : `${clipPath}.mp4`;
    return buildHfUrl(normalized);
  }

  return buildHfUrl(`${matchId}/clip/${matchId}_${String(segmentId).padStart(3, "0")}.mp4`);
}

function simplifyUnitName(unitName) {
  return String(unitName ?? "Unknown").replace(/^[A-Za-z]+_/, "");
}

function inferRaceFromUnitName(unitName) {
  const token = String(unitName ?? "").split("_")[0];
  return token || null;
}

function inferRaceFromNotableLabel(label) {
  const parts = String(label ?? "").split(":");
  return inferRaceFromUnitName(parts.length > 1 ? parts[1] : parts[0]);
}

function buildPlayerAliasMap(data) {
  const actualByRace = new Map();
  const canonicalByRace = new Map();

  for (const segment of data?.segments ?? []) {
    for (const player of segment.stateSummary?.players ?? []) {
      const race = inferRaceFromUnitName(player.topUnits?.[0]?.name);
      if (race && !actualByRace.has(race)) {
        actualByRace.set(race, player.name);
      }
    }

    const notableLabels = [
      ...(segment.alignedRichSummary?.notableUnits ?? []).map((item) => item.label),
      ...((segment.alignedRichSummary?.chunks ?? []).flatMap((chunk) => (chunk.topUnits ?? []).map((item) => item.label))),
    ];

    for (const label of notableLabels) {
      const canonical = String(label ?? "").split(":")[0];
      const race = inferRaceFromNotableLabel(label);
      if (canonical.startsWith("[PLAYER_") && race && !canonicalByRace.has(race)) {
        canonicalByRace.set(race, canonical);
      }
    }
  }

  const aliasMap = new Map();
  for (const [race, actualName] of actualByRace.entries()) {
    const canonicalName = canonicalByRace.get(race);
    if (actualName && canonicalName) {
      aliasMap.set(actualName, canonicalName);
    }
  }

  return aliasMap;
}

function maskPlayerName(playerName) {
  return playerAliasMap.get(playerName) ?? playerName;
}

function formatDisplayPlayerLabel(playerName) {
  const normalized = String(playerName ?? "").trim();
  const match = normalized.match(/^\[?PLAYER_(\d+)\]?$/i);
  if (match) {
    return `PLAYER ${match[1]}`;
  }
  return normalized.replace(/^\[|\]$/g, "").replace(/_/g, " ");
}

function buildSegmentFromContextRecord(record, metadata, matchId) {
  const segmentId = Number(record.seg_index);
  const normalizedSpeech = replaceDatasetPlaceholders(record.speech, metadata);
  const tagName = record.speech_tag || "UNTAGGED";

  return {
    id: segmentId,
    clip: {
      path: buildClipUrl(matchId, record.clip_path, segmentId),
      name: `${matchId}_${String(segmentId).padStart(3, "0")}.mp4`,
      sizeBytes: null,
    },
    frames: parseFrameWindow(record.time),
    tag: {
      name: tagName,
      confidence: null,
      rationale: "",
      model: "",
    },
    commentary: {
      refined: normalizedSpeech,
      raw: normalizedSpeech,
    },
    alignedCompact: {
      speech: normalizedSpeech,
      time: record.time || "N/A",
      events: Array.isArray(record.events) ? record.events : [],
    },
  };
}

function buildMatchData(matchIndex, metadata, contextRecords) {
  const matchId = formatMatchId(matchIndex);
  const segments = contextRecords
    .map((record) => buildSegmentFromContextRecord(record, metadata, matchId))
    .sort((left, right) => left.id - right.id);

  return {
    match: {
      id: matchId,
      title: SITE_TITLE,
      gameInfo: metadata?.game_info ?? {},
      source: metadata?.source ?? {},
      sampleStats: {
        clipCount: segments.filter((segment) => Boolean(segment.clip.path)).length,
        tagCounts: buildTagCounts(contextRecords),
      },
    },
    segments,
  };
}

async function loadMatchData(matchIndex) {
  if (matchDataCache.has(matchIndex)) {
    return matchDataCache.get(matchIndex);
  }

  const matchId = formatMatchId(matchIndex);
  const [metadata, contextRecords] = await Promise.all([
    fetchJson(buildHfUrl(`${matchId}/metadata.json`)),
    fetchJson(buildHfUrl(`${matchId}/context.json`)),
  ]);

  const data = buildMatchData(matchIndex, metadata, contextRecords);
  matchDataCache.set(matchIndex, data);
  return data;
}

function buildCompactPlayerSummaries(events) {
  const playerMap = new Map();

  for (const event of events) {
    for (const [player, units] of Object.entries(event.units ?? {})) {
      if (!playerMap.has(player)) {
        playerMap.set(player, {
          player,
          eventCount: 0,
          unitGroupCount: 0,
          unitCounter: new Map(),
        });
      }

      const summary = playerMap.get(player);
      summary.eventCount += 1;
      summary.unitGroupCount += units.length;

      for (const entry of units) {
        const parsed = parseCompactUnitEntry(entry);
        summary.unitCounter.set(parsed.label, (summary.unitCounter.get(parsed.label) ?? 0) + parsed.count);
      }
    }
  }

  return [...playerMap.values()]
    .map((summary) => ({
      player: summary.player,
      eventCount: summary.eventCount,
      unitGroupCount: summary.unitGroupCount,
      topUnits: [...summary.unitCounter.entries()]
        .sort((left, right) => right[1] - left[1])
        .slice(0, 6)
        .map(([label, count]) => ({ label, count })),
    }))
    .sort((left, right) => left.player.localeCompare(right.player));
}

function buildCompactStateRows(events) {
  const rows = [];

  events.forEach((event, index) => {
    const unitsByPlayer = event.units ?? {};
    const players = Object.keys(unitsByPlayer);

    if (!players.length) {
      rows.push({
        eventIndex: index + 1,
        frames: event.frames || "N/A",
        viewport: formatCompactViewport(event.viewport),
        player: "N/A",
        units: [],
      });
      return;
    }

    players.forEach((player) => {
      rows.push({
        eventIndex: index + 1,
        frames: event.frames || "N/A",
        viewport: formatCompactViewport(event.viewport),
        player,
        units: unitsByPlayer[player] ?? [],
      });
    });
  });

  return rows;
}

function renderCompactStateRows(segment) {
  const rows = buildCompactStateRows(segment.alignedCompact?.events ?? []);

  if (!rows.length) {
    statePreviewEl.innerHTML = '<tr><td colspan="5">No compressed state events are available for this segment.</td></tr>';
    return;
  }

  statePreviewEl.innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td>${formatNumber(row.eventIndex)}</td>
          <td>${row.frames}</td>
          <td>${row.viewport}</td>
          <td>${formatDisplayPlayerLabel(row.player)}</td>
          <td>
            <div class="state-unit-list">
              ${
                row.units.length
                  ? row.units.map((unit) => `<span class="state-unit-item">${unit}</span>`).join("")
                  : '<span class="state-unit-item">No visible units recorded.</span>'
              }
            </div>
          </td>
        </tr>
      `,
    )
    .join("");
}

function createMetaChip(label, value) {
  return `
    <article class="meta-chip">
      <p class="meta-chip-label">${label}</p>
      <p class="meta-chip-value">${value}</p>
    </article>
  `;
}

function scrollActiveButtonIntoView(container, selector, behavior = "smooth", alignment = "center") {
  const activeButton = container?.querySelector(selector);
  if (!activeButton) {
    return;
  }

  const computedStyle = getComputedStyle(container);
  const paddingLeft = Number.parseFloat(computedStyle.paddingLeft) || 0;
  const maxScrollLeft = Math.max(0, container.scrollWidth - container.clientWidth);
  const inset = 14;
  let targetLeft;

  if (alignment === "minimal") {
    const buttonLeft = activeButton.offsetLeft - container.scrollLeft;
    const buttonRight = buttonLeft + activeButton.clientWidth;
    if (buttonLeft >= inset && buttonRight <= container.clientWidth - inset) {
      return;
    }

    if (buttonLeft < inset) {
      targetLeft = container.scrollLeft + buttonLeft - inset;
    } else {
      targetLeft = container.scrollLeft + (buttonRight - (container.clientWidth - inset));
    }
  } else if (alignment === "start") {
    targetLeft = activeButton.offsetLeft - paddingLeft;
  } else {
    targetLeft = activeButton.offsetLeft - (container.clientWidth - activeButton.clientWidth) / 2;
  }

  container.scrollTo({
    left: Math.min(maxScrollLeft, Math.max(0, targetLeft)),
    behavior,
  });
}

function enableDragScroll(container) {
  if (!container || container.dataset.dragScrollReady === "true") {
    return;
  }

  container.dataset.dragScrollReady = "true";

  let isPointerDown = false;
  let hasDragged = false;
  let pointerId = null;
  let startX = 0;
  let startScrollLeft = 0;

  const finishDrag = () => {
    isPointerDown = false;
    pointerId = null;
    container.classList.remove("is-dragging");
    window.setTimeout(() => {
      hasDragged = false;
    }, 0);
  };

  container.addEventListener("pointerdown", (event) => {
    if (event.pointerType === "mouse" && event.button !== 0) {
      return;
    }

    if (event.target.closest("button, select, input, textarea, label, video")) {
      hasDragged = false;
      return;
    }

    isPointerDown = true;
    hasDragged = false;
    pointerId = event.pointerId;
    startX = event.clientX;
    startScrollLeft = container.scrollLeft;
    container.classList.add("is-dragging");
    container.setPointerCapture?.(event.pointerId);
  });

  container.addEventListener("pointermove", (event) => {
    if (!isPointerDown || (pointerId !== null && event.pointerId !== pointerId)) {
      return;
    }

    const delta = event.clientX - startX;
    if (Math.abs(delta) > 5) {
      hasDragged = true;
    }

    container.scrollLeft = startScrollLeft - delta;
  });

  container.addEventListener("pointerup", finishDrag);
  container.addEventListener("pointercancel", finishDrag);
  container.addEventListener("lostpointercapture", finishDrag);

  container.addEventListener(
    "click",
    (event) => {
      if (!hasDragged) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();
    },
    true,
  );
}

function getVisibleSegments() {
  if (!state.data) {
    return [];
  }

  if (state.filteredTag === "ALL") {
    return state.data.segments;
  }

  return state.data.segments.filter((segment) => segment.tag.name === state.filteredTag);
}

function getActiveSegment() {
  return state.data?.segments.find((segment) => segment.id === state.activeSegmentId) ?? null;
}

function ensureVisibleActiveSegment() {
  const visible = getVisibleSegments();
  if (visible.length && !visible.some((segment) => segment.id === state.activeSegmentId)) {
    state.activeSegmentId = visible[0].id;
  }
}

function setActiveTab(nextTab, updateHash = true) {
  if (!VALID_TABS.has(nextTab)) {
    return;
  }

  state.activeTab = nextTab;

  for (const button of tabButtons) {
    const isActive = button.dataset.tabTrigger === nextTab;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-selected", String(isActive));
  }

  for (const panel of tabPanels) {
    const isActive = panel.dataset.tabPanel === nextTab;
    panel.classList.toggle("is-active", isActive);
    panel.hidden = !isActive;
  }

  if (updateHash) {
    history.replaceState(null, "", `#${nextTab}`);
  }

  scheduleChartRender(nextTab);
}

function buildChartCommonOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 360,
      easing: "easeOutQuart",
    },
    plugins: {
      legend: {
        labels: {
          color: CHART_COLORS.text,
          font: {
            family: chartFontFamily,
            size: 12,
            weight: "700",
          },
          boxWidth: 12,
          boxHeight: 12,
        },
      },
      tooltip: {
        backgroundColor: CHART_COLORS.tooltip,
        titleColor: "#ffffff",
        bodyColor: "#ffffff",
        padding: 10,
        titleFont: {
          family: chartFontFamily,
          size: 12,
          weight: "700",
        },
        bodyFont: {
          family: chartFontFamily,
          size: 12,
          weight: "600",
        },
      },
    },
  };
}

function buildAxisOptions({ beginAtZero = true, min, max, stepSize, tickFormatter } = {}) {
  return {
    beginAtZero,
    min,
    max,
    ticks: {
      color: CHART_COLORS.muted,
      font: {
        family: chartFontFamily,
        size: 11,
        weight: "700",
      },
      padding: 8,
      callback: tickFormatter,
      stepSize,
    },
    grid: {
      color: CHART_COLORS.grid,
      tickLength: 0,
    },
    border: {
      display: false,
    },
  };
}

function ensureChart(handleKey, canvasId, createChart) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !window.Chart) {
    return;
  }

  if (chartHandles[handleKey]) {
    chartHandles[handleKey].resize();
    return;
  }

  const context = canvas.getContext("2d");
  chartHandles[handleKey] = createChart(context);
}

function renderOverviewTagChart() {
  ensureChart("overviewTag", "overview-tag-chart", (context) => {
    return new window.Chart(context, {
      type: "bar",
      data: {
        labels: TAG_DISTRIBUTION.map((item) => item.tag),
        datasets: [
          {
            data: TAG_DISTRIBUTION.map((item) => item.count),
            backgroundColor: TAG_DISTRIBUTION.map(() => CHART_COLORS.dark),
            hoverBackgroundColor: CHART_COLORS.ink,
            borderRadius: 0,
            maxBarThickness: 26,
          },
        ],
      },
      options: {
        ...buildChartCommonOptions(),
        indexAxis: "y",
        plugins: {
          ...buildChartCommonOptions().plugins,
          legend: {
            display: false,
          },
          tooltip: {
            ...buildChartCommonOptions().plugins.tooltip,
            callbacks: {
              label: (tooltipItem) => `${formatNumber(tooltipItem.raw)} segments`,
            },
          },
        },
        scales: {
          x: buildAxisOptions({
            tickFormatter: (value) => formatNumber(value),
          }),
          y: {
            ticks: {
              color: CHART_COLORS.text,
              font: {
                family: chartFontFamily,
                size: 11,
                weight: "700",
              },
            },
            grid: {
              display: false,
            },
            border: {
              display: false,
            },
          },
        },
      },
    });
  });
}

function renderOverviewRaceChart() {
  ensureChart("overviewRace", "overview-race-chart", (context) => {
    return new window.Chart(context, {
      type: "pie",
      data: {
        labels: RACE_DISTRIBUTION.map((item) => item.label),
        datasets: [
          {
            data: RACE_DISTRIBUTION.map((item) => item.count),
            backgroundColor: [CHART_COLORS.dark, CHART_COLORS.highlight, CHART_COLORS.soft],
            hoverBackgroundColor: [CHART_COLORS.ink, "#af6227", "#849cb5"],
            borderWidth: 0,
          },
        ],
      },
      options: {
        ...buildChartCommonOptions(),
        plugins: {
          ...buildChartCommonOptions().plugins,
          legend: {
            position: "bottom",
            labels: {
              color: CHART_COLORS.text,
              usePointStyle: true,
              pointStyle: "circle",
              boxWidth: 10,
              boxHeight: 10,
              padding: 14,
              font: {
                family: chartFontFamily,
                size: 11,
                weight: "700",
              },
            },
          },
          tooltip: {
            ...buildChartCommonOptions().plugins.tooltip,
            callbacks: {
              label: (tooltipItem) => `${tooltipItem.label}: ${formatNumber(tooltipItem.raw)} matches`,
            },
          },
        },
      },
    });
  });
}

function renderOverviewMatchupChart() {
  ensureChart("overviewMatchup", "overview-matchup-chart", (context) => {
    return new window.Chart(context, {
      type: "bar",
      data: {
        labels: MATCHUP_DISTRIBUTION.map((item) => item.label),
        datasets: [
          {
            data: MATCHUP_DISTRIBUTION.map((item) => item.count),
            backgroundColor: MATCHUP_DISTRIBUTION.map(() => CHART_COLORS.dark),
            hoverBackgroundColor: MATCHUP_DISTRIBUTION.map(() => CHART_COLORS.ink),
            borderRadius: 0,
            maxBarThickness: 22,
          },
        ],
      },
      options: {
        ...buildChartCommonOptions(),
        indexAxis: "y",
        plugins: {
          ...buildChartCommonOptions().plugins,
          legend: {
            display: false,
          },
          tooltip: {
            ...buildChartCommonOptions().plugins.tooltip,
            callbacks: {
              label: (tooltipItem) => `${formatNumber(tooltipItem.raw)} matches`,
            },
          },
        },
        scales: {
          x: buildAxisOptions({
            tickFormatter: (value) => formatNumber(value),
          }),
          y: {
            ticks: {
              color: CHART_COLORS.text,
              font: {
                family: chartFontFamily,
                size: 11,
                weight: "700",
              },
            },
            grid: {
              display: false,
            },
            border: {
              display: false,
            },
          },
        },
      },
    });
  });
}

function getTask1MetricRange(values) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min;
  const baselinePad = span === 0 ? Math.max(Math.abs(max) * 0.08, 0.05) : span * 0.16;

  return {
    min: Math.max(0, min - baselinePad),
    max: max + baselinePad,
  };
}

function renderTask1MetricChart(metric) {
  const values = TASK1_RESULTS.map((item) => item[metric.key]);
  const range = getTask1MetricRange(values);

  ensureChart(metric.handleKey, metric.canvasId, (context) => {
    return new window.Chart(context, {
      type: "bar",
      data: {
        labels: TASK1_RESULTS.map((item) => item.shortLabel),
        datasets: [
          {
            label: metric.label,
            data: values,
            backgroundColor: TASK1_MODEL_COLORS,
            borderRadius: 0,
            maxBarThickness: 38,
          },
        ],
      },
      options: {
        ...buildChartCommonOptions(),
        plugins: {
          ...buildChartCommonOptions().plugins,
          legend: {
            display: false,
          },
          tooltip: {
            ...buildChartCommonOptions().plugins.tooltip,
            callbacks: {
              label: (tooltipItem) => {
                const fullModelName = TASK1_RESULTS[tooltipItem.dataIndex]?.model ?? tooltipItem.label;
                return `${fullModelName}: ${Number(tooltipItem.raw).toFixed(metric.tooltipPrecision)}`;
              },
            },
          },
        },
        scales: {
          x: {
            ticks: {
              color: CHART_COLORS.text,
              font: {
                family: chartFontFamily,
                size: 11,
                weight: "700",
              },
              maxRotation: 0,
            },
            grid: {
              display: false,
            },
            border: {
              display: false,
            },
          },
          y: buildAxisOptions({
            beginAtZero: false,
            min: range.min,
            max: range.max,
            tickFormatter: (value) => Number(value).toFixed(metric.tickPrecision),
          }),
        },
      },
    });
  });
}

function renderBenchmarkTask1Charts() {
  for (const metric of TASK1_METRICS) {
    renderTask1MetricChart(metric);
  }
}

function renderTask2MetricChart(handleKey, canvasId, label, metricKey, range) {
  ensureChart(handleKey, canvasId, (context) => {
    return new window.Chart(context, {
      type: "bar",
      data: {
        labels: TASK2_OVERALL.map((item) => item.modality),
        datasets: [
          {
            label,
            data: TASK2_OVERALL.map((item) => item[metricKey]),
            backgroundColor: [CHART_COLORS.highlight, CHART_COLORS.dark, CHART_COLORS.soft],
            borderRadius: 0,
            maxBarThickness: 44,
          },
        ],
      },
      options: {
        ...buildChartCommonOptions(),
        plugins: {
          ...buildChartCommonOptions().plugins,
          legend: {
            display: false,
          },
        },
        scales: {
          x: {
            ticks: {
              color: CHART_COLORS.text,
              font: {
                family: chartFontFamily,
                size: 11,
                weight: "700",
              },
              maxRotation: 0,
            },
            grid: {
              display: false,
            },
            border: {
              display: false,
            },
          },
          y: buildAxisOptions({
            beginAtZero: false,
            min: range.min,
            max: range.max,
            stepSize: range.step,
            tickFormatter: (value) => Number(value).toFixed(range.precision ?? 1),
          }),
        },
      },
    });
  });
}

function formatTask2MetricValue(value) {
  return Number(value).toFixed(2);
}

function buildTask2ResultsRow(row) {
  const metricKeys = ["b4", "rl", "bs", "tc", "sc", "cn"];
  const modalities = ["multimodal", "videoOnly", "textOnly"];
  const bestByMetric = new Map();

  for (const metricKey of metricKeys) {
    const values = modalities.map((modalityKey) => row[modalityKey][metricKey]);
    bestByMetric.set(metricKey, Math.max(...values));
  }

  const cells = modalities
    .map((modalityKey) =>
      metricKeys
        .map((metricKey) => {
          const value = row[modalityKey][metricKey];
          const isBest = value === bestByMetric.get(metricKey);
          return `<td class="${isBest ? "table-best" : ""}">${formatTask2MetricValue(value)}</td>`;
        })
        .join("")
    )
    .join("");

  return `<tr${row.tag === "Overall" ? ' class="is-overall"' : ""}><td>${row.tag}</td>${cells}</tr>`;
}

function renderTask2ResultsTable() {
  if (!task2ResultsBodyEl) {
    return;
  }

  task2ResultsBodyEl.innerHTML = TASK2_RESULTS.map((row) => buildTask2ResultsRow(row)).join("");
}

function renderChartsForTab(tabName) {
  if (!window.Chart) {
    return;
  }

  if (tabName === "overview") {
    renderOverviewTagChart();
    renderOverviewRaceChart();
    renderOverviewMatchupChart();
  }

  if (tabName === "benchmark") {
    renderBenchmarkTask1Charts();
    renderTask2MetricChart("benchmarkTask2B4", "benchmark-task2-b4-chart", "BLEU-4", "b4", { min: 2.6, max: 3.5, step: 0.2, precision: 1 });
    renderTask2MetricChart("benchmarkTask2Rl", "benchmark-task2-rl-chart", "ROUGE-L", "rl", { min: 12.6, max: 14.4, step: 0.4, precision: 1 });
    renderTask2MetricChart("benchmarkTask2Bs", "benchmark-task2-bs-chart", "BERTScore", "bs", { min: 85.7, max: 86.5, step: 0.2, precision: 1 });
    renderTask2MetricChart("benchmarkTask2Tc", "benchmark-task2-tc-chart", "Tag Consistency", "tc", { min: 4.1, max: 4.2, step: 0.02, precision: 2 });
    renderTask2MetricChart("benchmarkTask2Sc", "benchmark-task2-sc-chart", "Strategic Correctness", "sc", { min: 1.64, max: 1.7, step: 0.01, precision: 2 });
    renderTask2MetricChart("benchmarkTask2Cn", "benchmark-task2-cn-chart", "Caster Naturalness", "cn", { min: 3.96, max: 4.0, step: 0.01, precision: 2 });
  }
}

function getChartHandlesForTab(tabName) {
  if (tabName === "overview") {
    return [chartHandles.overviewTag, chartHandles.overviewRace, chartHandles.overviewMatchup].filter(Boolean);
  }

  if (tabName === "benchmark") {
    return [
      chartHandles.benchmarkTask1Format,
      chartHandles.benchmarkTask1UnitF1,
      chartHandles.benchmarkTask1TIoU,
      chartHandles.benchmarkTask1CountMae,
      chartHandles.benchmarkTask1VpL2,
      chartHandles.benchmarkTask1UnitL2,
      chartHandles.benchmarkTask2B4,
      chartHandles.benchmarkTask2Rl,
      chartHandles.benchmarkTask2Bs,
      chartHandles.benchmarkTask2Tc,
      chartHandles.benchmarkTask2Sc,
      chartHandles.benchmarkTask2Cn,
    ].filter(Boolean);
  }

  return [];
}

function refreshChartsForTab(tabName) {
  for (const chart of getChartHandlesForTab(tabName)) {
    chart.resize();
    chart.update("none");
  }
}

function scheduleChartRender(tabName = state.activeTab) {
  if (!window.Chart) {
    return;
  }

  if (chartRenderFrame) {
    window.cancelAnimationFrame(chartRenderFrame);
  }

  if (chartRefreshFrame) {
    window.cancelAnimationFrame(chartRefreshFrame);
  }

  if (chartRefreshTimeout) {
    window.clearTimeout(chartRefreshTimeout);
  }

  chartRenderFrame = window.requestAnimationFrame(() => {
    renderChartsForTab(tabName);

    chartRefreshFrame = window.requestAnimationFrame(() => {
      refreshChartsForTab(tabName);
    });

    chartRefreshTimeout = window.setTimeout(() => {
      refreshChartsForTab(tabName);
    }, 120);
  });
}

function resetClipPlayer() {
  clipPlayerEl.pause();
  clipPlayerEl.removeAttribute("src");
  clipPlayerEl.load();
}

function setDatasetLoading(message) {
  clipTitleEl.textContent = "Loading match";
  clipTagPillEl.textContent = "REMOTE";
  resetClipPlayer();
  tagFilterEl.disabled = true;
  tagFilterEl.innerHTML = '<option value="ALL">Loading…</option>';
  segmentListEl.innerHTML = `<div class="empty-state">${message}</div>`;
  refinedCommentaryEl.textContent = message;
  stateGridEl.innerHTML = `<div class="empty-state">${message}</div>`;
  statePreviewEl.innerHTML = '<tr><td colspan="5">Loading…</td></tr>';
}

function setDatasetError(message) {
  clipTitleEl.textContent = "Match unavailable";
  clipTagPillEl.textContent = "ERROR";
  resetClipPlayer();
  tagFilterEl.disabled = true;
  tagFilterEl.innerHTML = '<option value="ALL">Unavailable</option>';
  segmentListEl.innerHTML = `<div class="empty-state">${message}</div>`;
  refinedCommentaryEl.textContent = message;
  stateGridEl.innerHTML = `<div class="empty-state">${message}</div>`;
  statePreviewEl.innerHTML = `<tr><td colspan="5">${message}</td></tr>`;
}

function renderMatchSelector(alignActive = true) {
  const buttons = [];

  for (let index = 1; index <= TOTAL_MATCHES; index += 1) {
    const isActive = index === state.activeMatch;

    buttons.push(`
      <button
        class="match-button${isActive ? " is-active" : ""}"
        type="button"
        data-match-index="${index}"
      >
        ${String(index).padStart(3, "0")}
      </button>
    `);
  }

  matchSelectorEl.innerHTML = buttons.join("");
  if (alignActive) {
    scrollActiveButtonIntoView(matchSelectorEl, ".match-button.is-active", "auto", "minimal");
  }
}

function renderTagFilter() {
  const options = ['<option value="ALL">All</option>'];

  for (const item of state.data.match.sampleStats.tagCounts) {
    const selected = item.tag === state.filteredTag ? " selected" : "";
    options.push(`<option value="${item.tag}"${selected}>${item.tag} (${item.count})</option>`);
  }

  tagFilterEl.disabled = false;
  tagFilterEl.innerHTML = options.join("");
}

function renderSegmentList() {
  const visible = getVisibleSegments();

  if (!visible.length) {
    segmentListEl.innerHTML = '<div class="empty-state">No segments match the current filter.</div>';
    return;
  }

  segmentListEl.innerHTML = visible
    .map((segment) => {
      const segmentNumber = String(segment.id).padStart(3, "0");
      const segmentLabel = segment.tag.name;

      return `
        <button
          class="segment-button${segment.id === state.activeSegmentId ? " is-active" : ""}"
          type="button"
          data-segment-id="${segment.id}"
          aria-label="Segment ${segmentNumber} ${segmentLabel}"
          title="${segmentLabel}"
        >
          <span class="segment-index">${segmentNumber}</span>
          ${segment.id === state.activeSegmentId ? `<span class="segment-chip-label">${segmentLabel}</span>` : ""}
        </button>
      `;
    })
    .join("");

  scrollActiveButtonIntoView(segmentListEl, ".segment-button.is-active", "auto", "minimal");
}

function renderStateSummary(segment) {
  const compact = segment.alignedCompact ?? {};
  const events = Array.isArray(compact.events) ? compact.events : [];
  const playerSummaries = buildCompactPlayerSummaries(events);
  const visiblePlayers = playerSummaries.map((item) => formatDisplayPlayerLabel(item.player));
  const uniqueViewports = [...new Set(events.map((event) => event.viewport).filter(Boolean))];

  if (!events.length) {
    stateGridEl.innerHTML = '<div class="empty-state">No compressed state observations are available for this segment.</div>';
  } else {
    stateGridEl.innerHTML = `
      <article class="state-card">
        <p><strong>Frames</strong> ${compact.time || "N/A"}</p>
        <p><strong>Events</strong> ${formatNumber(events.length)}</p>
        <p><strong>Players</strong> ${joinValues(visiblePlayers, "None")}</p>
        <p><strong>Viewports</strong> ${formatNumber(uniqueViewports.length)}</p>
      </article>
      ${playerSummaries
        .map((player) => {
          const topUnits = player.topUnits.length
            ? player.topUnits.map((unit) => `<li>${unit.label} x${formatNumber(unit.count)}</li>`).join("")
            : "<li>No unit counts available.</li>";

          return `
            <article class="state-card">
              <div class="state-card-head">
                <span class="state-player-tag">${formatDisplayPlayerLabel(player.player)}</span>
              </div>
              <p><strong>Event mentions</strong> ${formatNumber(player.eventCount)}</p>
              <p><strong>Unit groups</strong> ${formatNumber(player.unitGroupCount)}</p>
              <ul class="flat-list">${topUnits}</ul>
            </article>
          `;
        })
        .join("")}
    `;
  }

  renderCompactStateRows(segment);
}

function renderSelectedSegment({ autoplayClip = false } = {}) {
  const segment = getActiveSegment();
  if (!segment) {
    return;
  }

  clipTitleEl.textContent = `Segment ${String(segment.id).padStart(3, "0")}`;
  clipTagPillEl.textContent = segment.tag.name;

  const normalizedClipPath = segment.clip.path.replace("./", "/");
  const hasDifferentSource =
    !clipPlayerEl.src.endsWith(normalizedClipPath) && !clipPlayerEl.src.endsWith(segment.clip.path.replace("./", ""));

  if (hasDifferentSource) {
    clipPlayerEl.pause();
    clipPlayerEl.src = segment.clip.path;
  }

  if (autoplayClip) {
    clipPlayerEl.currentTime = 0;
    const playPromise = clipPlayerEl.play();
    if (typeof playPromise?.catch === "function") {
      playPromise.catch(() => {});
    }
  }

  refinedCommentaryEl.textContent = segment.commentary.refined || "No commentary available.";

  renderStateSummary(segment);
}

function render({ refreshMatchSelector = false, alignMatchSelector = false, autoplayClip = false } = {}) {
  ensureVisibleActiveSegment();
  if (refreshMatchSelector) {
    renderMatchSelector(alignMatchSelector);
  }
  renderTagFilter();
  renderSegmentList();
  renderSelectedSegment({ autoplayClip });
}

async function loadAndRenderMatch(matchIndex, { alignMatchSelector = true, autoplayClip = false } = {}) {
  const requestId = ++activeMatchRequestId;
  const matchId = formatMatchId(matchIndex);

  state.activeMatch = matchIndex;
  state.filteredTag = "ALL";
  state.activeSegmentId = 0;
  renderMatchSelector(alignMatchSelector);
  setDatasetLoading(`Loading ${matchId} from Hugging Face…`);

  try {
    const data = await loadMatchData(matchIndex);

    if (requestId !== activeMatchRequestId) {
      return;
    }

    state.data = data;
    playerAliasMap = buildPlayerAliasMap(state.data);
    render({ autoplayClip });
  } catch (error) {
    if (requestId !== activeMatchRequestId) {
      return;
    }

    console.error(error);
    state.data = null;
    setDatasetError(`Failed to load ${matchId} from Hugging Face.`);
  }
}

function moveSelection(direction) {
  const visible = getVisibleSegments();
  const currentIndex = visible.findIndex((segment) => segment.id === state.activeSegmentId);
  const nextIndex = currentIndex + direction;

  if (nextIndex < 0 || nextIndex >= visible.length) {
    return;
  }

  state.activeSegmentId = visible[nextIndex].id;
  render();
}

matchSelectorEl.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-match-index]");
  if (!button) {
    return;
  }

  const nextMatch = Number(button.dataset.matchIndex);
  if (nextMatch === state.activeMatch && state.data) {
    renderMatchSelector(true);
    return;
  }

  await loadAndRenderMatch(nextMatch, { alignMatchSelector: true });
});

segmentListEl.addEventListener("click", (event) => {
  const button = event.target.closest("[data-segment-id]");
  if (!button) {
    return;
  }

  state.activeSegmentId = Number(button.dataset.segmentId);
  render({ autoplayClip: true });
});

tagFilterEl.addEventListener("change", (event) => {
  state.filteredTag = event.target.value;
  render();
});

for (const button of tabButtons) {
  button.addEventListener("click", () => setActiveTab(button.dataset.tabTrigger));
}

window.addEventListener("load", () => {
  scheduleChartRender(state.activeTab);
});

window.addEventListener("resize", () => {
  scheduleChartRender(state.activeTab);
});

window.addEventListener("keydown", (event) => {
  if (state.activeTab !== "dataset") {
    return;
  }

  if (event.key === "ArrowLeft") {
    moveSelection(-1);
  }

  if (event.key === "ArrowRight") {
    moveSelection(1);
  }
});

async function init() {
  try {
    document.title = SITE_TITLE;
    enableDragScroll(matchSelectorEl);
    enableDragScroll(segmentListEl);
    renderTask2ResultsTable();
    renderMatchSelector(true);
    setDatasetLoading(`Loading ${formatMatchId(state.activeMatch)} from Hugging Face…`);
    await loadAndRenderMatch(state.activeMatch, { alignMatchSelector: true });

    const initialTab = VALID_TABS.has(location.hash.slice(1)) ? location.hash.slice(1) : "overview";
    setActiveTab(initialTab, false);
    scheduleChartRender(initialTab);
  } catch (error) {
    console.error(error);
  }
}

init();


