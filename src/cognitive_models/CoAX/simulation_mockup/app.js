const datasets = {
  wine_quality: {
    label: "Wine Quality",
    appId: "wine_quality",
    modelName: "mlp",
    expMethod: "lime",
    blocks: [
      {
        testWithXAI: [0, 4, 6, 22, 27, 36, 41, 42, 46, 62, 64, 65, 80, 86, 98, 101, 111, 117],
        testWithoutXAI: [3, 13, 31, 44, 45, 52, 59, 67, 74, 75, 76, 78, 90, 91, 97, 105, 106, 110],
      },
      {
        testWithXAI: [20, 29, 34, 38, 39, 47, 56, 60, 71, 84, 88, 96, 100, 102, 103, 107, 112, 118],
        testWithoutXAI: [1, 5, 14, 19, 21, 23, 26, 35, 55, 61, 68, 69, 77, 83, 108, 113, 116, 119],
      },
    ],
  },
  forest_cover: {
    label: "Forest Cover",
    appId: "forest_cover",
    modelName: "xgboost",
    expMethod: "shap",
    blocks: [
      {
        testWithXAI: [21, 61, 102, 110, 130, 137, 151, 152, 179, 217, 223, 234, 239, 247, 270, 273, 278, 290],
        testWithoutXAI: [8, 17, 22, 32, 53, 73, 81, 86, 95, 118, 122, 145, 172, 219, 220, 256, 260, 291],
      },
      {
        testWithXAI: [0, 44, 48, 65, 101, 135, 136, 139, 167, 175, 201, 207, 233, 236, 245, 246, 287, 288],
        testWithoutXAI: [2, 9, 20, 41, 42, 70, 82, 89, 91, 94, 107, 109, 149, 177, 194, 205, 226, 274],
      },
    ],
  },
  adult: {
    label: "Adult Income",
    appId: "adult",
    modelName: "xgboost",
    expMethod: "lime",
    blocks: [
      {
        testWithXAI: [2, 33, 35, 75, 76, 84, 95, 117, 125, 135, 158, 172, 190, 194, 210, 235, 246, 261],
        testWithoutXAI: [5, 6, 18, 58, 81, 132, 145, 156, 161, 165, 171, 179, 221, 275, 276, 277, 294, 296],
      },
      {
        testWithXAI: [3, 19, 23, 26, 29, 66, 78, 97, 101, 114, 121, 150, 184, 207, 232, 267, 281, 282],
        testWithoutXAI: [4, 20, 49, 50, 89, 94, 103, 111, 120, 224, 240, 242, 243, 268, 269, 290, 291, 298],
      },
    ],
  },
};

const demoCases = [
  {
    label: "Case 1",
    dataset: "adult",
    xaiType: "Attribution",
    tested: "w/ XAI",
    reasoningStrategy: "Attribution Sum",
    instanceId: 95,
    blockIndex: 0,
    session: 1,
    k: 3,
    retrievalThreshold: -1.8,
    sensitivity: 10,
    scalingFactor: 3,
  },
  {
    label: "Case 2",
    dataset: "adult",
    xaiType: "Importance",
    tested: "w/ XAI",
    reasoningStrategy: "Salient-features categorization",
    instanceId: 95,
    blockIndex: 0,
    session: 1,
    k: 3,
    retrievalThreshold: -2.3,
    sensitivity: 4,
    scalingFactor: 3,
  },
  {
    label: "Case 3",
    dataset: "forest_cover",
    xaiType: "Attribution",
    tested: "w/ XAI",
    reasoningStrategy: "Attribution Sum",
    instanceId: 44,
    blockIndex: 1,
    session: 2,
    k: 1,
    retrievalThreshold: -2.3,
    sensitivity: 10,
    scalingFactor: 1,
  },
  {
    label: "Case 4",
    dataset: "wine_quality",
    xaiType: "Attribution",
    tested: "w/o XAI",
    reasoningStrategy: "Attribution Sum",
    instanceId: 69,
    blockIndex: 1,
    session: 2,
    k: 2,
    retrievalThreshold: -1.8,
    sensitivity: 10,
    scalingFactor: 5,
  },
  {
    label: "Case 5",
    dataset: "adult",
    xaiType: "None",
    tested: "w/o XAI",
    reasoningStrategy: "Sensitive-features categorization",
    instanceId: 29,
    blockIndex: 1,
    session: 2,
    k: 3,
    retrievalThreshold: -2.3,
    sensitivity: 1,
    scalingFactor: 3,
  },
];

const reasoningStrategiesByXaiType = {
  None: [
    ["Sensitive-features categorization", "Sensitive-Features Categorization"],
  ],
  Importance: [
    ["Sensitive-features categorization", "Sensitive-Features Categorization"],
    ["Salient-features categorization", "Salient-Features Categorization"],
    ["Importance categorization", "Importance Categorization"],
    ["Attribution Sum", "Attribution Sum"],
  ],
  Attribution: [
    ["Sensitive-features categorization", "Sensitive-Features Categorization"],
    ["Attribution Sum", "Attribution Sum"],
  ],
};

const testedConditionsByXaiType = {
  None: [["w/o XAI", "Without XAI"]],
  Importance: [
    ["w/ XAI", "With XAI"],
    ["w/o XAI", "Without XAI"],
  ],
  Attribution: [
    ["w/ XAI", "With XAI"],
    ["w/o XAI", "Without XAI"],
  ],
};

const $ = (id) => document.getElementById(id);
let latestPredictionRows = [];

function selectedDemoCase() {
  return demoCases[Number($("case-number").value) || 0];
}

function toggleSidebar() {
  const shell = document.querySelector(".app-shell");
  const button = $("sidebar-toggle");
  const icon = button?.querySelector("i");
  if (!shell || !button || !icon) return;

  const isCollapsed = shell.classList.toggle("sidebar-collapsed");
  button.setAttribute("aria-expanded", String(!isCollapsed));
  button.setAttribute("aria-label", isCollapsed ? "Expand sidebar" : "Collapse sidebar");
  icon.className = isCollapsed
    ? "ti ti-layout-sidebar-left-expand"
    : "ti ti-layout-sidebar-left-collapse";

  if (isCollapsed) {
    shell.style.removeProperty("--panel-width");
  }
}

function routePrefix() {
  return window.location.pathname.startsWith("/api/") ? "/api" : "";
}

function currentTestList() {
  const ds = datasets[$("dataset").value];
  if (!ds) return [];

  const listKey = $("tested").value === "w/ XAI" ? "testWithXAI" : "testWithoutXAI";
  const lists = ds.blocks.map(block => block[listKey]);

  if ($("xai-type").value === "None") {
    lists.push(...ds.blocks.map(block => block.testWithXAI));
  }

  return Array.from(new Set(lists.flat())).sort((a, b) => a - b);
}

function currentInstanceId() {
  return Number($("instance-number").value);
}

function currentConditionInstanceIndex() {
  const instanceId = currentInstanceId();
  const index = currentTestList().indexOf(instanceId);
  return index >= 0 ? index : 0;
}

function currentInstanceLocation() {
  const ds = datasets[$("dataset").value];
  const instanceId = currentInstanceId();
  const tested = $("tested").value;
  const keys = tested === "w/ XAI"
    ? ["testWithXAI"]
    : $("xai-type").value === "None"
      ? ["testWithoutXAI", "testWithXAI"]
      : ["testWithoutXAI"];

  for (let blockIndex = 0; blockIndex < ds.blocks.length; blockIndex += 1) {
    for (const key of keys) {
      const instanceNumber = ds.blocks[blockIndex][key].indexOf(instanceId);
      if (instanceNumber >= 0) {
        return { blockIndex, session: blockIndex + 1, instanceNumber };
      }
    }
  }

  return { blockIndex: 0, session: 1, instanceNumber: currentConditionInstanceIndex() };
}

function displayInstanceNumber(instanceId) {
  return Number(instanceId) + 1;
}

function displayCaseNumber() {
  return Number($("case-number").value) + 1;
}

function displayInstanceStatus(instanceId = currentInstanceId()) {
  return `Case ${displayCaseNumber()} · Instance ${displayInstanceNumber(instanceId)}`;
}

function populateDatasets() {
  const select = $("dataset");
  Object.entries(datasets).forEach(([value, config]) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = config.label;
    select.appendChild(option);
  });
}

function populateCases() {
  const select = $("case-number");
  const previous = Number(select.value || 0);

  select.replaceChildren();
  demoCases.forEach((demoCase, index) => {
    const option = document.createElement("option");
    option.value = index;
    option.textContent = `Case ${index + 1}`;
    option.title = `${demoCase.label}: ${datasets[demoCase.dataset].label}`;
    select.appendChild(option);
  });
  select.value = String(Math.min(previous, demoCases.length - 1));
}

function populateInstances(preferredInstanceId = currentInstanceId()) {
  const select = $("instance-number");
  const list = currentTestList();
  const fallback = list[0];

  select.replaceChildren();
  list.forEach((instanceId) => {
    const option = document.createElement("option");
    option.value = String(instanceId);
    option.textContent = `Instance ${displayInstanceNumber(instanceId)}`;
    select.appendChild(option);
  });

  const nextValue = list.includes(Number(preferredInstanceId))
    ? Number(preferredInstanceId)
    : fallback;
  if (nextValue !== undefined) {
    select.value = String(nextValue);
  }
}

function setControlValue(id, value) {
  const control = $(id);
  if (!control) return;
  control.value = String(value);
  control.dispatchEvent(new Event("input", { bubbles: true }));
}

function applyDemoCase() {
  const demoCase = selectedDemoCase();
  if (!demoCase) return;

  $("dataset").value = demoCase.dataset;
  $("xai-type").value = demoCase.xaiType;
  updateXaiTypeLabels();
  updateTestedOptions();
  $("tested").value = demoCase.tested;
  updateReasoningStrategies();
  $("reasoning-strategy").value = demoCase.reasoningStrategy;
  populateInstances(demoCase.instanceId);

  setControlValue("retrieval-threshold", demoCase.retrievalThreshold);
  setControlValue("sensitivity", demoCase.sensitivity);
  setControlValue("k", demoCase.k);
  setControlValue("scaling-factor", demoCase.scalingFactor);
}

function updateXaiTypeLabels() {
  const ds = datasets[$("dataset").value];
  if (!ds) return;

  const method = ds.expMethod.toUpperCase();
  const labels = {
    Importance: `Importance (${method})`,
    Attribution: `Attribution (${method})`,
    None: "None",
  };

  Array.from($("xai-type").options).forEach((option) => {
    option.textContent = labels[option.value] || option.value;
  });
}

function availableReasoningStrategies(xaiType, tested) {
  return (reasoningStrategiesByXaiType[xaiType] || reasoningStrategiesByXaiType.None).filter(([value]) => {
    if (xaiType === "Importance" && tested === "w/o XAI" && value === "Importance categorization") {
      return false;
    }
    if (xaiType === "Attribution" && tested === "w/ XAI" && value === "Sensitive-features categorization") {
      return false;
    }
    return true;
  });
}

function updateTestedOptions() {
  const select = $("tested");
  const xaiType = $("xai-type").value;
  const conditions = testedConditionsByXaiType[xaiType] || testedConditionsByXaiType.None;
  const previous = select.value;

  select.replaceChildren();
  conditions.forEach(([value, label]) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    select.appendChild(option);
  });

  select.value = conditions.some(([value]) => value === previous)
    ? previous
    : conditions[0][0];
}

function updateReasoningStrategies() {
  const select = $("reasoning-strategy");
  const xaiType = $("xai-type").value;
  const tested = $("tested").value;
  const strategies = availableReasoningStrategies(xaiType, tested);
  const previous = select.value;

  select.replaceChildren();
  strategies.forEach(([value, label]) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    select.appendChild(option);
  });

  select.value = strategies.some(([value]) => value === previous)
    ? previous
    : strategies[0][0];
}

function updateIframe() {
  const ds = datasets[$("dataset").value];
  if (!ds) return;

  const tested = $("tested").value;                                          // 加这行
  const xaiType = $("xai-type").value.toLowerCase();
  const xaiParam = tested === "w/o XAI" ? "none" : xaiType;                 // 改这行
  const instanceId = currentInstanceId();
  $("instance-label").textContent = displayInstanceStatus(instanceId);
  $("status").textContent = displayInstanceStatus(instanceId);

  const params = new URLSearchParams({
    appId: ds.appId,
    modelName: ds.modelName,
    expMethod: ds.expMethod,
    instanceId: instanceId,
    xaiType: xaiParam,
    showPrediction: "0",
  });
  $("xai-frame").src = `${routePrefix()}/UI/iframe.html?${params.toString()}`;
}

function requestPayload() {
  const location = currentInstanceLocation();
  return {
    dataset: $("dataset").value,
    xai_type: $("xai-type").value,
    reasoning_strategy: $("reasoning-strategy").value,
    tested: $("tested").value,
    instance_number: location.instanceNumber,
    instance_id: currentInstanceId(),
    k: Number($("k").value),
    sensitivity: Number($("sensitivity").value),
    retrieval_threshold: Number($("retrieval-threshold").value),
    scaling_factor: Number($("scaling-factor").value),
    decay_param: 0.5,
    n_sessions: 1,
    closest_k: 7,
    session: location.session,
    block_index: location.blockIndex,
  };
}

function confidence(value) {
  return Number.isFinite(Number(value)) ? Math.round(Number(value) * 100) + "%" : "";
}

function choiceWithConfidence(choice, value) {
  if (choice === null || choice === undefined || choice === "") return "";
  return `${choice} — ${confidence(value)}`;
}

function setPredictionValue(id, value, confidenceValue, confidenceId) {
  const el = $(id);
  const displayValue = value === null || value === undefined || value === "" ? "-" : value;
  el.textContent = displayValue;

  const conf = confidence(confidenceValue);
  if (confidenceId) {
    $(confidenceId).textContent = conf ? `Confidence ${conf}` : "";
  }

  if (conf) {
    el.title = `Confidence: ${conf}`;
  } else {
    el.removeAttribute("title");
  }
}

function renderPredictionSummary(rows) {
  if (rows) latestPredictionRows = rows;
  const selectedStrategy = $("reasoning-strategy").value;
  const row = latestPredictionRows.find(item => item.Strategy === selectedStrategy);

  setPredictionValue("simulation-prediction", row?.["CoAX Choice"], row?.["CoAX Confidence"], "simulation-confidence");
  setPredictionValue("human-prediction", row?.["Human Response"], row?.["Human Confidence"], "human-confidence");
  setPredictionValue("actual-prediction", row?.["AI Prediction"]);
}

function renderResults(rows) {
  renderPredictionSummary(rows);
}

function clearPredictionSummary() {
  latestPredictionRows = [];
  renderPredictionSummary();
  $("status").textContent = displayInstanceStatus();
}




async function runSimulation() {
  const button = $("run-button");
  const label = button.querySelector(".run-label");
  const status = $("status");
  button.disabled = true;
  button.classList.add("loading");

  // Animate button text
  const originalLabel = "Run Simulation";
  label.textContent = "Thinking";

  try {
    await new Promise(r => setTimeout(r, 2000));
    const res = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestPayload()),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Simulation failed");
    renderPredictionSummary(data.strategy_predictions);
    status.textContent = displayInstanceStatus();
  } catch (error) {
    status.textContent = error.message;
  } finally {
    label.textContent = originalLabel;
    button.classList.remove("loading");
    button.disabled = false;
  }
}

function bindControls() {
  $("case-number").addEventListener("change", () => {
    applyDemoCase();
    updateIframe();
    clearPredictionSummary();
  });

  $("instance-number").addEventListener("change", () => {
    updateIframe();
    clearPredictionSummary();
  });

  ["dataset", "xai-type", "tested"].forEach((id) => {
    $(id).addEventListener("change", () => {
      if (id === "xai-type") updateTestedOptions();
      if (id === "dataset") updateXaiTypeLabels();
      if (id === "xai-type" || id === "tested") updateReasoningStrategies();
      populateInstances();
      updateIframe();
      clearPredictionSummary();
    });
  });
  ["k", "sensitivity", "retrieval-threshold", "scaling-factor"].forEach((id) => {
    const input = $(id);
    const output = $(`${id}-value`);
    const update = () => {
      output.textContent = input.value;
    };
    input.addEventListener("input", update);
    update();
  });
  $("reasoning-strategy").addEventListener("change", clearPredictionSummary);
  $("run-button").addEventListener("click", runSimulation);
  $("sidebar-toggle")?.addEventListener("click", toggleSidebar);
}

populateDatasets();
populateCases();
applyDemoCase();
bindControls();
updateIframe();

const handle = document.getElementById('resize-handle');
const shell  = document.querySelector('.app-shell');

handle.addEventListener('mousedown', e => {
  if (shell.classList.contains('sidebar-collapsed')) return;
  e.preventDefault();
  handle.classList.add('dragging');
  shell.classList.add('resizing'); 

  const onMove = e => {
    const w = Math.min(Math.max(e.clientX, 272), 480);
    shell.style.setProperty('--panel-width', w + 'px');
  };
  const onUp = () => {
    handle.classList.remove('dragging');
    shell.classList.remove('resizing'); 
    window.removeEventListener('mousemove', onMove);
    window.removeEventListener('mouseup', onUp);
  };
  window.addEventListener('mousemove', onMove);
  window.addEventListener('mouseup', onUp);
});
