// functions to round numbers

const thousands = (n, precision) =>
  Number(n.toPrecision(precision))
    .toString()
    .replace(/\B(?<!\.\d*)(?=(\d{3})+(?!\d))/g, ","); // to show comma separation in numbers

const round = (n, roundNearest, toFixed, snExp) => {
  n =
    snExp != 0 && snExp !== undefined
      ? new Decimal(n).mul(Decimal.pow(10, -snExp))
      : n;
  n = roundNearest > 0 ? Math.round(n / roundNearest) * roundNearest : n;
  n =
    snExp != 0 && snExp !== undefined
      ? new Decimal(n).div(Decimal.pow(10, -snExp))
      : n;
  n = toFixed !== undefined && toFixed > -1 ? Number(n.toFixed(toFixed)) : n;
  return n;
};

const roundToSignificantFigures = (num, significantFigures) => {
  if (num === 0) return 0; // Handle zero case
  const multiplier = Math.pow(
    10,
    significantFigures - Math.floor(Math.log10(Math.abs(num))) - 1
  );
  return Math.round(num * multiplier) / multiplier;
};

/**
 * Read URL key-values to get condition and trial
 */
const url = window.location.search; // e.g.: http://localhost:9000/transferable-xai-ui.html?appId=housingprice-kingcounty&instanceId=1&isRef=1&xaiType=new&showSimilarity=1&showPrediction=1&tutorial=1
const urlParams = new URLSearchParams(url);

const appId = urlParams.get("appId");
const modelName = urlParams.get("modelName");
const isSim2Real = appId === "adult_sim2real" || modelName === "synthetic_ai";
const expMethod = urlParams.get("expMethod") || (isSim2Real ? "lime" : null);
const expProperty = urlParams.get("expProperty");
const instanceId = Number(urlParams.get("instanceId"));

if (isSim2Real) {
  document.body.classList.add("sim2real-iframe");
}

const xaiType =
  urlParams.get("xaiType") === "none" ? null : urlParams.get("xaiType");
// const tutorial = Number(urlParams.get("tutorial"));
const userPredictionOption = urlParams.get("userPrediction") || "none";
const showPrediction =
  Number(urlParams.get("showPrediction")) === 1 ? true : false;
const userSimulation =
  Number(urlParams.get("userSimulation")) === 1 ? true : false;
const showTutorial = Number(urlParams.get("showTutorial")) === 1 ? true : false;
const showGroundTruth = Number(urlParams.get("showGroundTruth")) === 1 ? true : false;
const focusOnImportant = Number(urlParams.get("focusOnImportant")) === 1;

if (userSimulation) {
  var userValues = [
    { attribute: "v0_i", importance: 0 },
    { attribute: "v1_i", importance: 0 },
    { attribute: "v2_i", importance: 0 },
    { attribute: "v3_i", importance: 0 },
    { attribute: "v4_i", importance: 0 },
    { attribute: "v5_i", importance: 0 },
  ];
}

// Parsing of CSV files, then dispatching to the main renderUI function

let metadata_file = "xai_methods/metadata.csv";
let values_file = "xai_methods/values.csv";

// Mapping for explanation files
const explanationFiles = {
  importance: "xai_methods/importance.csv",
  attribution: "xai_methods/attribution.csv",
  weights: "xai_methods/weights_fake.csv",
  counterfactual: "xai_methods/counterfactuals_fake.csv",
  similar: "xai_methods/counterfactuals_fake.csv",
};

// Default to importance.csv if xaiType is undefined or invalid
let explanation_file = explanationFiles[xaiType] || "xai_methods/none.csv";

// Function to handle completed parsing for all files
function handleParsingComplete(atts, values, exp_result) {
  renderUI(atts, values, exp_result);
}

// Error handling function
function handleError(error) {
  console.error("Error parsing file:", error);
}

// Parse files in parallel
let metadataPromise = new Promise((resolve, reject) => {
  Papa.parse(metadata_file, {
    download: true,
    header: true,
    dynamicTyping: true,
    complete: (res) => resolve(extractAttributeNames(res.data)),
    error: reject,
  });
});

let valuesPromise = new Promise((resolve, reject) => {
  Papa.parse(values_file, {
    download: true,
    header: true,
    dynamicTyping: true,
    complete: (res) => resolve(res.data),
    error: reject,
  });
});

let explanationPromise = new Promise((resolve, reject) => {
  Papa.parse(explanation_file, {
    download: true,
    header: true,
    dynamicTyping: true,
    complete: (res) => resolve(res.data),
    error: reject,
  });
});

// Execute all parsing simultaneously and render UI when all are complete
Promise.all([metadataPromise, valuesPromise, explanationPromise])
  .then(([atts, values, exp_result]) => {
    handleParsingComplete(atts, values, exp_result);
  })
  .catch(handleError);

let chart;

// Main renderUI function that is responsible for creating the elements in the iframe
function renderUI(atts, data, explanation) {
  // remove explanation header rows if not showing explanation
  if (!xaiType) {
    let elements = document.getElementsByClassName("explanation-col");
    for (let i = 0; i < elements.length; i++) {
      elements[i].style.display = "none";
    }
  }

  // extract values from csv
  let {
    rows,
    prediction,
    iMax,
    predictionNum,
    userPrediction,
    counterfactualPrediction,
    yLabels,
    groundTruth
  } = constructRowValues(atts, data, explanation);

  // post a message to the parent window on the prediction
  window.parent.postMessage("Model prediction: " + prediction, "*");

  // extract the relevant table depending on xai type, display it, and extract table body
  if (xaiType === "weights") {
    var table = document.getElementById("feature-weights-table");
    table.style.display = "table";
    var tableBody = document.getElementById("feature-weights-tbody");
  } else if (xaiType === "importance") {
    var table = document.getElementById("feature-importance-div");
    table.style.display = "flex";
    var tableBody = document.getElementById("feature-importance-tbody");
  } else if (xaiType === "attribution") {
    var table = document.getElementById("feature-attribution-div");
    table.style.display = "flex";
    var tableBody = document.getElementById("feature-attribution-tbody");
  } else if (xaiType === "counterfactual") {
    var counterfactualDiv = document.getElementById("counterfactual-div");
    counterfactualDiv.style.display = "block";
    var table = document.getElementById("current-case-counterfactual-table");
    var tableBody = document.getElementById(
      "current-case-counterfactual-tbody"
    );
    var counterfactualTableBody = document.getElementById(
      "counterfactual-tbody"
    );
  } else if (xaiType === "similar") {
    var table = document.getElementById("similar-div");
    table.style.display = "block";
    var tableBody = document.getElementById("current-case-similar-tbody");
    var similarTableBody = document.getElementById("similar-tbody");
    } else if (!xaiType) {
      var table = document.getElementById("feature-importance-div");
      table.style.display = "flex";
      var tableBody = document.getElementById("feature-importance-tbody");
      document.querySelectorAll(".explanation-col").forEach(el => el.style.display = "none");
    }

  // if the xai type is similar or counterfactual, extract the current case and counterfactual values
  if (xaiType === "similar" || xaiType === "counterfactual") {
    var currentCaseValues = rows.map((row) =>
      // if the attribute is numerical, set the value to the numerical value, else set it to the categorical value
      row["category_index"] === undefined
        ? { numerical: roundToSignificantFigures(Number(row["value"]), 2) }
        : { categorical: row["value"] }
    );
    var counterfactualValues = rows.map((row) =>
      row["category_index"] === undefined
        ? { numerical: roundToSignificantFigures(Number(row["importance"]), 2) }
        : { categorical: row["options"][row["importance"]] }
    );
  }

  // number of attributes determines number of rows in the table
  const nAtts = rows.length;

  // create basic table rows (applies to all xai types)
  for (let i = 0; i < nAtts; i++) {
    const row = rows[i];
    let tr = createTableDataRow(i, row, nAtts);
    tableBody.appendChild(tr);
  }

  // create explanation elements depending on xai type
  let finalSum; // only applicable for feature weights

  if (xaiType === "weights") {
    // calculate sums for feature weights
    finalSum = rows.reduce((sum, row) => {
      if (row["attribute"] !== "Others") {
        if (row["category_index"] === undefined) {
          return sum + row["value"] * row["importance"];
        } else {
          return sum + row["category_index"] * row["importance"];
        }
      } else {
        return sum + row["importance"];
      }
    }, 0);
  } else if (xaiType === "importance") {
    let canvas = document.createElement("canvas");
    canvas.id = `feature-${xaiType}-canvas`;
    const chartContainer = document.getElementById(`feature-${xaiType}-chart`);
    chartContainer.appendChild(canvas);
    chart = userSimulation
      ? createFeatureImportanceChart(userValues, 1)
      : createFeatureImportanceChart(rows, iMax);
  } else if (xaiType === "attribution") {
    let canvas = document.createElement("canvas");
    canvas.id = `feature-${xaiType}-canvas`;
    const chartContainer = document.getElementById(`feature-${xaiType}-chart`);
    chartContainer.appendChild(canvas);
    chart = userSimulation
      ? createFeatureAttributionChart(userValues, 1)
      : createFeatureAttributionChart(rows, iMax);
  } else if (xaiType === "counterfactual") {
    createCaseComparisonDifferences(
      rows,
      nAtts,
      counterfactualTableBody,
      "counterfactual-tbody",
      currentCaseValues,
      counterfactualValues
    );
  } else if (xaiType === "similar") {
    createCaseComparisonDifferences(
      rows,
      nAtts,
      similarTableBody,
      "similar-tbody",
      currentCaseValues,
      counterfactualValues
    );
  }

  // add the "type 1" and "type 2" labels
  if (xaiType === "attribution"){
    let labelRow = document.createElement("tr");
    labelRow.className = "label_row";

    // Create cells with different configurations
    for (let i = 0; i < 6; i++) {
        let cell = document.createElement("td");

        // Place text between the third and fourth, and fifth and sixth cells
        if (i === 3 || i === 4) { // Adjust index for colspan just before the required cells
            //cell.colSpan = 2; // Span across two columns
            cell.style.padding = "0px";
            if (i === 3) {
                cell.innerText = yLabels[0]; // Text from yLabels[0] between third and fourth cells
                cell.style.textAlign = "left";
                cell.style.fontSize = "0.9em";
                cell.style.color = "rgba(234, 51, 53)"; // Set text color for the first label
                // cell.style.transform = "translateX(30px)";
            } else {
                cell.innerText = yLabels[1]; // Text from yLabels[1] between fifth and sixth cells
                cell.style.textAlign = "right";
                cell.style.fontSize = "0.9em";
                cell.style.color = "rgba(60, 136, 232)"; // Set text color for the second label
            }
            //i++; // Skip the next cell creation because this cell spans two columns
        }
        
        labelRow.appendChild(cell);
    }

    tableBody.appendChild(labelRow);
  }




  // ====================================================================


  if (showTutorial) {
    createTutorialElements(table);
  }

  // function to handle showing AI prediction
  if (showPrediction) {
    let firstRows = document.querySelectorAll(`.row_0`);
    if (firstRows.length === 1) {
      // if there is only one row -> only one table -> set up as usual
      createCurrentCasePredictionCols(
        finalSum,
        nAtts,
        prediction,
        predictionNum,
        userPrediction,
        groundTruth
      );
    }
  }

  // function to handle showing user prediction
  if (userPredictionOption === "none") {
    let elements = document.getElementsByClassName("user-prediction");
    for (let i = 0; i < elements.length; i++) {
      elements[i].style.display = "none";
    }
  }

  // Post the prediction number for qualtrics
  window.parent.postMessage("predictionNum:" + predictionNum, '*');
}

// Functions to extract data from CSV and make it a readable object for UI

const metadataFallbacks = {
  forest_cover: {
    appId: "forest_cover",
    a0: "Elevation",
    v0_min: 2050.0,
    v0_max: 2688.0,
    a1: "Angle",
    a2: "Dist to Water",
    a3: "Dist to Road",
    a4: "Hillshade",
    y: "Cover_Type",
    y0: "Spruce",
    y1: "Lodgepole",
    v1_max: 344.0,
    v1_min: 17.0,
    v2_max: 474.0,
    v2_min: 0.0,
    v3_max: 2155.0,
    v3_min: 162.0,
    v4_max: 251.0,
    v4_min: 126.0,
  },
  adult: {
    appId: "adult",
    a0: "Age",
    v0_min: 20.0,
    v0_max: 63.0,
    a1: "Years of Education",
    a2: "Married",
    v2_options: 4.0,
    v2_0: "no",
    v2_1: "separated",
    v2_2: "widow",
    v2_3: "yes",
    a3: "Sex",
    v3_options: 2.0,
    v3_0: "F",
    v3_1: "M",
    a4: "Capital Gain",
    y: "Income",
    y0: "Low Income",
    y1: "High Income",
    v1_max: 14.0,
    v1_min: 5.0,
    v4_max: 5013.0,
    v4_min: 0.0,
  },
};

function sameValue(left, right) {
  return String(left ?? "").trim() === String(right ?? "").trim();
}

function extractAttributeNames(data) {
  // extract the object where appId === appId
  const atts = data.find((d) => sameValue(d.appId, appId)) || metadataFallbacks[appId];
  if (!atts) {
    throw new Error(`No metadata row found for appId=${appId}`);
  }

  // for v1_max, v1_min ... make into numbers
  for (let key in atts) {
    const parsedValue = parseFloat(atts[key]);
    atts[key] = isNaN(parsedValue) ? atts[key] : parsedValue; // Only assign parsed number if it's a valid number
  }
  return atts;
}

function extractDataInstance(data) {
  const instance = data.find(
    (d) =>
      sameValue(d.appId, appId) &&
      Number(d.instanceId) === instanceId &&
      (!d.modelName || sameValue(d.modelName, modelName)) &&
      (!d.expMethod || sameValue(d.expMethod, expMethod)) &&
      (!("expProperty" in d) ||
        (expProperty
          ? sameValue(d.expProperty, expProperty)
          : !d.expProperty))
  );

  if (!instance) {
    throw new Error(
      `No data row found for appId=${appId}, modelName=${modelName}, expMethod=${expMethod}, expProperty=${expProperty}, instanceId=${instanceId}`
    );
  }

  for (let key in instance) {
    const parsedValue = parseFloat(instance[key]);
    instance[key] = isNaN(parsedValue) ? instance[key] : parsedValue; // Only assign parsed number if it's a valid number
  }
  return instance;
}


function constructRowValues(atts, data, explanation) {
  let datapoint = extractDataInstance(data);
  let explanationPoint = extractDataInstance(explanation);

  let rows = [];

  // Dynamically determine the number of attributes based on keys in `atts`
  const attributeKeys = Object.keys(atts).filter((key) => /^a\d+$/.test(key));

  // Iterate over each attribute dynamically
  attributeKeys.forEach((attrKey) => {
    let i = parseInt(attrKey.substring(1)); // Extract the index from the key (e.g., "a0" -> 0)
    let row = {};
    row["attribute"] = atts[`a${i}`];

    // Skip the attribute if it exists but has no valid data
    if (!row["attribute"] || row["attribute"].trim() === "") return;


    // need to check not only if the v{i}_min exists but is not null also
    if (`v${i}_min` in atts && atts[`v${i}_min`] !== null) {
      // Handle numeric attributes
      row["min"] =
        atts[`v${i}_min`] !== undefined && atts[`v${i}_min`] !== ""
          ? round(atts[`v${i}_min`], 0.001)
          : null;
      row["max"] =
        atts[`v${i}_max`] !== undefined && atts[`v${i}_max`] !== ""
          ? round(atts[`v${i}_max`], 0.001)
          : null;
      row["value"] = datapoint[`v${i}`] ?? null;
    } else {
      // Treat as categorical if min and max are not defined or do not exist
      row["min"] = null;
      row["max"] = null;

      let categoryIndex = datapoint[`v${i}`];
      row["category_index"] = categoryIndex ?? null;
      row["value"] =
        categoryIndex !== null && `v${i}_${categoryIndex}` in atts
          ? atts[`v${i}_${categoryIndex}`]
          : null;
      row["num_options"] = atts[`v${i}_options`] ?? null;

      // Collect categorical options dynamically
      let options = [];
      const matchingKeys = Object.keys(atts)
        .filter((key) => new RegExp(`^v${i}_\\d+$`).test(key) && atts[key] !== null)
        .sort((a, b) => parseInt(a.split("_")[1], 10) - parseInt(b.split("_")[1], 10));

      matchingKeys.forEach((key) => {
        options.push(atts[key]);
      });

      row["options"] = options.length > 0 ? options : null;
    }
    
    // Ensure explanationPoint is defined before accessing its properties
    if (explanationPoint && typeof explanationPoint === "object") {
      // Safely retrieve the importance value
      const importanceValue = (`a${i}_i` in explanationPoint) ? explanationPoint[`a${i}_i`] : null;
      const maxImportance = ('i_max' in explanationPoint) ? explanationPoint["i_max"] : null;

      // Calculate importance only if both values are available
      row["importance"] =
        importanceValue !== null && maxImportance !== null
          ? (importanceValue / maxImportance) * 100
          : null;
    } else {
      // Fallback if explanationPoint is undefined or not an object
      row["importance"] = null;
    }


    rows.push(row);
  });

  // Add a last column for "Others," reflected in "a0"
  if (xaiType && xaiType !== "counterfactual" && xaiType !== "similar") {
    let row = {
      attribute: "Others",
      value: null,
      min: null,
      max: null,
      importance: xaiType === "importance"
        ? 0.0001
        : (explanationPoint["intercept"] && explanationPoint["i_max"])
            ? (explanationPoint["intercept"] / explanationPoint["i_max"]) * 100
            : null,
    };
    rows.push(row);
  }

  const prediction = explanationPoint["pred"] === 1 ? atts["y1"] : atts["y0"];
  const counterfactualPrediction =
    explanationPoint["pred"] === 1 ? atts["y0"] : atts["y1"];
  let userPrediction = null;
  if (userPredictionOption !== "none") {
    userPrediction = userPredictionOption == 1 ? atts["y1"] : atts["y0"];
  }

  // Store the label names for the prediction
  const yLabels = [atts["y0"], atts["y1"]];

  // Map the ground truth to text label, if it exists
  let groundTruthNum = datapoint["y"];
  let groundTruth = null;

  if (groundTruthNum !== null && groundTruthNum !== undefined) {
    groundTruth = groundTruthNum === 1 ? atts["y1"] : atts["y0"];
  }

  return {
    rows,
    prediction,
    iMax: explanationPoint["i_max"],
    predictionNum: explanationPoint["pred"],
    userPrediction: userPrediction,
    counterfactualPrediction: counterfactualPrediction,
    yLabels,
    groundTruth: groundTruth,
  };
}


// Functions to create elements for each xAI type

function createFeatureWeightsExplanation(i, row) {
  let x = document.createElement("td");
  x.id = `x_${i}`;
  x.className = "symbol explanation-col";
  x.innerHTML = "&times;";

  let factor = document.createElement("td");
  factor.className = "factor explanation-col";
  factor.id = `v${i}_i`;
  factor.innerText = thousands(round(row["importance"] / 1000, 1));

  let equal = document.createElement("td");
  equal.id = `equal_${i}`;
  equal.className = "symbol explanation-col";
  equal.innerHTML = "=";

  let sum = document.createElement("td");
  sum.id = `sum_${i}`;
  sum.className = "partial explanation-col";
  if (row["attribute"] !== "Others") {
    if (row["category_index"] === undefined) {
      sum.innerText = thousands(
        round((row["value"] * row["importance"]) / 1000, 1)
      );
    } else {
      sum.innerText = thousands(
        round((row["category_index"] * row["importance"]) / 1000, 1)
      );
    }
  } else {
    sum.innerText = thousands(round(row["importance"] / 1000, 1));
  }

  return { x, factor, equal, sum };
}

function createFeatureImportanceChart(rows, max_importance) {
  let ctx = document
    .getElementById("feature-importance-canvas")
    .getContext("2d");
  let labels = rows.map((row) => row["attribute"]);
  let data = rows.map((row) => row["importance"]);
  // let color = "rgba(60, 136, 232)";
  let color = "rgba(34, 139, 34, 1)"; // Darker green color


  chart = new Chart(ctx, {
    id: "chart",
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Feature Importance",
          data: data,
          backgroundColor: color,
          barThickness: 21,
        },
      ],
    },
    options: {
      indexAxis: "y", // horizontal bar chart
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          display: false,
        },
        x: {
          display: false,
          max: 100,
        },
      },
      plugins: {
        tooltip: {
          enabled: false, // Disable tooltips
        },
        legend: {
          display: false,
        },
      },
      animation: {
        duration: 2000, // Set animation duration in milliseconds
      },
    },
  });

  return chart;
}



function createFeatureAttributionChart(rows, max_importance) {
  let ctx = document
    .getElementById("feature-attribution-canvas")
    .getContext("2d");
  let labels = rows.map((row) => row["attribute"]);
  let data = rows.map((row) => row["importance"]);
  let colors = data.map((d) =>
    d >= 0 ? "rgba(60, 136, 232)" : "rgba(234, 51, 53)"
  );

  let chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Feature Attribution",
          data: data,
          backgroundColor: colors,
          barThickness: 21,
        },
      ],
    },
    options: {
      indexAxis: "y", // horizontal bar chart
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          display: false,
        },
        x: {
          display: false,
          min: -100,
          max: 100,
        },
      },
      plugins: {
        tooltip: {
          enabled: false, // Disable tooltips
        },
        legend: {
          display: false,
        },
      },
      animation: {
        duration: 2000, // Set animation duration in milliseconds
      },      
    },
  });

  return chart;
}


// function for creating table row with attribute values
function createTableDataRow(i, row, length) {
  let newRow = document.createElement("tr");
  newRow.className = `row_${i}`;

  let att = document.createElement("td");
  att.id = `a${i}`;
  att.className = "attribute";
  att.innerText = row["attribute"];

  // special styling for "Others" row
  if (row["attribute"] === "Others") {
    att.classList.add("tooltip", "others");
    att.title = "Other attributes that contributed to the prediction";

    // create a tutorial span for others
    let tutorialSpan = document.createElement("span");
    tutorialSpan.className = "tutorial";
    tutorialSpan.textContent = "6";
    att.appendChild(tutorialSpan);
    if (!showTutorial) {
      tutorialSpan.style.visibility = "hidden";
    }
    // position it to the left of the attribute name
    tutorialSpan.style.position = "absolute";
    tutorialSpan.style.left = "-20px";
  }
  newRow.appendChild(att);


  let value = document.createElement("td");
  value.id = `v${i}`;
  value.className = "value";
  if (userSimulation) {
    value.style.width = "90px";
  }
  if (typeof row["value"] === "number") {
    value.innerText = thousands(roundToSignificantFigures(row["value"], 2));
  } else {
    value.innerText = row["value"];
  }
  value.style.textAlign = "center";

  newRow.appendChild(value);


  let dataCell = document.createElement("td");

  // if min and max are defined, create a meter element
  if (row["min"] != null && row["max"] != null) {
    dataCell.className = "meter-container";

    let meter = document.createElement("meter");
    meter.min = row["min"];
    meter.max = row["max"];
    meter.value = row["value"];
    meter.title = `min: ${thousands(
      round(row["min"], 0.0001, true)
    )}, max: ${thousands(round(row["max"], 0.0001, true))}`;
    meter.className = "meter";
    dataCell.appendChild(meter);
  } else if (row["attribute"] != "Others") {
    // const icons = ["fa-circle", "fa-square", "fa-heart", "fa-star"];
    dataCell.className = "icons-container";
    for (let j = 0; j < row["num_options"]; j++) {
      let icon = document.createElement("i");

      if (j === parseInt(row["category_index"])) {
        icon.className = "fas " + "fa-circle";
        icon.style.color = "black";
      } else {
        icon.className = "far " + "fa-circle";
        icon.style.color = "#aaa";
      }

      icon.title = row["options"][j];
      icon.style.cursor = "pointer";
      icon.style.margin = "0 2px";
      icon.style.fontSize = "12px";

      dataCell.appendChild(icon);
    }
  }
  newRow.appendChild(dataCell);


  if (xaiType === "weights") {
    // if weights, create the factor/partial price rows
    if (userSimulation) {
      var { x, factor, equal, sum } = createUserSimulationForFeatureWeights(
        i,
        row
      );
    } else {
      var { x, factor, equal, sum } = createFeatureWeightsExplanation(i, row);
    }

    // border styling for the first row
    if (i === 0) {
      factor.style.borderTop = "2px solid black";
    }
    // border and opacity styling for the last row
    else if (i === length - 1) {
      factor.style.opacity = 0.5;
      sum.style.opacity = 0.5;
      factor.style.borderBottom = "2px solid black";
      factor.style.backgroundClip = "padding-box";
    }

    newRow.appendChild(x);
    newRow.appendChild(factor);
    newRow.appendChild(equal);
    newRow.appendChild(sum);
  } else if (xaiType === "importance" || xaiType === "attribution") {
    if (i === 0) {
      // create the container for the chart if it's the first row
      let chartContainer = document.createElement("td");
      chartContainer.id = `feature-${xaiType}-chart`;
      chartContainer.colSpan = 2;
      chartContainer.rowSpan = length;
      chartContainer.classList.add('dont-grey');
      newRow.appendChild(chartContainer);
    }


    if (userSimulation) {
      // if user simulation, make all the importance values fields editable
      let importance = document.createElement("td");
      importance.style.borderRight = "2px solid black";

      let importanceInput = document.createElement("input");
      importanceInput.type = "number";

      if (xaiType === "importance") {
        importanceInput.min = 0;
      } else if (xaiType === "attribution") {
        importanceInput.min = -1;
      }

      importanceInput.max = 1;
      importanceInput.step = 0.1;

      importanceInput.className = "user-simulation-input";
      importanceInput.id = `v${i}_i`;
      importanceInput.value = 0.0;
      importance.appendChild(importanceInput);
      if (row["attribute"] === "Others") {
        importance.classList.add("others");
        importance.style.borderBottom = "2px solid black";
      }
      newRow.appendChild(importance);
    }
  }

  return newRow;
}

// creating prediction columns
function createCurrentCasePredictionCols(
  finalSum,
  nAtts,
  prediction,
  predictionNum,
  userPrediction,
  groundTruth
) {
  let firstRow = document.querySelector(".row_0");

  let symbols = document.createElement("td");
  symbols.class = "prediction-col";
  symbols.rowSpan = nAtts;
  symbols.innerText = "} →";
  symbols.style.fontSize = "1.2em";
  symbols.classList.add('dont-grey');

  let predictionContainer = document.createElement("td");
  predictionContainer.rowSpan = nAtts;

  firstRow.appendChild(symbols);
  firstRow.appendChild(predictionContainer);

  let predictionTable = document.createElement("table");
  predictionTable.style.margin = 0;

  let tableHead = document.createElement("thead");
  let headerRow = document.createElement("tr");

  if (xaiType === "weights") {
    let header = document.createElement("th");
    header.innerText = "Final sum";
    header.colSpan = 3;
    headerRow.appendChild(header);

    let symbol = document.createElement("th");
    symbol.style.fontSize = "1.2em";
    headerRow.appendChild(symbol);
  }

  let predictionHeader = document.createElement("th");
  predictionHeader.innerText = "AI prediction";
  predictionHeader.title = "The AI's prediction for this instance";
  predictionHeader.classList.add("tooltip");
  headerRow.appendChild(predictionHeader);

  let symbolHeader = document.createElement("th");
  headerRow.appendChild(symbolHeader);

  let userPredictionHeader = document.createElement("th");
  userPredictionHeader.innerText = "Your selection";
  userPredictionHeader.title = "Your selection for the AI's prediction";
  userPredictionHeader.classList.add("tooltip", "user-prediction");
  headerRow.appendChild(userPredictionHeader);

  tableHead.appendChild(headerRow);
  predictionTable.appendChild(tableHead);

  let tableBody = document.createElement("tbody");
  let dataRow = document.createElement("tr");

  if (xaiType === "weights") {
    let sumData = document.createElement("td");
    sumData.id = "final-sum";
    sumData.classList.add("prediction");
    if (userSimulation) {
      sumData.innerText = 0;
    } else {
      sumData.innerText = thousands(round(finalSum, 1));
    }
    sumData.colSpan = 3;
    dataRow.appendChild(sumData);

    let sumSymbol = document.createElement("td");
    sumSymbol.innerText = "=";
    sumSymbol.style.fontSize = "1.2em";
    dataRow.appendChild(sumSymbol);
  }

  let predictionData = document.createElement("td");
  predictionData.classList.add("prediction", "dont-grey");
  predictionData.innerText = prediction;
  predictionData.style.color = predictionNum === 1 ? "rgba(60, 136, 232)" : "rgba(234, 51, 53)";;
  dataRow.appendChild(predictionData);

  let symbolData = document.createElement("td");
  symbolData.innerText = prediction == userPrediction ? "=" : "≠";
  symbolData.classList.add("user-prediction");
  symbolData.style.fontSize = "1.2em";
  dataRow.appendChild(symbolData);

  if (showGroundTruth){
    let groundTruthData = document.createElement("td"); // New data cell for ground truth
    groundTruthData.classList.add("prediction");
    groundTruthData.innerText = groundTruth !== null ? groundTruth : "N/A"; // Display "N/A" if ground truth is null
    dataRow.appendChild(groundTruthData);
  }

  let userPredictionData = document.createElement("td");
  userPredictionData.classList.add("prediction", "user-prediction");
  userPredictionData.innerText = userPrediction;
  userPredictionData.style.backgroundColor =
    prediction == userPrediction
      ? "rgba(0, 255, 0, 0.2)"
      : "rgba(255, 0, 0, 0.2)";
  dataRow.appendChild(userPredictionData);

  tableBody.appendChild(dataRow);

  let tutorialRow = document.createElement("tr");
  let tutorialData = document.createElement("td");
  tutorialData.style.textAlign = "center";
  let tutorialSpan = document.createElement("span");
  tutorialSpan.className = "tutorial";
  tutorialSpan.textContent = "4";
  tutorialRow.appendChild(tutorialData);
  tutorialData.appendChild(tutorialSpan);
  tableBody.appendChild(tutorialRow);
  if (!showTutorial) {
    tutorialSpan.style.visibility = "hidden";
  }

  predictionTable.appendChild(tableBody);
  predictionContainer.appendChild(predictionTable);

  predictionContainer.classList.add('dont-grey')
}

// add tutorial elements
function createTutorialElements(table) {
  // add tutorial text above the table
  const header = table.querySelector("thead"); // Assuming there is a single thead
  let newRow = document.createElement("tr");
  const cellsText = ["1", "2", "3"]; // Tutorial numbers only. Make this dependent on the importance vs attribution
  cellsText.forEach((text) => {
    const th = document.createElement("th");
    if (text) {
      // Only create tutorial circles if there is a number
      const span = document.createElement("span");
      span.className = "tutorial";
      span.textContent = text;
      th.appendChild(span);
      if (text === "1") {
        span.parentElement.style.textAlign = "right";
        span.parentElement.style.paddingRight = "30px";
      }
    }
    newRow.appendChild(th);
  });

  let expCellsText;
  if (xaiType === "weights") {
    expCellsText = ["", "5", "", "6"];
  } else if (
    xaiType === "counterfactual" ||
    xaiType === "similar" ||
    !xaiType
  ) {
    expCellsText = ["", "", ""];
  } else if (xaiType === "attribution" || xaiType === "importance") {
    expCellsText = ["", "5", ""];
  }

  if (xaiType) {
    expCellsText.forEach((text) => {
      const th = document.createElement("th");
      if (text) {
        // Only create tutorial circles if there is a number
        const span = document.createElement("span");
        span.className = "tutorial"; // Apply tutorial class
        span.textContent = text;
        th.appendChild(span);
      }
      newRow.appendChild(th);
    });
  }
  header.insertBefore(newRow, header.firstChild);

}
