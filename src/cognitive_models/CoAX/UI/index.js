let urlRef = "";
let url = "";
// const urlRoot = "https://brianlimyl.github.io/transferable-xai/";
const urlRoot = ""; // https://louth-bin.github.io/xaikit-test-ui/
let tutorial = 0;

function updateIframeUrl() {
  const appId = document.querySelector("#input_appId").value;
  const instanceId = document.querySelector("#input_instanceId").value;
  const modelName = document.querySelector("#input_model_name").value;
  const expMethod = document.querySelector("#input_exp_method").value;
  const xaiType = document.querySelector("#input_xaiType").value;
  const userSimulation = document.querySelector("#input_userSimulation").checked
    ? 1
    : 0;
  const showPrediction = document.querySelector("#input_showPrediction").checked
    ? 1
    : 0;
  const showTutorial = document.querySelector("#input_showTutorial").checked
    ? 1
    : 0;
  const userPrediction = document.querySelector("#input_userPrediction").value;

  const iframe = document.querySelector("#interactiveIframe");
  const showGroundTruth = document.querySelector("#input_showGroundTruth").checked ? 1 : 0;

  const focusOnImportant = document.querySelector('#input_focusOnImportant').checked ? 1 : 0;

  // // Sensitivity array
  // const sensitivityArray = [1, 0.5, 0.2, 0.3, 1];
  // const encodedSensitivity = encodeURIComponent(JSON.stringify(sensitivityArray));

  url =
    "iframe.html?appId=" +
    appId +
    "&instanceId=" +
    instanceId +
    "&modelName=" +
    modelName +
    "&expMethod=" +
    expMethod +
    "&xaiType=" +
    xaiType +
    "&userSimulation=" +
    userSimulation +
    "&showPrediction=" +
    showPrediction +
    "&showTutorial="+
    showTutorial+
    "&userPrediction=" +
    userPrediction+
    "&showGroundTruth=" +
    showGroundTruth +
    "&focusOnImportant="+
    focusOnImportant 
    // "&sensitivity=" + 
    // encodedSensitivity;
    ;

  iframe.src = url;
  iframe.height = 900;

  document.querySelector("#text_url").href = url;
  document.querySelector("#text_url").innerHTML = url;
}

function updateinstanceId() {
  const input_instanceId = document.querySelector("#input_instanceId");
  let instanceId = input_instanceId.value;
  instanceId = instanceId < 0 ? 0 : instanceId;
  instanceId = instanceId > 10000 ? 10000 : instanceId;
  input_instanceId.value = instanceId;
  updateIframeUrl();
}
function updateinstanceIdTyping(e) {
  const input_instanceId = document.querySelector("#input_instanceId");
  let instanceId = input_instanceId.value;
  if (e.key == "Enter") {
    return;
  } else if (isNaN(parseInt(e.key)) || instanceId.toString().length >= 5) {
    // invalid typing to revert to previous number
    event.preventDefault();
    input_instanceId.value = instanceId;
  }
}

function copyIframeHtmlRef() {
  let html = document.querySelector("#interactiveIframeRef").outerHTML;
  html = html.replace('src="', 'src="' + urlRoot);
  html = html.replaceAll("&amp;", "&");
  html = html.replaceAll('id="interactiveIframeRef" ', "");
  navigator.clipboard.writeText(html);
}
function copyUrlRef() {
  navigator.clipboard.writeText(urlRoot + urlRef);
}
function copyIframeHtml() {
  let html = document.querySelector("#interactiveIframe").outerHTML;
  html = html.replace('src="', 'src="' + urlRoot);
  html = html.replaceAll("&amp;", "&");
  html = html.replaceAll('id="interactiveIframe" ', "");
  navigator.clipboard.writeText(html);
}
function copyUrl() {
  navigator.clipboard.writeText(urlRoot + url);
}

// receive postMessage from iframe
window.addEventListener("message", function (event) {
  console.log(event.data);
});

document.querySelector("#input_appId").onchange = updateIframeUrl;
document.querySelector("#input_model_name").onchange = updateIframeUrl;
document.querySelector("#input_exp_method").onchange = updateIframeUrl;
document.querySelector("#input_instanceId").onchange = updateinstanceId;
document.querySelector("#input_instanceId").onkeypress = updateinstanceIdTyping;
document.querySelector("#input_xaiType").onchange = updateIframeUrl;
document.querySelector("#input_showPrediction").onchange = updateIframeUrl;
document.querySelector("#input_showTutorial").onchange = updateIframeUrl;
document.querySelector("#input_userPrediction").onchange = updateIframeUrl;
document.querySelector("#input_userSimulation").onchange = updateIframeUrl;
document.querySelector("#input_showGroundTruth").onchange = updateIframeUrl;
document.querySelector("#input_focusOnImportant").onchange = updateIframeUrl;
updateIframeUrl();
