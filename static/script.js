const statusBox = document.getElementById("statusBox");
const sourcePill = document.getElementById("dataSourcePill");
const predictedClassEl = document.getElementById("predictedClass");
const confidenceEl = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");
const accuracyEl = document.getElementById("accuracyValue");
const cvEl = document.getElementById("cvValue");

let analyticsCache = null;

const EEG_COLORS = ["#22d3ee", "#a855f7", "#22c55e", "#f97316", "#3b82f6", "#14b8a6", "#6366f1", "#84cc16"];
const HAS_PLOTLY = typeof window.Plotly !== "undefined";

function setStatus(text, source = "") {
  statusBox.textContent = text;
  sourcePill.textContent = source || "No source";
}

async function postJson(url, body = {}) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!data.ok) throw new Error(data.error || "Request failed");
  return data;
}

function renderEEG(signalPayload) {
  if (!HAS_PLOTLY) return;
  const { times, channels, signal } = signalPayload;
  const traces = channels.map((ch, i) => ({
    x: times,
    y: signal[i].map(v => v * 1e6 + i * 35),
    mode: "lines",
    type: "scattergl",
    name: ch,
    line: { width: 1.6, color: EEG_COLORS[i % EEG_COLORS.length] },
  }));
  Plotly.newPlot("eegChart", traces, {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "#020617",
    font: { color: "#e2e8f0" },
    margin: { l: 44, r: 16, t: 8, b: 36 },
    xaxis: { title: "Time (s)", gridcolor: "#1e293b", zeroline: false },
    yaxis: {
      title: "Amplitude (μV) + offset",
      gridcolor: "#1e293b",
      zeroline: false,
      tickvals: channels.map((_, i) => i * 35),
      ticktext: channels,
    },
    legend: { orientation: "h", y: 1.12, x: 0 },
  }, { responsive: true, displayModeBar: false });
}

function renderSpectrogram(analytics) {
  if (!HAS_PLOTLY) return;
  const z = analytics.spectrogram;
  const x = analytics.freqs;
  const y = [...Array(z.length).keys()].map(i => `Ch ${i + 1}`);
  Plotly.newPlot("spectrogramChart", [{
    x,
    y,
    z,
    type: "heatmap",
    colorscale: "Turbo",
    showscale: true,
  }], {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "#020617",
    font: { color: "#e2e8f0", size: 11 },
    margin: { l: 50, r: 14, t: 8, b: 28 },
    xaxis: {
      title: "Frequency (Hz)",
      gridcolor: "#1e293b",
      range: [2, 40],
    },
    yaxis: { title: "Channels", gridcolor: "#1e293b" },
    shapes: [
      { type: "rect", x0: 8, x1: 12, y0: -0.5, y1: y.length - 0.5, line: { width: 0 }, fillcolor: "rgba(34,211,238,0.14)" },
      { type: "rect", x0: 13, x1: 30, y0: -0.5, y1: y.length - 0.5, line: { width: 0 }, fillcolor: "rgba(59,130,246,0.14)" },
    ],
  }, { responsive: true, displayModeBar: false });
}

function renderBandPower(analytics) {
  if (!HAS_PLOTLY) return;
  const bp = analytics.band_power;
  Plotly.newPlot("bandPowerChart", [{
    x: ["Alpha", "Beta", "Gamma"],
    y: [bp.alpha, bp.beta, bp.gamma],
    type: "bar",
    marker: { color: ["#22d3ee", "#3b82f6", "#a855f7"] },
  }], {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "#020617",
    font: { color: "#e2e8f0", size: 11 },
    margin: { l: 40, r: 10, t: 8, b: 32 },
    xaxis: { gridcolor: "#1e293b" },
    yaxis: { title: "Power", gridcolor: "#1e293b" },
  }, { responsive: true, displayModeBar: false });
}

function renderConfusion(cm = [[0, 0], [0, 0]], labels = ["Left", "Right"]) {
  if (!HAS_PLOTLY) return;
  Plotly.newPlot("confusionChart", [{
    z: cm,
    x: labels,
    y: labels,
    type: "heatmap",
    colorscale: "Blues",
    text: cm.map(r => r.map(v => String(v))),
    texttemplate: "%{text}",
    textfont: { color: "#e2e8f0", size: 15 },
    showscale: true,
  }], {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "#020617",
    font: { color: "#e2e8f0", size: 11 },
    margin: { l: 44, r: 10, t: 8, b: 34 },
    xaxis: { title: "Predicted", gridcolor: "#1e293b" },
    yaxis: { title: "Actual", gridcolor: "#1e293b" },
  }, { responsive: true, displayModeBar: false });
}

function updateMetrics(result) {
  predictedClassEl.textContent = result.predicted_class;
  const confPct = (result.confidence * 100).toFixed(1);
  confidenceEl.textContent = `${confPct}%`;
  confidenceBar.style.width = `${Math.max(0, Math.min(100, Number(confPct)))}%`;
  accuracyEl.textContent = `${(result.accuracy * 100).toFixed(1)}%`;
  cvEl.textContent = `${(result.cv_score * 100).toFixed(1)}%`;
  renderConfusion(result.confusion_matrix, result.class_labels);
}

async function loadPhysioNet() {
  setStatus("Loading PhysioNet data...", "PhysioNet EEGBCI");
  try {
    const data = await postJson("/api/load_physionet", { subject: 1 });
    analyticsCache = data.analytics;
    setStatus(data.status, data.source);
    renderEEG(data.signal);
    renderSpectrogram(data.analytics);
    renderBandPower(data.analytics);
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  }
}

async function generateSynthetic() {
  setStatus("Generating synthetic EEG...", "Synthetic EEG");
  try {
    const data = await postJson("/api/generate_synthetic");
    analyticsCache = data.analytics;
    setStatus(data.status, data.source);
    renderEEG(data.signal);
    renderSpectrogram(data.analytics);
    renderBandPower(data.analytics);
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  }
}

async function runClassification() {
  setStatus("Running classification...", sourcePill.textContent);
  try {
    const data = await postJson("/api/run_classification");
    setStatus(data.status, data.source);
    updateMetrics(data.result);
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  }
}

async function uploadEDF(file) {
  const formData = new FormData();
  formData.append("file", file);
  setStatus("Uploading EDF...", "Uploaded EDF");
  try {
    const res = await fetch("/api/upload_edf", { method: "POST", body: formData });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "Upload failed");
    analyticsCache = data.analytics;
    setStatus(data.status, data.source);
    renderEEG(data.signal);
    renderSpectrogram(data.analytics);
    renderBandPower(data.analytics);
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  }
}

document.getElementById("btnDownloadLoad").addEventListener("click", loadPhysioNet);
document.getElementById("btnLoadS001").addEventListener("click", loadPhysioNet);
document.getElementById("btnSynthetic").addEventListener("click", generateSynthetic);
document.getElementById("btnRunClassSide").addEventListener("click", runClassification);
document.getElementById("btnRunClassMain").addEventListener("click", runClassification);
document.getElementById("btnDemo").addEventListener("click", async () => {
  await generateSynthetic();
  await runClassification();
});
document.getElementById("edfUpload").addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  if (file) uploadEDF(file);
});

if (!HAS_PLOTLY) {
  setStatus("Plotly.js unavailable in this environment. Core controls remain functional.");
} else {
  renderConfusion();
}
if (analyticsCache && HAS_PLOTLY) {
  renderSpectrogram(analyticsCache);
  renderBandPower(analyticsCache);
}
