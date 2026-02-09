const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const statusEl = document.getElementById("status");
const predictionEl = document.getElementById("prediction");
const confidenceEl = document.getElementById("confidence");
const interpretationEl = document.getElementById("interpretation");
const probabilitiesEl = document.getElementById("probabilities");

const renderProbabilities = (probabilities) => {
  probabilitiesEl.innerHTML = "";
  const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  entries.forEach(([label, value]) => {
    const row = document.createElement("div");
    row.className = "flex items-center justify-between rounded-xl bg-slate-950/60 px-4 py-3";
    row.innerHTML = `
      <span class="text-slate-200">${label}</span>
      <span class="text-emerald-300">${(value * 100).toFixed(2)}%</span>
    `;
    probabilitiesEl.appendChild(row);
  });
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    statusEl.textContent = "Please select an ECG file before submitting.";
    return;
  }

  statusEl.textContent = "Analyzing ECG...";
  predictionEl.textContent = "Processing...";
  confidenceEl.textContent = "--";
  interpretationEl.textContent = "";
  probabilitiesEl.innerHTML = "";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Unable to analyze ECG.");
    }

    const data = await response.json();
    predictionEl.textContent = data.label;
    confidenceEl.textContent = `${(data.confidence * 100).toFixed(2)}%`;
    interpretationEl.textContent = data.interpretation;
    renderProbabilities(data.probabilities);
    statusEl.textContent = "Analysis complete.";
  } catch (error) {
    statusEl.textContent = error.message;
    predictionEl.textContent = "Error";
    confidenceEl.textContent = "--";
    interpretationEl.textContent = "Please verify the ECG file contains numeric samples.";
    probabilitiesEl.innerHTML = "<p>--</p>";
  }
});
