// API base URL — loaded from config.js (generated from .env)
const API_BASE = window.__CONFIG__?.BE_BASE_URL || "http://localhost:8000";

// --- Tab switching ---
document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach((s) => s.classList.remove("active"));

        btn.classList.add("active");
        document.getElementById(btn.dataset.tab).classList.add("active");
    });
});

// --- Prediction form ---
const form = document.getElementById("predict-form");
const predictBtn = document.getElementById("predict-btn");
const resultsDiv = document.getElementById("results");
const ensembleDiv = document.getElementById("ensemble-result");
const modelResultsDiv = document.getElementById("model-results");
const errorDiv = document.getElementById("error-msg");
const feedbackSection = document.getElementById("feedback-section");
const feedbackForm = document.getElementById("feedback-form");
const feedbackBtn = document.getElementById("feedback-btn");

// Store the last prediction context so we can attach it to feedback
let lastPredictionContext = null;

// Chart instances (so we can destroy them before re-drawing)
let modelComparisonChart = null;
let shapChart = null;

function formatModelName(name) {
    return name
        .split("_")
        .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
        .join(" ");
}

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    // Hide previous results / errors
    resultsDiv.classList.add("hidden");
    errorDiv.classList.add("hidden");

    // Gather form data
    const checkboxFields = [
        "chest_pain",
        "shortness_of_breath",
        "irregular_heartbeat",
        "fatigue_weakness",
        "dizziness",
        "swelling_edema",
        "pain_neck_jaw_shoulder_back",
        "excessive_sweating",
        "persistent_cough",
        "nausea_vomiting",
        "high_blood_pressure",
        "chest_discomfort_activity",
        "cold_hands_feet",
        "snoring_sleep_apnea",
        "anxiety_feeling_of_doom",
    ];

    const payload = {};
    checkboxFields.forEach((field) => {
        const cb = form.querySelector(`[name="${field}"]`);
        payload[field] = cb.checked ? 1 : 0;
    });

    const ageValue = parseFloat(form.querySelector('[name="age"]').value);
    if (isNaN(ageValue) || ageValue < 0 || ageValue > 120) {
        showError("Please enter a valid age between 0 and 120.");
        return;
    }
    payload.age = ageValue;

    // Disable button while loading
    predictBtn.disabled = true;
    predictBtn.textContent = "Predicting...";

    try {
        const response = await fetch(`${API_BASE}/api/v1/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => null);
            throw new Error(errData?.detail || `Server error (${response.status})`);
        }

        const data = await response.json();
        lastPredictionContext = { input: payload, result: data };
        renderResults(data);
    } catch (err) {
        showError(err.message || "Failed to connect to the API. Make sure the backend is running.");
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = "Predict";
    }
});

function renderResults(data) {
    // Ensemble result
    const isHighRisk = data.ensemble_prediction === 1;
    const pct = (data.ensemble_probability * 100).toFixed(1);
    ensembleDiv.className = `ensemble-box ${isHighRisk ? "high-risk" : "low-risk"}`;
    ensembleDiv.innerHTML = isHighRisk
        ? `Ensemble Verdict: <span>At Risk</span> (${pct}% probability)`
        : `Ensemble Verdict: <span>Low Risk</span> (${pct}% probability)`;

    // Model comparison chart
    renderModelComparisonChart(data.predictions);

    // SHAP feature importance chart
    if (data.shap_contributions && data.shap_contributions.length > 0) {
        renderShapChart(data.shap_contributions);
    }

    // Individual models
    modelResultsDiv.innerHTML = "";
    data.predictions.forEach((pred) => {
        const atRisk = pred.prediction === 1;
        const card = document.createElement("div");
        card.className = `model-card ${atRisk ? "at-risk" : "no-risk"}`;
        card.innerHTML = `
            <div class="model-name">${formatModelName(pred.model_name)}</div>
            <div class="model-verdict">${atRisk ? "At Risk" : "Low Risk"}</div>
            <div class="model-prob">${(pred.probability_at_risk * 100).toFixed(1)}% risk probability</div>
        `;
        modelResultsDiv.appendChild(card);
    });

    resultsDiv.classList.remove("hidden");
    feedbackSection.classList.remove("hidden");
    resultsDiv.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderModelComparisonChart(predictions) {
    // Destroy previous chart if it exists
    if (modelComparisonChart) {
        modelComparisonChart.destroy();
    }

    const ctx = document.getElementById("model-comparison-chart").getContext("2d");

    // Sort by probability descending
    const sorted = [...predictions].sort(
        (a, b) => b.probability_at_risk - a.probability_at_risk
    );

    const labels = sorted.map((p) => formatModelName(p.model_name));
    const values = sorted.map((p) => +(p.probability_at_risk * 100).toFixed(1));
    const colors = sorted.map((p) =>
        p.prediction === 1 ? "rgba(229, 62, 62, 0.75)" : "rgba(56, 161, 105, 0.75)"
    );
    const borders = sorted.map((p) =>
        p.prediction === 1 ? "rgba(197, 48, 48, 1)" : "rgba(39, 103, 73, 1)"
    );

    modelComparisonChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [
                {
                    label: "Risk Probability (%)",
                    data: values,
                    backgroundColor: colors,
                    borderColor: borders,
                    borderWidth: 1,
                },
            ],
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.parsed.x}% risk probability`,
                    },
                },
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 100,
                    title: { display: true, text: "Risk Probability (%)" },
                },
            },
        },
    });
}

function renderShapChart(contributions) {
    // Destroy previous chart if it exists
    if (shapChart) {
        shapChart.destroy();
    }

    const ctx = document.getElementById("shap-chart").getContext("2d");

    // Already sorted by absolute value from backend, take all features
    const labels = contributions.map((c) => c.feature_name);
    const values = contributions.map((c) => c.shap_value);
    const colors = values.map((v) =>
        v > 0 ? "rgba(229, 62, 62, 0.75)" : "rgba(56, 161, 105, 0.75)"
    );
    const borders = values.map((v) =>
        v > 0 ? "rgba(197, 48, 48, 1)" : "rgba(39, 103, 73, 1)"
    );

    shapChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [
                {
                    label: "SHAP Value",
                    data: values,
                    backgroundColor: colors,
                    borderColor: borders,
                    borderWidth: 1,
                },
            ],
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const val = ctx.parsed.x;
                            const dir = val > 0 ? "increases" : "decreases";
                            return `${dir} risk by ${Math.abs(val).toFixed(4)}`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: "SHAP Value (red = increases risk, green = decreases risk)",
                    },
                },
            },
        },
    });
}

function showError(message) {
    errorDiv.textContent = message;
    errorDiv.classList.remove("hidden");
}

// --- Feedback form ---
feedbackForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const hadStrokeRadio = feedbackForm.querySelector('input[name="had_stroke"]:checked');
    if (!hadStrokeRadio) {
        showError("Please indicate whether you have experienced a stroke.");
        return;
    }

    const payload = {
        name: feedbackForm.querySelector('[name="name"]').value.trim(),
        email: feedbackForm.querySelector('[name="email"]').value.trim(),
        had_stroke: hadStrokeRadio.value === "true",
        comment: feedbackForm.querySelector('[name="comment"]').value.trim(),
        prediction_input: lastPredictionContext?.input || null,
        prediction_result: lastPredictionContext?.result || null,
    };

    feedbackBtn.disabled = true;
    feedbackBtn.textContent = "Submitting...";

    try {
        const response = await fetch(`${API_BASE}/api/v1/feedback`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => null);
            throw new Error(errData?.detail || `Server error (${response.status})`);
        }

        alert("Thank you for your feedback! Your response has been recorded.");
        window.location.reload();
    } catch (err) {
        showError(err.message || "Failed to submit feedback.");
        feedbackBtn.disabled = false;
        feedbackBtn.textContent = "Submit Feedback";
    }
});
