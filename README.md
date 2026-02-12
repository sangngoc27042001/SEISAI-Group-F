# Stroke Risk Prediction System

A machine learning system for binary classification of stroke risk based on patient symptoms and demographics.

## Live Demo

The application is deployed on Rahti (CSC OpenShift):

- **Frontend**: https://stroke-fe-sdx-assignment-ngvo.2.rahtiapp.fi
- **Backend API**: https://stroke-api-sdx-assignment-ngvo.2.rahtiapp.fi

## Problem Description

This project predicts whether a patient is at risk of stroke using clinical symptoms and demographic data. The system uses supervised learning techniques to classify patients into "At Risk" or "Not At Risk" categories.

## Dataset

The dataset contains 16 input features:
- **Symptoms**: Chest Pain, Shortness of Breath, Irregular Heartbeat, Fatigue & Weakness, Dizziness, Swelling, Pain in Neck/Jaw/Shoulder/Back, Excessive Sweating, Persistent Cough, Nausea/Vomiting, Chest Discomfort, Cold Hands/Feet, Snoring/Sleep Apnea, Anxiety
- **Medical indicators**: High Blood Pressure
- **Demographics**: Age

**Target variable**: `At Risk (Binary)` - 0 (not at risk) or 1 (at risk)

## Getting Started

### 1. Setup

```bash
cp .env.sample .env   # Copy and fill in your environment variables
make setup             # Install uv, create venv, install dependencies
```

Edit `.env` with your values:

```
MONGODB_URI="your-mongodb-uri"
BE_BASE_URL="http://localhost:8000"
```

### 2. Train models

```bash
make train
```

### 3. Run the app

Start the backend and frontend in **separate terminals**:

```bash
make run-api  # Backend  → http://localhost:8000
make run-fe   # Frontend → http://localhost:3000
```

### 4. Deploy to Rahti

```bash
make deploy      # Build & deploy both BE and FE
make deploy-api  # Backend only
make deploy-fe   # Frontend only
```

Requires `oc` CLI logged in and `docker` running.

## Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: Proportion of positive predictions that are correct
- **Recall**: Proportion of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall (primary metric)
- **ROC-AUC**: Model's ability to distinguish between classes

## Explainable AI with SHAP

The system uses [SHAP (SHapley Additive exPlanations)](https://shap.readthedocs.io/) to explain **why** a particular prediction was made, not just what the prediction is.

### What are SHAP values?

SHAP values originate from cooperative game theory (Shapley values). The core idea is to answer: **how much did each feature (symptom) contribute to pushing the prediction away from the baseline (average) toward the final result?**

For a given prediction, each feature receives a SHAP value:
- **Positive SHAP value** → the feature *increased* the predicted stroke risk
- **Negative SHAP value** → the feature *decreased* the predicted stroke risk
- **Magnitude** → how strongly the feature influenced the prediction

### How we compute SHAP values

We use model-specific explainers for speed (real-time predictions require fast computation):

| Explainer | Models | Speed |
|-----------|--------|-------|
| `TreeExplainer` | Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM | Near-instant (exact solution using tree structure) |
| `LinearExplainer` | Logistic Regression | Near-instant (exact solution using model coefficients) |
| *Skipped* | SVM, KNN, Naive Bayes | `KernelExplainer` is too slow for real-time use |

The final SHAP contributions shown to the user are **averaged across all 6 supported models**, giving a robust ensemble explanation. Features are sorted by absolute SHAP value so the most influential symptoms appear first.

### Example interpretation

If a patient with High Blood Pressure and Chest Pain receives:

```
High Blood Pressure    +0.46  (red  — increases risk)
Chest Pain             +0.46  (red  — increases risk)
Cold Hands/Feet        -0.49  (green — decreases risk, symptom absent)
```

This tells us the model's prediction was primarily driven by the presence of High Blood Pressure and Chest Pain, while the absence of Cold Hands/Feet pulled the risk score down.

## Team Members

- Niko Hokkanen
- Erik Johansson
- Eetu Koivupalo
- Dinusha Pilana Gardiya Godakandage
- Sang Vo
