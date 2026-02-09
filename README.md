# Stroke Risk Prediction System

A machine learning system for binary classification of stroke risk based on patient symptoms and demographics.

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

## Team Members

- Niko Hokkanen
- Erik Johansson
- Eetu Koivupalo
- Dinusha Pilana Gardiya Godakandage
- Sang Vo
