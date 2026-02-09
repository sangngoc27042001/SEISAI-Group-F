"""
Stroke Risk Prediction Services
Handles model loading and prediction logic.
"""

from pathlib import Path
from typing import Optional
import joblib
import numpy as np
from pydantic import BaseModel, Field


# Feature names in the expected order
FEATURE_NAMES = [
    "Chest Pain",
    "Shortness of Breath",
    "Irregular Heartbeat",
    "Fatigue & Weakness",
    "Dizziness",
    "Swelling (Edema)",
    "Pain in Neck/Jaw/Shoulder/Back",
    "Excessive Sweating",
    "Persistent Cough",
    "Nausea/Vomiting",
    "High Blood Pressure",
    "Chest Discomfort (Activity)",
    "Cold Hands/Feet",
    "Snoring/Sleep Apnea",
    "Anxiety/Feeling of Doom",
    "Age",
]

# Available models
MODEL_NAMES = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "adaboost",
    "svm",
    "k_nearest_neighbors",
    "naive_bayes",
    "xgboost",
    "lightgbm",
]


class PatientInput(BaseModel):
    """Input schema for patient data."""
    chest_pain: int = Field(..., ge=0, le=1, description="Chest Pain (0 or 1)")
    shortness_of_breath: int = Field(..., ge=0, le=1, description="Shortness of Breath (0 or 1)")
    irregular_heartbeat: int = Field(..., ge=0, le=1, description="Irregular Heartbeat (0 or 1)")
    fatigue_weakness: int = Field(..., ge=0, le=1, description="Fatigue & Weakness (0 or 1)")
    dizziness: int = Field(..., ge=0, le=1, description="Dizziness (0 or 1)")
    swelling_edema: int = Field(..., ge=0, le=1, description="Swelling/Edema (0 or 1)")
    pain_neck_jaw_shoulder_back: int = Field(..., ge=0, le=1, description="Pain in Neck/Jaw/Shoulder/Back (0 or 1)")
    excessive_sweating: int = Field(..., ge=0, le=1, description="Excessive Sweating (0 or 1)")
    persistent_cough: int = Field(..., ge=0, le=1, description="Persistent Cough (0 or 1)")
    nausea_vomiting: int = Field(..., ge=0, le=1, description="Nausea/Vomiting (0 or 1)")
    high_blood_pressure: int = Field(..., ge=0, le=1, description="High Blood Pressure (0 or 1)")
    chest_discomfort_activity: int = Field(..., ge=0, le=1, description="Chest Discomfort during Activity (0 or 1)")
    cold_hands_feet: int = Field(..., ge=0, le=1, description="Cold Hands/Feet (0 or 1)")
    snoring_sleep_apnea: int = Field(..., ge=0, le=1, description="Snoring/Sleep Apnea (0 or 1)")
    anxiety_feeling_of_doom: int = Field(..., ge=0, le=1, description="Anxiety/Feeling of Doom (0 or 1)")
    age: float = Field(..., ge=0, le=120, description="Age in years")


class SingleModelPrediction(BaseModel):
    """Prediction result from a single model."""
    model_name: str
    prediction: int = Field(..., description="0 = No Risk, 1 = At Risk")
    probability_no_risk: float = Field(..., description="Probability of no stroke risk")
    probability_at_risk: float = Field(..., description="Probability of stroke risk")


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    predictions: list[SingleModelPrediction]
    ensemble_prediction: int = Field(..., description="Majority vote across all models")
    ensemble_probability: float = Field(..., description="Average probability of stroke risk")


class ModelService:
    """Service for loading models and making predictions."""

    def __init__(self, models_dir: Optional[Path] = None):
        """Initialize the model service."""
        if models_dir is None:
            models_dir = Path(__file__).parent.parent / "trained_models"
        self.models_dir = models_dir
        self.models: dict = {}
        self.scaler = None
        self._loaded = False

    def load_models(self) -> None:
        """Load all models and the scaler from disk."""
        if self._loaded:
            return

        # Load scaler
        scaler_path = self.models_dir / "scaler.joblib"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        self.scaler = joblib.load(scaler_path)

        # Load all models
        for model_name in MODEL_NAMES:
            model_path = self.models_dir / f"{model_name}.joblib"
            if model_path.exists():
                self.models[model_name] = joblib.load(model_path)
            else:
                print(f"Warning: Model {model_name} not found at {model_path}")

        self._loaded = True

    def get_available_models(self) -> list[str]:
        """Return list of available model names."""
        if not self._loaded:
            self.load_models()
        return list(self.models.keys())

    def _input_to_array(self, patient: PatientInput) -> np.ndarray:
        """Convert patient input to numpy array in correct feature order."""
        return np.array([[
            patient.chest_pain,
            patient.shortness_of_breath,
            patient.irregular_heartbeat,
            patient.fatigue_weakness,
            patient.dizziness,
            patient.swelling_edema,
            patient.pain_neck_jaw_shoulder_back,
            patient.excessive_sweating,
            patient.persistent_cough,
            patient.nausea_vomiting,
            patient.high_blood_pressure,
            patient.chest_discomfort_activity,
            patient.cold_hands_feet,
            patient.snoring_sleep_apnea,
            patient.anxiety_feeling_of_doom,
            patient.age,
        ]])

    def predict_single_model(self, patient: PatientInput, model_name: str) -> SingleModelPrediction:
        """Make prediction using a single model."""
        if not self._loaded:
            self.load_models()

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")

        model = self.models[model_name]
        X = self._input_to_array(patient)
        X_scaled = self.scaler.transform(X)

        prediction = int(model.predict(X_scaled)[0])

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_scaled)[0]
            prob_no_risk = float(probabilities[0])
            prob_at_risk = float(probabilities[1])
        else:
            prob_no_risk = 1.0 - prediction
            prob_at_risk = float(prediction)

        return SingleModelPrediction(
            model_name=model_name,
            prediction=prediction,
            probability_no_risk=prob_no_risk,
            probability_at_risk=prob_at_risk,
        )

    def predict_all_models(self, patient: PatientInput) -> PredictionResponse:
        """Make predictions using all available models."""
        if not self._loaded:
            self.load_models()

        predictions = []
        total_prob_at_risk = 0.0
        total_prediction = 0

        for model_name in self.models:
            pred = self.predict_single_model(patient, model_name)
            predictions.append(pred)
            total_prob_at_risk += pred.probability_at_risk
            total_prediction += pred.prediction

        num_models = len(predictions)
        ensemble_probability = total_prob_at_risk / num_models if num_models > 0 else 0.0
        ensemble_prediction = 1 if total_prediction > num_models / 2 else 0

        return PredictionResponse(
            predictions=predictions,
            ensemble_prediction=ensemble_prediction,
            ensemble_probability=ensemble_probability,
        )


# Global service instance
model_service = ModelService()
