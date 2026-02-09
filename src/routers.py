"""
Stroke Risk Prediction API Routers
Defines API endpoints for predictions.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from .services import (
    model_service,
    PatientInput,
    PredictionResponse,
    SingleModelPrediction,
    MODEL_NAMES,
    FEATURE_NAMES,
)

router = APIRouter(prefix="/api/v1", tags=["predictions"])


@router.get("/models")
def get_available_models() -> dict:
    """Get list of available models."""
    return {
        "available_models": model_service.get_available_models(),
        "total_models": len(model_service.get_available_models()),
    }


@router.get("/features")
def get_feature_info() -> dict:
    """Get information about required input features."""
    return {
        "features": FEATURE_NAMES,
        "total_features": len(FEATURE_NAMES),
        "description": "All symptom features are binary (0 or 1), Age is numeric (0-120)",
    }


@router.post("/predict", response_model=PredictionResponse)
def predict_all(patient: PatientInput) -> PredictionResponse:
    """
    Make stroke risk predictions using all available models.

    Returns predictions from each model along with ensemble results.
    """
    try:
        return model_service.predict_all_models(patient)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model files not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/predict/{model_name}", response_model=SingleModelPrediction)
def predict_single(model_name: str, patient: PatientInput) -> SingleModelPrediction:
    """
    Make stroke risk prediction using a specific model.

    Available models:
    - logistic_regression
    - decision_tree
    - random_forest
    - gradient_boosting
    - adaboost
    - svm
    - k_nearest_neighbors
    - naive_bayes
    - xgboost
    - lightgbm
    """
    try:
        return model_service.predict_single_model(patient, model_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model files not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/predict/selected", response_model=PredictionResponse)
def predict_selected(
    patient: PatientInput,
    models: list[str] = Query(..., description="List of model names to use for prediction"),
) -> PredictionResponse:
    """
    Make stroke risk predictions using selected models.

    Pass model names as query parameters, e.g.:
    /predict/selected?models=logistic_regression&models=xgboost&models=lightgbm
    """
    try:
        available = model_service.get_available_models()
        invalid_models = [m for m in models if m not in available]
        if invalid_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid models: {invalid_models}. Available: {available}",
            )

        predictions = []
        total_prob_at_risk = 0.0
        total_prediction = 0

        for model_name in models:
            pred = model_service.predict_single_model(patient, model_name)
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
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model files not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Health check router
health_router = APIRouter(tags=["health"])


@health_router.get("/health")
def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy"}


@health_router.get("/ready")
def readiness_check() -> dict:
    """Readiness check - verifies models are loaded."""
    try:
        model_service.load_models()
        return {
            "status": "ready",
            "models_loaded": len(model_service.models),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Not ready: {str(e)}")
