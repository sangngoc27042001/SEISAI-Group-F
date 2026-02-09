"""
Stroke Risk Prediction FastAPI Application
Main application entry point.
"""

from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from .routers import router, health_router
from .feedback_router import feedback_router
from .services import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - load models on startup."""
    print("Loading ML models...")
    model_service.load_models()
    print(f"Loaded {len(model_service.models)} models successfully.")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Stroke Risk Prediction API",
    description="""
    API for predicting stroke risk using multiple machine learning models.

    ## Features
    - Predict stroke risk using 10 different ML models
    - Get ensemble predictions (majority vote and average probability)
    - Select specific models for prediction

    ## Models Available
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Gradient Boosting
    - AdaBoost
    - SVM
    - K-Nearest Neighbors
    - Naive Bayes
    - XGBoost
    - LightGBM
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(router)
app.include_router(feedback_router)


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "name": "Stroke Risk Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
    }
