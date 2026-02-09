"""
Stroke Risk Prediction - Multiple ML Models Training Script
Binary classification to predict stroke risk from patient symptoms and demographics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings('ignore')


def load_data(data_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load and prepare the stroke risk dataset."""
    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.drop(columns=['At Risk (Binary)', 'Stroke Risk (%)'])
    y = df['At Risk (Binary)']

    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")

    return X, y


def get_models() -> dict:
    """Return a dictionary of models to train."""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    }
    return models


def evaluate_model(model, X_train, X_test, y_train, y_test) -> dict:
    """Train and evaluate a single model."""
    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob) if y_prob is not None else None,
    }

    return metrics


def cross_validate_models(models: dict, X, y, cv: int = 5) -> pd.DataFrame:
    """Perform cross-validation for all models."""
    results = []
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
        results.append({
            'Model': name,
            'CV F1 Mean': scores.mean(),
            'CV F1 Std': scores.std(),
        })
        print(f"{name}: CV F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")

    return pd.DataFrame(results)


def save_models(models: dict, scaler: StandardScaler, models_dir: Path):
    """Save all trained models and the scaler to disk."""
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save scaler
    scaler_path = models_dir / 'scaler.joblib'
    joblib.dump(scaler, scaler_path)
    print(f"  Saved scaler to: {scaler_path}")

    # Save each model
    for name, model in models.items():
        # Create filename from model name
        filename = name.lower().replace(' ', '_').replace('-', '_') + '.joblib'
        model_path = models_dir / filename
        joblib.dump(model, model_path)
        print(f"  Saved {name} to: {model_path}")


def train_and_compare(X, y, models_dir: Path, test_size: float = 0.2):
    """Train all models and compare their performance."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Get models
    models = get_models()

    # Evaluate each model
    results = []
    trained_models = {}
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        metrics = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
        metrics['Model'] = name
        results.append(metrics)
        trained_models[name] = model

        print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall:    {metrics['Recall']:.4f}")
        print(f"  F1-Score:  {metrics['F1-Score']:.4f}")
        if metrics['ROC-AUC']:
            print(f"  ROC-AUC:   {metrics['ROC-AUC']:.4f}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
    results_df = results_df.sort_values('F1-Score', ascending=False)

    print("\n" + "="*60)
    print("SUMMARY - Models Ranked by F1-Score")
    print("="*60)
    print(results_df.to_string(index=False))

    # Cross-validation
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS (5-Fold)")
    print("="*60)

    # Scale all data for CV
    X_scaled = scaler.fit_transform(X)
    cv_results = cross_validate_models(models, X_scaled, y)

    # Find best model
    best_model_name = results_df.iloc[0]['Model']
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"{'='*60}")

    # Save all trained models
    print(f"\n{'='*60}")
    print("SAVING MODELS")
    print("="*60)
    save_models(trained_models, scaler, models_dir)

    return results_df, cv_results


def main():
    # Path to dataset (relative to project root)
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'stroke_risk_dataset.csv'
    models_dir = project_root / 'trained_models'

    print("="*60)
    print("STROKE RISK PREDICTION - ML MODEL COMPARISON")
    print("="*60)

    # Load data
    X, y = load_data(data_path)

    # Train and compare models
    results_df, cv_results = train_and_compare(X, y, models_dir)

    # Merge test results with CV results
    combined_results = results_df.merge(cv_results, on='Model')
    combined_results = combined_results.sort_values('F1-Score', ascending=False)

    # Save results
    results_path = project_root / 'model_results.csv'
    combined_results.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    print(f"Models saved to: {models_dir}/")


if __name__ == '__main__':
    main()
