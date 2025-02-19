"""
AquaPulse — Water Quality Prediction ML Pipeline
Author: Mohammed Kaif Ahmed (GitHub: kaifahmedcode02)
Fixes legacy integer truncation, column swap, unscaled KNN, and NaN drop issues.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

FEATURE_COLUMNS = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]

INDUSTRY_STANDARDS = {
    "pharma": {
        "ph": (6.8, 7.4), "Hardness": (47, 85), "Solids": (320, 2500),
        "Chloramines": (0.5, 1.5), "Sulfate": (129, 180), "Conductivity": (181, 240),
        "Organic_carbon": (2.2, 4.5), "Trihalomethanes": (0.7, 20.0), "Turbidity": (1.45, 1.85)
    },
    "municipal": {
        "ph": (6.5, 8.5), "Hardness": (60, 200), "Solids": (100, 20000),
        "Chloramines": (0.5, 4.0), "Sulfate": (129, 350), "Conductivity": (181, 450),
        "Organic_carbon": (2.2, 14.0), "Trihalomethanes": (0.7, 80.0), "Turbidity": (1.45, 4.2)
    },
    "boiler": {
        "ph": (8.0, 9.2), "Hardness": (47, 110), "Solids": (320, 15000),
        "Chloramines": (0.5, 3.0), "Sulfate": (129, 280), "Conductivity": (181, 380),
        "Organic_carbon": (2.2, 10.0), "Trihalomethanes": (0.7, 60.0), "Turbidity": (1.45, 3.0)
    },
    "agricultural": {
        "ph": (5.5, 9.0), "Hardness": (47, 300), "Solids": (320, 42000),
        "Chloramines": (0.5, 7.5), "Sulfate": (129, 435), "Conductivity": (181, 640),
        "Organic_carbon": (2.2, 20.0), "Trihalomethanes": (0.7, 100.0), "Turbidity": (1.45, 5.2)
    },
}


def build_aquapulse_pipeline():
    """Constructs the leakage-free Imputation + Scaling + SMOTE + Soft-Voting Ensemble pipeline."""
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=14, random_state=42, class_weight="balanced"
    )
    gbc = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
    )
    knn = KNeighborsClassifier(
        n_neighbors=9, weights="distance", metric="minkowski"
    )

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gbc", gbc), ("knn", knn)],
        voting="soft",
        weights=[2, 2, 1]
    )

    pipeline = ImbPipeline([
        ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("classifier", ensemble)
    ])
    return pipeline


def match_industry_standard(sample_dict):
    """Evaluates a 9-parameter water sample against 4 industrial regulatory benchmarks."""
    scores = {}
    for std_name, limits in INDUSTRY_STANDARDS.items():
        passed = sum(1 for k, (low, high) in limits.items() if low <= sample_dict[k] <= high)
        scores[std_name] = round((passed / len(limits)) * 100, 1)
    best_match = max(scores, key=scores.get)
    return {"best_category": best_match, "compatibility_scores": scores}


if __name__ == "__main__":
    print("AquaPulse ML Pipeline initialized — Engineered by Mohammed Kaif Ahmed (GitHub: kaifahmedcode02)")
