from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.app.catalog import DISEASES
from backend.app.config import DATASET_DIR, MODEL_DIR, REPORT_DIR
from backend.app.database import save_prediction
from backend.app.services.explain_service import explain_prediction


def model_path(disease: str) -> Path:
    return MODEL_DIR / f"{disease}.joblib"


def metrics_path(disease: str) -> Path:
    return REPORT_DIR / f"{disease}.json"


@lru_cache(maxsize=3)
def _load_artifact(path_string: str) -> dict[str, Any]:
    artifact = joblib.load(path_string)
    if (
        not isinstance(artifact, dict)
        or not all(key in artifact for key in ("pipeline", "metadata"))
        or not isinstance(artifact["metadata"], dict)
        or not all(
            key in artifact["metadata"]
            for key in ("feature_names", "features", "class_labels")
        )
    ):
        raise ValueError("The saved model artifact has an unsupported structure.")
    return artifact


def get_artifact(disease: str) -> dict[str, Any]:
    if disease not in DISEASES:
        raise KeyError(disease)
    path = model_path(disease)
    if not path.is_file():
        raise FileNotFoundError(f"No trained {disease} model is available yet.")
    return _load_artifact(str(path.resolve()))


def model_status(disease: str) -> dict[str, Any]:
    available = model_path(disease).is_file() and metrics_path(disease).is_file()
    return {
        **DISEASES[disease],
        "id": disease,
        "available": available,
        "features": get_artifact(disease)["metadata"]["features"] if available else [],
    }


def load_metrics(disease: str) -> dict[str, Any]:
    path = metrics_path(disease)
    if not path.is_file():
        raise FileNotFoundError(f"No evaluation report is available for {disease}.")
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def validate_features(disease: str, values: dict[str, Any]) -> tuple[dict[str, Any], dict]:
    artifact = get_artifact(disease)
    metadata = artifact["metadata"]
    expected = metadata["feature_names"]
    if set(values) != set(expected):
        missing = sorted(set(expected) - set(values))
        extra = sorted(set(values) - set(expected))
        raise ValueError(f"Feature mismatch. Missing: {missing}; unexpected: {extra}.")

    normalized: dict[str, Any] = {}
    schema_by_name = {feature["name"]: feature for feature in metadata["features"]}
    for name in expected:
        value = values[name]
        schema = schema_by_name[name]
        if schema["kind"] == "categorical":
            if value not in schema["categories"]:
                raise ValueError(f"Unsupported category for {name}.")
            normalized[name] = value
        else:
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be numeric.") from exc
            if not np.isfinite(numeric):
                raise ValueError(f"{name} must be a finite number.")
            if numeric < schema["minimum"] or numeric > schema["maximum"]:
                raise ValueError(
                    f"{name} must be within the observed dataset range "
                    f"({schema['minimum']:g} to {schema['maximum']:g})."
                )
            normalized[name] = numeric
    return normalized, artifact


def predict(disease: str, values: dict[str, Any], persist: bool = True) -> dict[str, Any]:
    normalized, artifact = validate_features(disease, values)
    names = artifact["metadata"]["feature_names"]
    features = pd.DataFrame([[normalized[name] for name in names]], columns=names)
    prediction = int(artifact["pipeline"].predict(features)[0])
    labels = artifact["metadata"]["class_labels"]
    predicted_class = labels[str(prediction)]
    positive_score = None
    pipeline = artifact["pipeline"]
    if hasattr(pipeline, "predict_proba"):
        classes = list(pipeline.classes_)
        if 1 in classes:
            positive_score = float(pipeline.predict_proba(features)[0][classes.index(1)])
    result = {
        "disease": disease,
        "predicted_class": predicted_class,
        "class_value": prediction,
        "positive_score": positive_score,
        "score_label": "Model score for the dataset positive class (not necessarily calibrated)",
        "model_name": artifact["metadata"]["selected_model"],
        "model_version": artifact["metadata"]["trained_at"],
        "target_definition": DISEASES[disease]["target"],
        "explanation": explain_prediction(artifact, features),
    }
    if persist:
        record = save_prediction(disease, predicted_class, positive_score)
        result["history_id"] = record.id
    return result


def dataset_summary(disease: str) -> dict[str, Any]:
    artifact = get_artifact(disease)
    path = DATASET_DIR / f"{disease}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"The local dataset export for {disease} is unavailable.")
    frame = pd.read_csv(path)
    target = "target"
    preview = frame.head(12)
    return {
        "name": DISEASES[disease]["dataset"],
        "source": DISEASES[disease]["source"],
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "features": artifact["metadata"]["features"],
        "missing": {key: int(value) for key, value in frame.isna().sum().items()},
        "duplicates": int(frame.duplicated().sum()),
        "target_distribution": {
            str(key): int(value) for key, value in frame[target].value_counts(dropna=False).items()
        },
        "preview": preview.astype(object).where(pd.notna(preview), None).to_dict("records"),
        "records": frame.astype(object).where(pd.notna(frame), None).to_dict("records"),
    }