from typing import Any

import numpy as np
import pandas as pd


def explain_prediction(artifact: dict[str, Any], features: pd.DataFrame) -> dict[str, Any]:
    pipeline = artifact["pipeline"]
    metadata = artifact["metadata"]
    estimator = pipeline.named_steps["classifier"]

    if hasattr(estimator, "coef_") and len(estimator.classes_) == 2:
        transformed = pipeline.named_steps["preprocessor"].transform(features)
        coefficients = np.asarray(estimator.coef_[0], dtype=float)
        values = np.asarray(transformed[0], dtype=float)
        intercept = float(np.asarray(estimator.intercept_).reshape(-1)[0])
        names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
        contributions = [
            {
                "feature": str(name),
                "transformed_value": float(value),
                "contribution": float(value * coefficient),
            }
            for name, value, coefficient in zip(names, values, coefficients)
        ]
        contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        decision_score = float(estimator.decision_function(transformed)[0])
        return {
            "method": "Logistic Regression additive log-odds contributions for estimator class 1",
            "scope": "this input",
            "class_value": int(estimator.classes_[1]),
            "class_label": metadata["class_labels"][str(estimator.classes_[1])],
            "intercept": intercept,
            "decision_score": decision_score,
            "contribution_sum": float(sum(item["contribution"] for item in contributions)),
            "features": contributions,
        }

    global_importance = metadata.get("feature_importance", [])
    method = metadata.get("explanation_method")
    if method and global_importance:
        return {
            "method": method,
            "scope": "global model summary; not this input",
            "features": global_importance,
        }
    return {
        "method": None,
        "scope": "unsupported",
        "features": [],
    }