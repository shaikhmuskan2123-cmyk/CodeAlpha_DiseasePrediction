"""Download documented datasets, compare models, and save reproducible pipelines."""

from datetime import datetime, timezone
import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer

from backend.app.config import DATASET_DIR, MODEL_DIR, REPORT_DIR

RANDOM_STATE = 42
UCI_FEATURES = {
    "heart": ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
    "diabetes": [
        "gender", "polyuria", "polydipsia", "sudden_weight_loss", "weakness",
        "polyphagia", "genital_thrush", "visual_blurring", "itching", "irritability",
        "delayed_healing", "partial_paresis", "muscle_stiffness", "alopecia", "obesity",
    ],
}
CATEGORY_LABELS = {
    "sex": {"0": "Female", "1": "Male"},
    "cp": {
        "1": "Typical angina",
        "2": "Atypical angina",
        "3": "Non-anginal pain",
        "4": "Asymptomatic",
    },
    "fbs": {"0": "Not greater than 120 mg/dL", "1": "Greater than 120 mg/dL"},
    "restecg": {
        "0": "Normal",
        "1": "ST-T wave abnormality",
        "2": "Probable or definite left ventricular hypertrophy",
    },
    "exang": {"0": "No", "1": "Yes"},
    "slope": {"1": "Upsloping", "2": "Flat", "3": "Downsloping"},
    "ca": {"0": "0 vessels", "1": "1 vessel", "2": "2 vessels", "3": "3 vessels"},
    "thal": {"3": "Normal", "6": "Fixed defect", "7": "Reversible defect"},
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def fetch_dataset(disease: str) -> tuple[pd.DataFrame, pd.Series, str]:
    if disease in {"heart", "diabetes"}:
        from ucimlrepo import fetch_ucirepo

        dataset_id = 45 if disease == "heart" else 529
        source = fetch_ucirepo(id=dataset_id)
        features = source.data.features.copy()
        target = source.data.targets.iloc[:, 0].copy()
        features.columns = [_slug(column) for column in features.columns]
        features = features.replace("?", np.nan)
        if disease == "heart":
            target = pd.to_numeric(target, errors="coerce")
            keep = target.notna()
            features, target = features.loc[keep], target.loc[keep]
            target = (target.astype(int) > 0).astype(int)
        else:
            target = target.astype(str).str.strip().str.lower().map(
                {"positive": 1, "negative": 0}
            )
            keep = target.notna()
            features, target = features.loc[keep], target.loc[keep]
            target = target.astype(int)
        return features.reset_index(drop=True), target.reset_index(drop=True), "UCI CC BY 4.0"

    source = load_breast_cancer(as_frame=True)
    features = source.data.copy()
    target = (source.target == 0).astype(int)
    return features, target, "Wisconsin Diagnostic Breast Cancer; UCI source; scikit-learn BSD-3-Clause"


def _feature_metadata(frame: pd.DataFrame, categorical: list[str]) -> list[dict]:
    output = []
    for name in frame.columns:
        values = frame[name].dropna()
        if name in categorical:
            categories = sorted(values.unique().tolist(), key=str)
            labels = CATEGORY_LABELS.get(name, {})
            def code_key(value: object) -> str:
                if isinstance(value, (int, float, np.integer, np.floating)) and float(value).is_integer():
                    return str(int(value))
                return str(value)

            output.append(
                {
                    "name": name,
                    "kind": "categorical",
                    "categories": categories,
                    "category_labels": {
                        code_key(value): labels.get(code_key(value), code_key(value))
                        for value in categories
                    },
                }
            )
        else:
            numeric = pd.to_numeric(values, errors="coerce").dropna()
            output.append(
                {
                    "name": name,
                    "kind": "numeric",
                    "minimum": float(numeric.min()),
                    "maximum": float(numeric.max()),
                    "median": float(numeric.median()),
                }
            )
    return output


def _candidate_models() -> list[tuple[str, object, dict]]:
    return [
        (
            "Logistic Regression",
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            {"classifier__C": [0.1, 1.0]},
        ),
        (
            "Random Forest",
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {"classifier__n_estimators": [100], "classifier__max_depth": [None, 8]},
        ),
        (
            "Support Vector Machine",
            SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
            {"classifier__C": [0.5, 1.0], "classifier__kernel": ["linear", "rbf"]},
        ),
        (
            "Gradient Boosting",
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"classifier__n_estimators": [80, 120], "classifier__learning_rate": [0.05, 0.1]},
        ),
    ]


def _make_preprocessor(frame: pd.DataFrame, categorical: list[str]) -> ColumnTransformer:
    numeric = [name for name in frame.columns if name not in categorical]
    numeric_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def train_one(disease: str) -> dict:
    features, target, license_note = fetch_dataset(disease)
    categorical = UCI_FEATURES.get(disease, [])
    categorical = [name for name in categorical if name in features.columns]
    features = features.replace([np.inf, -np.inf], np.nan)
    if disease == "heart":
        class_labels = {"0": "Dataset negative (target 0)", "1": "Dataset positive (target 1-4)"}
        positive_name = "angiographic target 1-4"
    elif disease == "diabetes":
        class_labels = {"0": "Dataset negative", "1": "Dataset positive"}
        positive_name = "UCI questionnaire class: Positive"
    else:
        class_labels = {"0": "Benign (dataset class)", "1": "Malignant (dataset class)"}
        positive_name = "malignant"

    dataset = features.copy()
    dataset["target"] = target.to_numpy()
    dataset.to_csv(DATASET_DIR / f"{disease}.csv", index=False)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    comparisons = []
    best = None
    for name, estimator, grid in _candidate_models():
        pipeline = Pipeline(
            [("preprocessor", _make_preprocessor(features, categorical)), ("classifier", estimator)]
        )
        search = GridSearchCV(
            pipeline,
            grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        search.fit(x_train, y_train)
        comparison = {
            "model": name,
            "cv_roc_auc_mean": float(search.best_score_),
            "cv_roc_auc_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            "best_params": search.best_params_,
        }
        comparisons.append(comparison)
        if best is None or search.best_score_ > best["score"]:
            best = {"score": search.best_score_, "name": name, "pipeline": search.best_estimator_}

    pipeline = best["pipeline"]
    predictions = pipeline.predict(x_test)
    scores = pipeline.predict_proba(x_test)[:, list(pipeline.classes_).index(1)]
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    metrics = {
        "disease": disease,
        "selected_model": best["name"],
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "brier_score": float(brier_score_loss(y_test, scores)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "test_size": int(len(y_test)),
        "dataset_size": int(len(dataset)),
        "class_balance": {str(key): int(value) for key, value in target.value_counts().sort_index().items()},
        "cross_validation": {
            "method": "3-fold stratified CV on training split; ROC-AUC selection",
            "best_mean_roc_auc": float(best["score"]),
        },
        "model_comparison": comparisons,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
    }

    transformed_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    estimator = pipeline.named_steps["classifier"]
    explanation_method = None
    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_)
        explanation_method = "tree impurity-based feature importance"
    elif hasattr(estimator, "coef_"):
        importance = np.abs(np.asarray(estimator.coef_[0]))
        explanation_method = "absolute standardized logistic-regression coefficient"
    else:
        importance = np.asarray([])
    global_importance = (
        sorted(
            [
                {"feature": str(name), "importance": float(value)}
                for name, value in zip(transformed_names, importance)
            ],
            key=lambda item: item["importance"],
            reverse=True,
        )[:20]
        if explanation_method
        else []
    )
    metadata = {
        "disease": disease,
        "selected_model": best["name"],
        "trained_at": metrics["trained_at"],
        "feature_names": list(features.columns),
        "features": _feature_metadata(features, categorical),
        "class_labels": class_labels,
        "positive_class": positive_name,
        "transformed_feature_names": [str(name) for name in transformed_names],
        "feature_importance": global_importance,
        "explanation_method": explanation_method,
        "dataset_license": license_note,
        "dataset_size": int(len(features)),
    }
    artifact = {"pipeline": pipeline, "metadata": metadata}
    joblib.dump(artifact, MODEL_DIR / f"{disease}.joblib")
    metrics["dataset_license"] = license_note
    metrics["positive_class"] = positive_name
    (REPORT_DIR / f"{disease}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    for disease in ("heart", "diabetes", "breast-cancer"):
        print(f"Training {disease}...")
        metrics = train_one(disease)
        print(
            f"  {metrics['selected_model']}: ROC-AUC={metrics['roc_auc']:.3f}, "
            f"accuracy={metrics['accuracy']:.3f} ({metrics['test_size']} held-out rows)"
        )


if __name__ == "__main__":
    main()