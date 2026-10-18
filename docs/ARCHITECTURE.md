# Architecture

```text
React + TypeScript + Vite (port 5173)
  ├── dynamic forms, analytics, dataset explorer, local preferences
  └── /api proxy
       ↓
FastAPI (port 8000)
  ├── Pydantic request boundary and explicit CORS allowlist
  ├── model_service: schema validation, trusted local inference, metrics/data loading
  ├── explain_service: compatible binary Logistic Regression additive log-odds
  ├── report_service: in-memory ReportLab PDF
  └── SQLAlchemy / SQLite: output-only prediction history
       ↓
scikit-learn training pipelines (one per dataset)
  ├── preprocessing fitted inside CV/training folds
  ├── Logistic Regression / Random Forest / SVM / Gradient Boosting
  └── joblib artifact + JSON test metrics + local CSV export
```

Model artifacts are trusted local outputs from the training script. User uploads are never deserialized. Training compares candidates using 3-fold stratified CV within the training partition, then evaluates the selected estimator once on a held-out stratified 20% partition. The same test partition is not used for model selection.

Run the API and frontend as separate local processes. The development server proxies `/api` to `127.0.0.1:8000`. SQLite persistence can be disabled with `HISTORY_ENABLED=false`; the UI preference also opts out per prediction.