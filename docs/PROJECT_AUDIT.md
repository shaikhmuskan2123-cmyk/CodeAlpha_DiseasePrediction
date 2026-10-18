# Project Audit

## Initial state

- Inspected workspace: `C:\Users\HP\Downloads\CodeAlpha_DiseasePredicition` (the open VS Code folder).
- The directory was empty before implementation; no app, dataset, model, report, test, backup, or virtual environment was present to preserve.
- The alternate requested path `C:\Users\Sumiya\Downloads\CodeAlpha_DiseasePrediction` did not exist in this environment. No files were copied from or written to that path.
- Python and Node were initially unavailable on PATH. Python 3.10.11 was installed through the Python Install Manager and isolated in `.venv`. Node.js v24.20.0 was downloaded from the official distribution, SHA-256 checked against the official checksums, and unpacked under ignored `.tools/`.
- A pre-edit backup was unnecessary because the workspace contained no user project files. Generated models, datasets, reports, virtual environments, and runtime binaries are excluded from Git.

## Data source verification

- Heart: UCI Heart Disease, DOI `10.24432/C52P4X`, CC BY 4.0. The fetched Cleveland subset has 303 rows and 13 predictor features. The UCI angiographic target is 0-4; this build maps 0 to 0 and 1-4 to 1.
- Diabetes: UCI Early Stage Diabetes Risk Prediction, DOI `10.24432/C5VG8H`, CC BY 4.0. It contains 520 questionnaire records from Sylhet, Bangladesh. It is not the unrelated UCI ID 34 time-series dataset and is not a laboratory-confirmed diagnosis.
- Breast cancer: Wisconsin Diagnostic Breast Cancer distributed via scikit-learn, based on UCI ID 17. It contains 569 samples, 30 real-valued cell-nucleus features, and dataset classes benign/malignant.

## Implementation record

FastAPI serves trusted local joblib pipelines, real metrics, dataset summaries/records, prediction history, and in-memory PDF reports. React/Vite provides the dashboard, dynamic per-dataset input forms, analytics, dataset filtering/export, history, reports, and settings. Training is explicit (`python -m ml.training.train`); the application does not silently retrain or substitute models.

## Verification

- Three dataset/model/evaluation artifacts were generated locally from the cited sources.
- Backend tests: 17 passed, 1 dependency deprecation warning.
- Frontend: strict TypeScript check and Vite production build passed.
- Browser: dashboard, all three real prediction forms (13/16/30 fields), analytics, dataset filtering/export, documented heart category labels, and desktop/mobile layout were exercised. The API returned a valid PDF response; the embedded browser harness did not expose a download event for the client-side blob.
- Initial Tailwind 3 dependencies reported 7 advisories. After migrating to Vite-compatible Tailwind 4.1.11, `npm install` reported 0 vulnerabilities.

This is not clinical validation, a security audit, or regulatory review.