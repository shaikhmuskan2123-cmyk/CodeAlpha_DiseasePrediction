# MediPredict AI

MediPredict AI is an educational, local-first multi-disease machine-learning demonstration built for the CodeAlpha Machine Learning Internship, Task 4. It is not a medical device, clinical decision-support system, or diagnostic service.

## Modules and datasets

- Heart: UCI Heart Disease, using the Cleveland subset. The original target is 0-4; this project maps 0 to dataset-negative and 1-4 to dataset-positive. UCI license: CC BY 4.0, DOI [10.24432/C52P4X](https://doi.org/10.24432/C52P4X).
- Diabetes: UCI Early Stage Diabetes Risk Prediction, 520 questionnaire records from Sylhet, Bangladesh. Output is the source dataset's Positive/Negative class, not a lab-confirmed diagnosis. UCI license: CC BY 4.0, DOI [10.24432/C5VG8H](https://doi.org/10.24432/C5VG8H).
- Breast cancer: Wisconsin Diagnostic Breast Cancer, 569 samples and 30 nucleus measurements, distributed with scikit-learn. Positive is the dataset's malignant class; it is not a symptom questionnaire. Source: [UCI](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).

Datasets are fetched from UCI or scikit-learn when training is explicitly run. Feature imputation, scaling, and category encoding are fitted inside scikit-learn pipelines on the training folds only. Four classifier families are compared with 3-fold stratified cross-validation on the training partition; final reported metrics use a stratified held-out 20% test partition. These small datasets make estimates uncertain and do not establish generalization to current clinical populations.

## Local setup (Windows PowerShell)

The project was created in the currently open workspace `C:\Users\HP\Downloads\CodeAlpha_DiseasePredicition`. Open that folder in VS Code. Python 3.10.11 is installed and `.venv` is configured for this workspace.

```powershell
cd "C:\Users\HP\Downloads\CodeAlpha_DiseasePredicition"
if (-not (Test-Path .venv)) { py -3.10 -m venv .venv }
& .\.venv\Scripts\Activate.ps1
python --version
python -m pip install -r requirements.txt
python -m ml.training.train
python -m pytest -q
python -m uvicorn backend.app.main:app --reload
```

The API is at `http://127.0.0.1:8000`; interactive API documentation is at `http://127.0.0.1:8000/docs`. Stop it with `Ctrl+C`.

The React frontend is in `frontend/`. On a fresh clone, install Node.js LTS from [nodejs.org](https://nodejs.org/) or Windows Package Manager (`winget install --id OpenJS.NodeJS.LTS`), then open a new PowerShell terminal and run:

```powershell
cd frontend
npm install
npm run dev
```

The Vite development URL is `http://localhost:5173`. The backend currently starts independently; the frontend proxies `/api` requests to port 8000.

## Privacy and intended use

Prediction history stores only module, output label, positive-class score, and timestamp in a local SQLite file. It does not store submitted feature values. PDF reports are generated in memory and include the submitted values only in the downloaded report. This local database is not a secure medical record system. Do not submit real patient data.

Model scores may not be calibrated probabilities. Coefficients or feature importances are model explanations, not causation or clinical importance. Correlated variables can affect interpretation. Never use this output to diagnose, screen, treat, or make care decisions.

## Project layout

```text
backend/app/       FastAPI, data validation, local history, PDF reports
backend/tests/     API smoke tests
frontend/          React + TypeScript web client
ml/training/       Reproducible dataset fetching and model comparison
ml/models/         Locally generated trusted model artifacts (ignored by Git)
ml/reports/        Locally generated evaluation reports (ignored by Git)
docs/              Architecture, dataset notes, and verification record
```

## Recorded local evaluation

These are the actual saved test-split metrics from the run recorded in `ml/reports/`; rerunning training can produce different values.

| Dataset module | Selected model | Accuracy | Precision | Recall | F1 | ROC-AUC | Test rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Heart | Logistic Regression | 0.885 | 0.839 | 0.929 | 0.881 | 0.965 | 61 / 303 |
| Diabetes indicators | Random Forest | 0.990 | 0.985 | 1.000 | 0.992 | 0.998 | 104 / 520 |
| Breast cell classification | Logistic Regression | 0.974 | 0.976 | 0.952 | 0.964 | 0.995 | 114 / 569 |

These small-dataset estimates are not clinical performance claims. The diabetes module learns a Bangladesh hospital questionnaire dataset, and breast cancer uses cell-nucleus measurements, not symptoms. See [the model card](docs/MODEL_CARD.md) for methodology and limitations.

## Verification

- Backend: 17 pytest tests passed; one dependency deprecation warning remains.
- Frontend: strict TypeScript and Vite production build passed.
- Browser smoke checks: dashboard, inference, metrics, dataset filtering, and narrow viewport were exercised.
- PDF endpoint returned a valid PDF response; automated browser download-event capture was unavailable in the integrated harness.
- `npm install` reports 0 vulnerabilities after the Tailwind 4 migration.

## Screenshot checklist

- Overview dashboard showing the three real model modules.
- Heart prediction form and model output with the log-odds explanation.
- Model analytics showing the actual held-out confusion matrix and CV comparison.
- Filtered dataset explorer with CSV export control.
- Prediction history/settings showing opt-out and deletion controls.
- Repeat at desktop and narrow mobile widths. Do not use screenshots that imply clinical approval.

See `docs/PROJECT_AUDIT.md`, `docs/ARCHITECTURE.md`, `docs/DATA_DICTIONARY.md`, `docs/PRIVACY.md`, and `docs/TEST_REPORT.md` for the audit and verification details. No clinical validation is claimed.