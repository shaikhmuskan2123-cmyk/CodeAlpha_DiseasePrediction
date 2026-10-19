<img width="1913" height="925" alt="Home 1" src="https://github.com/user-attachments/assets/7b5b518a-0827-4acf-b4fb-c7c5584a1cdf" /># MediPredict AI

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
  
## Project Screenshots

### 1.Dashboard
![Home]
<img width="1913" height="925" alt="Home 1" src="https://github.com/user-attachments/assets/370aded5-5dd7-4475-9238-3eba1b0883a5" />
<img width="1917" height="913" alt="Home 2" src="https://github.com/user-attachments/assets/f4a346be-6026-49e6-9acf-bec7ea931239" />
<img width="1917" height="921" alt="Home 3" src="https://github.com/user-attachments/assets/1fd1a5bf-31d9-4ef3-889c-e56b35542acc" />
<img width="1917" height="921" alt="Home 4" src="https://github.com/user-attachments/assets/3494699a-5aee-4425-b846-d487f02161f8" />
<img width="1917" height="920" alt="Home 5" src="https://github.com/user-attachments/assets/0718b11d-6627-4681-9087-b4553a59a2bf" />
<img width="1912" height="912" alt="Home 6" src="https://github.com/user-attachments/assets/8b1a3dd2-82fb-4149-a93f-f873ae413f68" />
<img width="1917" height="906" alt="Home 7" src="https://github.com/user-attachments/assets/218a06b5-accd-485e-b916-55454a6aaf06" />
<img width="1913" height="917" alt="Home 8" src="https://github.com/user-attachments/assets/34cb907b-2dc1-4647-9c96-70cabf6bfbd9" />
<img width="1882" height="911" alt="Home dr" src="https://github.com/user-attachments/assets/db7f2557-6c21-4a42-9691-34962b3a9cc4" />

### 2.Downloading Report
![Report]
<img width="502" height="731" alt="Report pdf" src="https://github.com/user-attachments/assets/2f4c25e1-485b-46a5-9ead-bd2c05c03dcb" />

### 3.Health Insights & Models
![Model Evaluation]
<img width="1907" height="912" alt="Health Model 1" src="https://github.com/user-attachments/assets/26cba2a4-3fd5-4c51-b330-4b832f2ba829" />
<img width="1915" height="912" alt="Health Model 2" src="https://github.com/user-attachments/assets/616bcea3-9b37-4b4c-ba9c-c46ea78962b7" />
<img width="1917" height="915" alt="Health Model 3" src="https://github.com/user-attachments/assets/d201f705-48d1-4f4e-8fb5-f6372da0bdbf" />
<img width="1917" height="902" alt="Health Model dr" src="https://github.com/user-attachments/assets/72b5d303-3c52-4a39-89d8-92c1adcf5af5" />

### 4.Dataset Explorer
![Dataset]
<img width="1917" height="913" alt="Data 1" src="https://github.com/user-attachments/assets/229b431a-802d-4f06-a294-4fd58afac8ea" />
<img width="1906" height="917" alt="Data 2" src="https://github.com/user-attachments/assets/a453d98a-99b0-4ba6-888d-9099621f1af4" />
<img width="1917" height="915" alt="Data 3" src="https://github.com/user-attachments/assets/c37806da-4cc7-4c63-901a-ab7d91f3d450" />
<img width="1917" height="912" alt="Data dr" src="https://github.com/user-attachments/assets/d9444fe0-2847-4810-ab71-ce7cfc50f337" />

### 5.History & Records
![Prediction History]
<img width="1917" height="910" alt="History 1" src="https://github.com/user-attachments/assets/fe8da2dd-3a57-4247-a7c9-249554cc595b" />
<img width="1917" height="916" alt="History 2" src="https://github.com/user-attachments/assets/b7f9cbc6-0945-4ab2-ad9c-61f6687154b7" />
<img width="1913" height="903" alt="History 3" src="https://github.com/user-attachments/assets/38cfb746-87c9-4e7a-b30d-27d825ac03cb" />
<img width="1917" height="905" alt="History dr" src="https://github.com/user-attachments/assets/e5e8e4a2-be4c-436e-bbad-4e2c64f38857" />

### 6.Clinical Ethics
![Clinical Information]
<img width="1917" height="910" alt="Clinical" src="https://github.com/user-attachments/assets/a529584e-e7db-4de1-a533-da063f29b302" />
<img width="1917" height="916" alt="Clinical dr" src="https://github.com/user-attachments/assets/4a5ee198-f7fa-4f04-8055-0cd5bdb81471" />

## Screenshot checklist

- Overview dashboard showing the three real model modules.
- Heart prediction form and model output with the log-odds explanation.
- Model analytics showing the actual held-out confusion matrix and CV comparison.
- Filtered dataset explorer with CSV export control.
- Prediction history/settings showing opt-out and deletion controls.
- Repeat at desktop and narrow mobile widths. Do not use screenshots that imply clinical approval.

See `docs/PROJECT_AUDIT.md`, `docs/ARCHITECTURE.md`, `docs/DATA_DICTIONARY.md`, `docs/PRIVACY.md`, and `docs/TEST_REPORT.md` for the audit and verification details. No clinical validation is claimed.
