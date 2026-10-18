# Model Card

## Intended use

Educational demonstration of independent tabular classification pipelines, model comparison, input validation, and explainability. The software is not intended for clinical decisions, diagnosis, screening, triage, treatment, or patient-specific risk assessment.

## Dataset targets

- Heart: target 0 versus 1-4 from the UCI Cleveland angiographic disease-status field.
- Diabetes: the UCI questionnaire dataset's Positive/Negative class. This is not glucose-test-confirmed diabetes.
- Breast cancer: the Wisconsin Diagnostic Breast Cancer dataset's malignant/benign class, based on cell-nucleus measurements; not symptoms.

## Evaluation method

Each module uses a fixed random seed of 42, stratified 80/20 train/test split, preprocessing in scikit-learn pipelines, and 3-fold stratified ROC-AUC model selection on the training split. Results below are this local run, not guaranteed future results.

| Module | Selected model | Rows | Held-out n | Accuracy | Precision | Recall | F1 | ROC-AUC | Specificity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Heart | Logistic Regression | 303 | 61 | 0.885 | 0.839 | 0.929 | 0.881 | 0.965 | 0.848 |
| Diabetes indicators | Random Forest | 520 | 104 | 0.990 | 0.985 | 1.000 | 0.992 | 0.998 | 0.975 |
| Breast cell classification | Logistic Regression | 569 | 114 | 0.974 | 0.976 | 0.952 | 0.964 | 0.995 | 0.986 |

The strong scores, especially for the small questionnaire set, may reflect dataset structure, sample size, or overfitting and must not be interpreted as clinical accuracy. Validation is a single local split and limited hyperparameter search, without external or temporal validation. Probability-like outputs may not be calibrated; Brier score is reported as a descriptive measure only.

## Explainability

For binary Logistic Regression, the app transforms the submitted row using the fitted preprocessor and reports the estimator class-1 intercept plus signed transformed-feature contributions. A test verifies these sum to `decision_function`. Other supported estimators show global native feature importance with its scope stated; estimators without a supported explanation say so rather than showing fabricated values. Importance is not causation or medical importance.

## Known limitations and prohibited interpretations

- Small historical datasets with restricted populations and possible selection/measurement bias.
- No prospective, external, geographic, subgroup fairness, or clinical validation.
- A dataset-positive class or model score is not an individual diagnosis or calibrated probability of disease.
- Do not use in care, screening, treatment, referral, or patient communication.
- Input ranges describe observed training data, not safe or clinically accepted ranges.