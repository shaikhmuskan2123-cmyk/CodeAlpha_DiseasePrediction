# Data Dictionary

## Heart: UCI Cleveland

UCI DOI `10.24432/C52P4X`, CC BY 4.0. Predictor order in the downloaded source: `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`. Target `num` is 0-4; this application treats 0 as negative and 1-4 as positive, consistent with the UCI documentation's common binary task.

Documented categorical values: `sex` 0 female/1 male; `cp` 1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic; `fbs` 1 means fasting blood sugar >120 mg/dL; `restecg` 0 normal, 1 ST-T abnormality, 2 probable/definite LVH; `exang` 1 yes; `slope` 1 upsloping, 2 flat, 3 downsloping; `ca` 0-3 vessels colored by fluoroscopy; `thal` 3 normal, 6 fixed defect, 7 reversible defect. Numeric measurements and units follow UCI's variable descriptions; this is not a clinical input standard.

## Diabetes: UCI Early Stage Diabetes Risk Prediction

UCI DOI `10.24432/C5VG8H`, CC BY 4.0. Features: `age`, `gender`, `polyuria`, `polydipsia`, `sudden_weight_loss`, `weakness`, `polyphagia`, `genital_thrush`, `visual_blurring`, `itching`, `irritability`, `delayed_healing`, `partial_paresis`, `muscle_stiffness`, `alopecia`, `obesity`. The 15 categorical questionnaire variables use source-provided encoded categories; `age` is numeric. Target is source `Positive`/`Negative`. Records came from a hospital in Sylhet, Bangladesh and reflect questionnaire indicators, not a lab-confirmed outcome.

## Breast cancer: Wisconsin Diagnostic

The scikit-learn copy of UCI ID 17 contains 569 samples and 30 numeric, positive cell-nucleus measurements. Ten measurement families (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension) are each represented as mean, standard error, and worst value. Target labels are benign and malignant; the positive class in this project is malignant. No symptom questionnaire is used.

Preprocessing imputes missing numeric data with training-fold medians and categorical data with training-fold modes; numerical features are standardized and categories one-hot encoded. These transforms are fitted inside cross-validation and final pipelines to avoid leakage. Raw dataset exports are created locally and ignored by Git.