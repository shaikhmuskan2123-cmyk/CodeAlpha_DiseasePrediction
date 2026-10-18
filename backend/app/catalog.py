DISEASES = {
    "heart": {
        "name": "Heart disease",
        "dataset": "UCI Heart Disease (Cleveland subset)",
        "source": "https://doi.org/10.24432/C52P4X",
        "target": "Positive means the dataset angiographic target is 1-4 (>50% narrowing); negative is 0.",
        "population": "A small clinical cohort collected in Cleveland; not representative of general populations.",
    },
    "diabetes": {
        "name": "Early-stage diabetes indicators",
        "dataset": "UCI Early Stage Diabetes Risk Prediction",
        "source": "https://doi.org/10.24432/C5VG8H",
        "target": "The dataset's Positive/Negative questionnaire class, not a laboratory-confirmed diagnosis.",
        "population": "520 questionnaire records from a hospital in Sylhet, Bangladesh.",
    },
    "breast-cancer": {
        "name": "Breast cancer cell-nucleus classification",
        "dataset": "Wisconsin Diagnostic Breast Cancer (scikit-learn)",
        "source": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        "target": "Positive means malignant; negative means benign, using the dataset labels.",
        "population": "569 diagnostic samples; cell-nucleus measurements only, not symptoms or screening data.",
    },
}