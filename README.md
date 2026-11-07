# Breast Cancer Diagnostic Classification

## A. Problem Statement

Breast cancer diagnosis benefits from models that can distinguish malignant from benign tumors using measurements derived from digitized fine-needle aspirate images. This project trains and compares five machine learning classifiers on the same dataset, evaluates them with six required metrics, and provides an interactive Streamlit application that scores uploaded labeled test data without retraining.

This application is an educational comparison tool and is not intended for clinical use.

## B. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Public source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)  
**Dataset access in code:** scikit-learn's unmodified `load_breast_cancer` copy of the UCI dataset

| Property | Value |
|---|---:|
| Instances | 569 |
| Input features | 30 numeric features |
| Target | Binary classification |
| Class 0 | Malignant |
| Class 1 | Benign |
| Missing values | 0 |
| Training rows | 455 (80%) |
| Test rows | 114 (20%) |
| Split method | Stratified, `random_state=42` |

The 30 input variables describe the mean, standard error, and worst value of ten cell-nucleus measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension. This satisfies the assignment minimum of 12 features and 500 instances.

## C. GitHub Repository Link

**Repository:** [ADD_GITHUB_REPOSITORY_LINK]

## D. Models Used

All models use the same stratified hold-out test set. Logistic Regression, KNN, Gaussian Naive Bayes, and SVM include feature standardization inside their saved pipelines. Decision Tree operates on the original feature scale. The models are fitted only on the training partition.

### Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9035 | 0.9373 | 0.9420 | 0.9028 | 0.9220 | 0.7969 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Performance Observations

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Joint-best accuracy, precision, recall, F1, and MCC, with the highest AUC. The strong result indicates that standardized features are close to linearly separable in the transformed feature space. |
| Decision Tree | Lowest score on every reported metric. Its simple depth-limited rules remain interpretable, but they discard information that the smoother decision boundaries use. |
| KNN | Strong second tier, with 0.9722 recall and 0.9655 F1. Scaling is essential because predictions depend directly on feature distances. |
| Naive Bayes | Good AUC despite lower thresholded metrics. Its conditional-independence assumption is restrictive because many tumor measurements are correlated. |
| Support Vector Machine | Ties Logistic Regression on accuracy, precision, recall, F1, and MCC, while producing a slightly lower AUC. The RBF boundary models non-linearity effectively but is less interpretable. |
| **Overall winner** | **Logistic Regression.** It ties SVM on the main classification metrics, has the highest AUC (0.9954), and offers simpler interpretation and lower prediction complexity. |

## Live Streamlit App

**Application:** [ADD_STREAMLIT_APP_LINK]

The application provides:

- CSV upload for labeled test data
- Dropdown selection across all five classifiers
- Accuracy, AUC, precision, recall, F1, and MCC
- Confusion matrix and classification report
- Comparison of all models on the uploaded data
- Downloadable correctly formatted sample test data

## Repository Structure

```text
.
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
└── models/
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── logistic_regression.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── metadata.json
```

## Reproduce the Results

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

Open `http://localhost:8501`, upload `test_data.csv`, and choose a model. Running `train_models.py` recreates every saved model, metric, and the test file from the fixed stratified split.

## Deployment

1. Push this folder to a public GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://streamlit.io/cloud) with GitHub.
3. Create an app from the repository's main branch.
4. Set the entry point to `app.py`.
5. Deploy and replace both link placeholders in this README.

## Author

**Name:** [ADD_NAME]  
**BITS ID:** 2026AJ05414  
**Course:** M.Tech (AIML) - Machine Learning

## References

1. Wolberg, W., Mangasarian, O., Street, N., and Street, W. (1995). Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B
2. Scikit-learn documentation: https://scikit-learn.org/
3. Streamlit documentation: https://docs.streamlit.io/
