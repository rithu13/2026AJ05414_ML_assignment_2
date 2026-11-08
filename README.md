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

**Repository:** https://github.com/rithu13/2026AJ05414_ML_assignment_2

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

**Logistic Regression**
- Best overall model with 98.25% accuracy and highest AUC of 0.9954
- Works well because the features become linearly separable after scaling
- High recall (98.61%) means it rarely misses malignant cases, which is important for medical diagnosis
- Also gives probability scores, so we can adjust the threshold if needed

**Decision Tree**
- Lowest accuracy (90.35%) among all models tested
- Limited the depth to 5 to prevent overfitting, but this made it too simple for the data
- The upside is that we can actually see the rules it uses to make decisions
- Would not use this for actual diagnosis due to lower recall

**KNN (K-Nearest Neighbors)**
- Performed well with 97.22% recall, second only to Logistic Regression and SVM
- Makes predictions by looking at the 5 nearest training samples
- Needs scaled features since it uses distance calculations
- Slower at prediction time compared to other models

**Naive Bayes (Gaussian)**
- Decent AUC (0.9868) even though its assumption of independent features is not true here
- Many features like radius, perimeter, and area are clearly correlated
- Lower precision means more false alarms, but for cancer screening that might be acceptable
- Very fast to train and predict

**Support Vector Machine**
- Tied with Logistic Regression at 98.25% accuracy
- Used RBF kernel to handle non-linear patterns in the data
- Slightly lower AUC (0.9950) compared to Logistic Regression
- Harder to interpret since we cannot easily see why it made a decision

**Overall Winner: Logistic Regression**
- Matches SVM on most metrics but has the highest AUC
- Easy to understand which features matter most by looking at the coefficients
- Fast predictions and simple to deploy
- Best choice when we need to explain the model's decisions

## Live Streamlit App

**Application:** https://2026aj05414.streamlit.app/

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

## Author

**Name:** Rithika A
**BITS ID:** 2026AJ05414  

## References

1. Wolberg, W., Mangasarian, O., Street, N., and Street, W. (1995). Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B
2. Scikit-learn documentation: https://scikit-learn.org/
3. Streamlit documentation: https://docs.streamlit.io/