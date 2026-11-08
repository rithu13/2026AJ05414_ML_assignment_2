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
- Achieved top performance: 98.25% accuracy, 0.9954 AUC (highest), 0.9623 MCC
- The 30 cell-nucleus features are nearly linearly separable after standardization
- High recall (98.61%) is critical for cancer screening—missing a malignant case is costly
- Probabilistic outputs enable clinicians to adjust decision thresholds based on risk tolerance
- Interpretable coefficients reveal which features (e.g., "worst concave points") drive predictions

**Decision Tree**
- Lowest performer: 90.35% accuracy, 0.7969 MCC due to depth-limited rules (max_depth=5)
- Oversimplifies complex feature interactions present in the tumor measurements
- Offers full interpretability—each prediction traced through human-readable if-then rules
- The 9.72% missed recall rate makes it unsuitable for primary diagnosis
- Best used for explainability and generating initial insights about feature importance

**KNN (K-Nearest Neighbors)**
- Second-best performer with highest recall (97.22%) among non-linear models
- Misses fewer malignant cases due to instance-based local pattern matching
- Performance sensitive to k=5 choice; requires standardized features for Euclidean distance
- No explicit model—predictions based on similarity to training instances
- Computationally expensive at prediction time for large datasets

**Naive Bayes (Gaussian)**
- Strong AUC (0.9868) despite violated independence assumption (radius, perimeter, area are correlated)
- Lower precision (94.44%) results in more false positives
- In cancer screening, accepting false positives may be preferable to missing true positives
- Extremely fast training and prediction—suitable for quick baseline comparisons
- Probabilistic framework provides calibrated confidence scores

**Support Vector Machine**
- Matched Logistic Regression: 98.25% accuracy, 0.9623 MCC using RBF kernel
- Slightly lower AUC (0.9950) suggests marginally less confident probability estimates
- RBF kernel effectively captures non-linear decision boundaries
- Higher computational cost during both training and prediction
- Lacks interpretability—operates as a "black box" classifier

**Overall Winner: Logistic Regression**
- Ties SVM on threshold metrics but achieves best AUC (0.9954)
- Provides interpretable coefficients and calibrated probabilities
- Efficient inference with minimal computational overhead
- For clinical decision support, transparency outweighs marginal gains from complex models
- Recommended for deployment where explainability and regulatory compliance matter

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