# Credit Risk Prediction via Machine Learning: Comprehensive Analysis and Benchmarking

## Project Overview

This repository contains reproducible code and data supporting the paper
**"Credit Risk Prediction via Machine Learning: A Comparative Study of
Benchmark Datasets."**\
The goal is to compare modern machine learning models for credit risk
assessment using two benchmark datasets (UCI Credit Card Default and
South German Credit).\
The repository enables technical review, code audit, and results
reproduction for editors, reviewers, and domain experts.

## Key Features

- Implements and compares four supervised learning models: **Logistic
  Regression, Random Forest, XGBoost, and Support Vector Machine
  (SVM).**
- Addresses class imbalance with the **Synthetic Minority Oversampling
  Technique (SMOTE).**
- Includes scripts for preprocessing, model training,
  cross-validation, performance evaluation, feature analysis, and
  visualization.
- Stores all results, plots, and models for full reproducibility.

## Repository Structure

---

Folder Description

---

**data/** Contains raw datasets and
preprocessed files (e.g., numpy
formats)

**models/** Trained model objects and feature
name indices

**evaluation/** Performance metrics, plots (ROC
curves, feature importance), CSVs

**notebooks/** Jupyter notebooks for data
exploration and step-by-step
analysis

**src/** Python scripts for workflow
automation (preprocessing,
training, plotting)

---

## Datasets Utilized

- **UCI Credit Card Default Dataset:** 30,000 records, 24 columns,
  including payment history, credit limit, demographic features, and
  binary target variable (default/no-default).
- **South German Credit Dataset:** 1,000 records, 21 columns,
  including sociodemographic and financial predictors, binary
  classification for good/bad credit risk.

Both datasets are public benchmarks widely used in academic and
industrial research for credit scoring model comparison.

## Methodology Highlights

### Preprocessing

- Handles missing values if any (both datasets confirm none).
- Categorical variables encoded using numerical codes or one-hot
  encoding as appropriate.
- All numerical features scaled using **z-score normalization**.

### Model Building

- SMOTE is applied for balancing minority and majority classes.
- Four classifiers (LR, RF, XGBoost, SVM) implemented using
  **scikit-learn** and **XGBoost**.
- **5-Fold stratified cross-validation** used to ensure robust
  estimates.

### Evaluation Metrics

- Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
- Model training times and computational considerations are
  benchmarked.

### Feature Analysis

- Feature importance produced using tree-based models and **SHAP
  values**.
- Comparative performance plots and ROC curves are included.

### Statistical Testing

- Model comparisons validated with significance testing (**independent
  two-tailed t-tests**) on cross-validation results.

## Usage Instructions

### 1. Environment Setup

```bash
git clone https://github.com/PrashantJha183/Credit-Risk-Prediction-via-Mahcine-Learning-A-Comparative-Study-Of-Benchmark-Datasets.git
cd Credit-Risk-Prediction-via-Mahcine-Learning-A-Comparative-Study-Of-Benchmark-Datasets
python -m venv venv
source venv/bin/activate        # For Linux/macOS users
venv\Scripts\activate           # For Windows users
pip install -r requirements.txt
```

### 2. Run Preprocessing and Train Models

```bash
python src/preprocess.py
python src/train_model.py
```

### 3. Model Evaluation

```bash
python src/evaluate_all_models_uci.py
python src/evaluate_all_models_german.py
```

### 4. Visualizations

Plots and summary CSVs are saved in the **evaluation/** folder (e.g.,
ROC curves, feature importance, training time breakdown).

## Reproducibility and Audit

- All datasets, code, and result files required for reproduction are
  provided.
- Full experimental workflow and foldwise cross-validation results
  available.
- Model and metric benchmarks can be compared to published results in
  the related paper and repository documentation.

## Citation

If used for publication, comparison, or audit, cite as:

**Prashant Jha, Credit Risk Prediction using Machine Learning, GitHub
Repository, <https://github.com/PrashantJha183/Credit-Risk-Prediction-via-Mahcine-Learning-A-Comparative-Study-Of-Benchmark-Datasets>**

## License

Repository code is covered under the license included in the project.

## Contact

For technical questions, issues or further collaboration:

**Prashant Jha -- jhaprashant.works@gmail.com**

## Repository Authors

See paper for full list and ORCID IDs.

## References

See in-paper references and repository for full citation list.\
Peer-reviewed source data: **UCI Credit Card Default, South German
Credit.**
