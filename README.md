# Credit Risk Prediction using Machine Learning

This repository contains Python scripts, Jupyter notebooks, and data for a comprehensive machine learning study on credit risk prediction. The work evaluates various ML models on the **UCI Credit Card Default dataset** and the **South German Credit dataset**. It includes data preprocessing, feature analysis, class balancing with SMOTE, model training, evaluation metrics, and visualization of results.

---

## 🔗 Project Overview

- Analyze credit risk using modern ML models
- Compare performance of Logistic Regression, Random Forest, XGBoost, and SVM
- Address class imbalance via SMOTE
- Visualize:
  - Feature importances
  - Class distributions
  - ROC curves
- Benchmark model training times

This work contributes to improving credit risk scoring and aligns with regulatory and ethical considerations for AI in finance.

---

## 📁 Project Structure

Your project folders and key files:

```
data/
├── default_of_credit_card_clients.xls
├── south_german_credit.csv
├── X_resampled_uci.npy
├── y_resampled_uci.npy
├── X_resampled_german.npy
├── y_resampled_german.npy

evaluation/
├── class_dist_before_smote_uci.png
├── class_dist_after_smote_uci.png
├── class_dist_before_smote_german.png
├── class_dist_after_smote_german.png
├── feature_importance_uci.png
├── feature_importance_german.png
├── roc_curve_uci.png
├── roc_curve_german.png
├── results_plot_uci.png
├── results_plot_german.png
├── results_table_uci.csv
├── results_table_german.csv
├── training_times_table.csv

models/
├── rf_model_uci.pkl
├── rf_model_german.pkl
├── feature_names_uci.pkl
├── feature_names_german.pkl

notebooks/
├── 01_data_exploration.ipynb
├── 02_preprocessing.ipynb
├── 03_model_training.ipynb
├── 04_evaluation.ipynb
├── 05_class_distribution_german.ipynb

src/
├── train_model.py
├── evaluate.py
├── evaluate_all_models_uci.py
├── evaluate_all_models_german.py
├── plot_rfc_uci.py
├── plot_rfc_german.py
├── plot_uci_class_distribution.py
├── features_importance_uci.py
├── features_importance_german.py
├── data_utils.py
├── convert_german_to_csv.py
├── benchmark_train_times.py
├── preprocess.py
```

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

2. **Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate      # On Linux/macOS
venv\Scripts\activate         # On Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Train and evaluate a model

Example: Train a Random Forest on the UCI dataset:

```bash
python src/train_model.py --dataset uci --model rf
```

Train XGBoost on the German dataset:

```bash
python src/train_model.py --dataset german --model xgb
```

See more arguments in:

```bash
python src/train_model.py --help
```

---

## 📊 Generating Plots

- Class distributions:
  - `src/plot_uci_class_distribution.py`
  - `notebooks/05_class_distribution_german.ipynb`

- Feature importance:
  - `src/features_importance_uci.py`
  - `src/features_importance_german.py`

- ROC curves:
  - Plots saved in `evaluation/`

- Training times benchmark:
  - `src/benchmark_train_times.py`

---

## 📄 Results

**Tables and plots** summarizing model performance are saved in the `evaluation/` directory:

- ROC curves for UCI and German datasets
- Feature importance plots
- Class balance before and after SMOTE
- Training time comparisons
- CSV files with detailed results

---

## 📝 Citation

If you use this code or results, please cite the repository:

```
Prashant Jha, Credit Risk Prediction using Machine Learning, GitHub Repository, https://github.com/PrashantJha183/Credit-risk-prediction
```

---

## 📊 Data Sources

Data used in this project comes from publicly available benchmark datasets:

- [UCI Credit Card Default Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- [South German Credit Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+german+credit+data)

---

## 🔗 Related Work

See references in the paper for further reading on credit scoring and machine learning.

---
