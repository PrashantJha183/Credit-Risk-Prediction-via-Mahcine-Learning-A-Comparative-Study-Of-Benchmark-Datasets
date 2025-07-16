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


- **data/** → Contains raw datasets and preprocessed numpy files
- **evaluation/** → Stores generated plots, performance metrics, and CSV result files
- **models/** → Saved trained models and feature names
- **notebooks/** → Jupyter notebooks for data exploration, preprocessing, and analysis
- **src/** → Python scripts for data processing, training, evaluation, plotting, and benchmarking


---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/PrashantJha183/Credit-risk-prediction.git
cd Credit-risk-prediction
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

1. **Preprocess data**


```bash
python src/preprocess.py
```

2. **Train Models**


```bash
python src/train_model.py 
```

3. **Evaluate All Models**

```bash
python src/evaluate_all_models_uci.py
python src/evaluate_all_models_german.py
```


---

## 📊 Generating Plots

- Class distributions before and after SMOTE

- Feature importance visualizations

- ROC curves for model comparisons

- Performance summary plots
  

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
