# Credit Risk Prediction Using Machine Learning

This repository implements experiments from the paper:

> **Credit Risk Prediction Using Machine Learning: A Comparative Study on Benchmark Datasets**

## 🔍 Project Description

We compare machine learning models for credit risk prediction using:

- **UCI Credit Card Default Dataset** (Taiwan credit clients)
- **South German Credit Dataset** (socio-demographic and financial data)

We evaluate:

- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machine

Metrics used:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## 💾 Project Structure

```
credit-risk-prediction/
├── data/
│   ├── default_of_credit_card_clients.xls
│   ├── south_german_credit.csv
│   ├── X_resampled_german.npy
│   └── y_resampled_german.npy
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── data_utils.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── convert_german_to_csv.py
├── evaluation/
│   ├── results_uci.csv
│   ├── results_german.csv
│   └── results_plot.png
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

Install dependencies:


1. **Clone the repo**

```bash
git clone https://github.com/PrashantJha183/Credit-risk-prediction.git
cd Credit-risk-prediction
```

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📊 Run Analysis

### Convert German Data (if using raw file)

```bash
python src/convert_german_to_csv.py
```

### Run Preprocessing

```bash
python src/preprocess.py --dataset uci
python src/preprocess.py --dataset german
```

### Train Models

```bash
python src/train_model.py --dataset uci --model xgb
python src/train_model.py --dataset german --model rf
```

### Evaluate Results

```bash
python src/evaluate.py
```

---

## 🗂 Results

Evaluation results stored in:

- `evaluation/results_uci.csv`
- `evaluation/results_german.csv`
- `evaluation/results_plot.png`

---

## 📚 Citation

If using this repository, please cite:

```
[22] I.-C. Yeh and C.-h. Lien, “The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients,” Expert Systems with Applications, vol. 36, no. 2, pp. 2473–2480, Mar. 2009.

[23] J. H. Park and H. Y. Kim, “Comparative Study of Machine Learning Classifiers for Credit Scoring,” IEEE Access, vol. 10, pp. 11496-11507, 2022.

[24] D. Zhao et al., “Credit Card Default Prediction Based on Improved XGBoost and SHAP Explainability,” IEEE Access, vol. 10, pp. 43562-43572, 2022.

[25] K. Das and K. Deb, “Hybrid Machine Learning Models for Credit Risk Prediction,” IEEE Access, vol. 10, pp. 96923–96937, 2022.

[26] Y. Zhang et al., “A Novel Credit Scoring Model Based on Ensemble Learning and Cost-Sensitive Learning,” IEEE Access, vol. 10, pp. 8375–8388, 2022.

[27] N. A. Jaffar et al., “Credit Default Prediction using Machine Learning and Imbalanced Data Techniques,” IEEE Access, vol. 11, pp. 14387-14401, 2023.

[28] H. Kim et al., “Interpretable Credit Scoring Using Machine Learning: A SHAP-Based Approach,” IEEE Transactions on Knowledge and Data Engineering, Early Access, 2023.
```

---
## 📊 Data Sources

Data used in this project comes from publicly available benchmark datasets:

- [UCI Credit Card Default Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- [South German Credit Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))

---
## 🔗 Related Work

See references in the paper for further reading on credit scoring and machine learning.

---


