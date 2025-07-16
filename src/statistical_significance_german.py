# statistical_significance_german.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import joblib
from pathlib import Path
from scipy.stats import ttest_ind

# Paths
data_dir = Path("data")
models_dir = Path("models")
results_dir = Path("evaluation")
results_dir.mkdir(exist_ok=True)

# Load data
X = np.load(data_dir / "X_resampled_german.npy")
y = np.load(data_dir / "y_resampled_german.npy")

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Models to evaluate
model_names = ["rf", "logreg", "xgb", "svm"]
auc_results = {}

for model_name in model_names:
    model_path = models_dir / f"{model_name}_model_german.pkl"

    if not model_path.exists():
        print(f"⚠️ Model file not found: {model_path}")
        continue

    model = joblib.load(model_path)
    fold_aucs = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
        else:
            y_probs = model.decision_function(X_test)

        fold_auc = roc_auc_score(y_test, y_probs)
        fold_aucs.append(fold_auc)

    auc_results[model_name] = fold_aucs
    print(f"{model_name.upper()} ROC-AUCs:", fold_aucs)

# Convert to DataFrame
df_auc = pd.DataFrame(auc_results)
df_auc.to_csv(results_dir / "auc_results_german.csv", index=False)
print(f"✅ Saved AUC results to {results_dir / 'auc_results_german.csv'}")

# T-test between models
pairs_to_compare = [
    ("logreg", "xgb"),
    ("logreg", "rf"),
    ("xgb", "rf"),
    ("xgb", "svm"),
]

t_results = []

for model1, model2 in pairs_to_compare:
    auc1 = df_auc[model1]
    auc2 = df_auc[model2]
    t_stat, p_value = ttest_ind(auc1, auc2)

    t_results.append({
        "Comparison": f"{model1.upper()} vs {model2.upper()}",
        "T-statistic": t_stat,
        "P-value": p_value
    })

# Save t-test results
df_ttest = pd.DataFrame(t_results)
df_ttest.to_csv(results_dir / "t_test_results_german.csv", index=False)
print(f"✅ Saved t-test results to {results_dir / 't_test_results_german.csv'}")

# Optional boxplot
plt.figure(figsize=(8, 5))
df_auc.boxplot()
plt.title("ROC-AUC Scores Across Models (South German Credit Dataset)")
plt.ylabel("ROC-AUC Score")
plt.xticks(rotation=45)
plt.tight_layout()
plot_path = results_dir / "auc_boxplot_german.png"
plt.savefig(plot_path, dpi=300)
plt.show()

print(f"✅ Boxplot saved to {plot_path}")
