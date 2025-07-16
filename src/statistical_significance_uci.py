# statistical_significance_uci.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import joblib
from pathlib import Path
from scipy.stats import ttest_ind
import warnings

# Suppress XGBoost and convergence warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Define paths
data_dir = Path("data")
models_dir = Path("models")
results_dir = Path("evaluation")
results_dir.mkdir(exist_ok=True)

# Load preprocessed UCI dataset
X = np.load(data_dir / "X_resampled_uci.npy")
y = np.load(data_dir / "y_resampled_uci.npy")

# Models to evaluate
model_names = ["rf", "logreg", "xgb", "svm"]
auc_results = {}

# Stratified 5-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name in model_names:
    model_path = models_dir / f"{model_name}_model_uci.pkl"

    if not model_path.exists():
        print(f"⚠️ Model file not found: {model_path}")
        continue

    model = joblib.load(model_path)
    fold_aucs = []

    print(f"\n🔄 Running CV for model: {model_name.upper()}")

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Optimize SVM if applicable
        if model_name == "svm":
            if hasattr(model, "set_params"):
                model.set_params(max_iter=1000, cache_size=1000)

        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
        else:
            y_probs = model.decision_function(X_test)

        auc = roc_auc_score(y_test, y_probs)
        fold_aucs.append(auc)

        print(f"  Fold {fold}: AUC = {auc:.4f}")

    auc_results[model_name] = fold_aucs
    print(f"✅ {model_name.upper()} AUCs: {np.round(fold_aucs, 4).tolist()}")

# Save AUC results
df_auc = pd.DataFrame(auc_results)
auc_csv_path = results_dir / "auc_results_uci.csv"
df_auc.to_csv(auc_csv_path, index=False)
print(f"\n📁 Saved AUC results to {auc_csv_path}")

# Generate pairwise t-tests
pairs = []
for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        pairs.append((model_names[i], model_names[j]))

t_results = []

for m1, m2 in pairs:
    auc1 = df_auc[m1]
    auc2 = df_auc[m2]

    t_stat, p_value = ttest_ind(auc1, auc2)

    mean1 = np.mean(auc1)
    std1 = np.std(auc1, ddof=1)
    mean2 = np.mean(auc2)
    std2 = np.std(auc2, ddof=1)

    t_results.append({
        "Model 1": m1.upper(),
        "Model 2": m2.upper(),
        "Mean AUC Model 1": round(mean1, 4),
        "Std Dev Model 1": round(std1, 4),
        "Mean AUC Model 2": round(mean2, 4),
        "Std Dev Model 2": round(std2, 4),
        "T-statistic": round(t_stat, 4),
        "P-value": round(p_value, 6)
    })

    print(f"\n📊 {m1.upper()} vs {m2.upper()}")
    print(f"  Mean AUC {m1.upper()}: {mean1:.4f} ± {std1:.4f}")
    print(f"  Mean AUC {m2.upper()}: {mean2:.4f} ± {std2:.4f}")
    print(f"  T-statistic: {t_stat:.4f}")
    print(f"  P-value: {p_value:.6f}")
    print("  ➤ Statistically significant" if p_value < 0.05 else "  ➤ Not significant")

# Save t-test results
df_ttest = pd.DataFrame(t_results)
ttest_csv_path = results_dir / "t_test_results_uci.csv"
df_ttest.to_csv(ttest_csv_path, index=False)
print(f"\n📁 T-test results saved to {ttest_csv_path}")

# Generate boxplot
plt.figure(figsize=(8, 5))
df_auc.boxplot()
plt.title("ROC-AUC Scores Across Models (UCI Dataset)")
plt.ylabel("ROC-AUC Score")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
boxplot_path = results_dir / "auc_boxplot_uci.png"
plt.savefig(boxplot_path, dpi=300)
plt.show()

print(f"\n✅ Boxplot saved to {boxplot_path}")
