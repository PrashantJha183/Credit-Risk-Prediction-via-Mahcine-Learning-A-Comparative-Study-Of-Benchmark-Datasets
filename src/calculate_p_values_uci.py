# calculate_p_values_uci.py

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Paths
eval_dir = Path("evaluation")
eval_dir.mkdir(exist_ok=True)

# Load AUC results
auc_csv_path = eval_dir / "auc_results_uci.csv"

if not auc_csv_path.exists():
    raise FileNotFoundError(f"AUC results file not found: {auc_csv_path}")

df_auc = pd.read_csv(auc_csv_path)
model_names = df_auc.columns.tolist()

print("✅ Loaded AUC data:")
print(df_auc)

# Prepare pairs for t-tests
pairs = []
for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        pairs.append((model_names[i], model_names[j]))

results = []

for m1, m2 in pairs:
    auc1 = df_auc[m1]
    auc2 = df_auc[m2]

    t_stat, p_value = ttest_ind(auc1, auc2)

    mean1 = np.mean(auc1)
    std1 = np.std(auc1, ddof=1)
    mean2 = np.mean(auc2)
    std2 = np.std(auc2, ddof=1)

    significant = "Yes" if p_value < 0.05 else "No"

    results.append({
        "Model 1": m1.upper(),
        "Model 2": m2.upper(),
        "Mean AUC Model 1": round(mean1, 4),
        "Std Dev Model 1": round(std1, 4),
        "Mean AUC Model 2": round(mean2, 4),
        "Std Dev Model 2": round(std2, 4),
        "T-statistic": round(t_stat, 4),
        "P-value": format(p_value, ".6f"),
        "Significant": significant
    })

    print(f"\n📊 {m1.upper()} vs {m2.upper()}")
    print(f"  Mean AUC {m1.upper()}: {mean1:.4f} ± {std1:.4f}")
    print(f"  Mean AUC {m2.upper()}: {mean2:.4f} ± {std2:.4f}")
    print(f"  T-statistic: {t_stat:.4f}")
    print(f"  P-value: {p_value:.6f}")
    print(f"  ➤ Significant Difference: {significant}")

# Convert results to DataFrame
df_ttest = pd.DataFrame(results)

# Enforce column order identical to German dataset
df_ttest = df_ttest[
    [
        "Model 1",
        "Model 2",
        "Mean AUC Model 1",
        "Std Dev Model 1",
        "Mean AUC Model 2",
        "Std Dev Model 2",
        "T-statistic",
        "P-value",
        "Significant"
    ]
]

# Save t-test results
ttest_csv_path = eval_dir / "t_test_results_uci.csv"
df_ttest.to_csv(ttest_csv_path, index=False)
print(f"\n✅ T-test results saved to {ttest_csv_path}")

# Optional boxplot
plt.figure(figsize=(8, 5))
df_auc.boxplot()
plt.title("ROC-AUC Scores Across Models (UCI Dataset)")
plt.ylabel("ROC-AUC Score")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
boxplot_path = eval_dir / "auc_boxplot_uci.png"
plt.savefig(boxplot_path, dpi=300)
plt.show()

print(f"✅ Boxplot saved to {boxplot_path}")
