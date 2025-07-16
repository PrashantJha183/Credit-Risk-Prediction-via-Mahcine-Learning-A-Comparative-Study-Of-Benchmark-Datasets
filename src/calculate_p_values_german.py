# calculate_p_values_german.py

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from pathlib import Path

# Create evaluation directory if it doesn't exist
eval_dir = Path("evaluation")
eval_dir.mkdir(exist_ok=True)

# --- YOUR DATA ---
# Replace these numbers if your values change

rf = np.array([0.9380, 0.8904, 0.9402, 0.9426, 0.9258])
logreg = np.array([0.8419, 0.8030, 0.8451, 0.8162, 0.8320])
xgb = np.array([0.9287, 0.8617, 0.9176, 0.9180, 0.9162])
svm = np.array([0.9179, 0.8940, 0.9300, 0.9198, 0.9096])

# Put them into a dictionary for convenience
model_dict = {
    "Random Forest": rf,
    "Logistic Regression": logreg,
    "XGBoost": xgb,
    "SVM": svm
}

# --- CALCULATE PAIRWISE P-VALUES ---

# Create all possible unique pairs
pairs = []
model_names = list(model_dict.keys())

for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        pairs.append((model_names[i], model_names[j]))

# List to store results
results = []

for m1, m2 in pairs:
    auc1 = model_dict[m1]
    auc2 = model_dict[m2]

    # Calculate t-test
    t_stat, p_value = ttest_ind(auc1, auc2)

    # Calculate means and std devs
    mean1 = np.mean(auc1)
    std1 = np.std(auc1, ddof=1)
    mean2 = np.mean(auc2)
    std2 = np.std(auc2, ddof=1)

    result = {
        "Model 1": m1,
        "Model 2": m2,
        "Mean AUC Model 1": round(mean1, 4),
        "Std Dev Model 1": round(std1, 4),
        "Mean AUC Model 2": round(mean2, 4),
        "Std Dev Model 2": round(std2, 4),
        "T-statistic": round(t_stat, 4),
        "P-value": round(p_value, 6)
    }

    results.append(result)

    print(f"\n{m1} vs {m2}")
    print(f"  Mean AUC {m1}: {mean1:.4f} ± {std1:.4f}")
    print(f"  Mean AUC {m2}: {mean2:.4f} ± {std2:.4f}")
    print(f"  T-statistic: {t_stat:.4f}")
    print(f"  P-value: {p_value:.6f}")
    if p_value < 0.05:
        print("  --> Statistically significant difference")
    else:
        print("  --> No statistically significant difference")

# Save results to CSV
df = pd.DataFrame(results)
csv_path = eval_dir / "t_test_results_german.csv"
df.to_csv(csv_path, index=False)

print(f"\n✅ T-test results saved to {csv_path}")
