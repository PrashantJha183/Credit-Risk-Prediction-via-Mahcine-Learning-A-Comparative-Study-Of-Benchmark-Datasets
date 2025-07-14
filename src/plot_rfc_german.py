import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
eval_dir = Path("evaluation")
dataset = "german"

# List of models
model_names = ["rf", "logreg", "xgb", "svm"]

# Colors for plotting
colors = {
    "rf": "blue",
    "logreg": "darkorange",
    "xgb": "green",
    "svm": "purple"
}

plt.figure(figsize=(7, 6))

# Loop through models and plot their curves
for model_name in model_names:
    roc_csv_path = eval_dir / f"roc_curve_{dataset}_{model_name}.csv"
    if roc_csv_path.exists():
        roc_data = pd.read_csv(roc_csv_path)
        fpr = roc_data["fpr"]
        tpr = roc_data["tpr"]
        
        # Compute AUC from loaded curve
        auc_value = np.trapz(tpr, fpr)
        
        plt.plot(
            fpr, tpr,
            label=f"{model_name.upper()} (AUC = {auc_value:.3f})",
            color=colors.get(model_name, None),
            lw=2
        )
    else:
        print(f"⚠️ ROC curve file not found for {model_name.upper()}")

# Diagonal line
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curves - South German Credit Dataset")
plt.legend(loc="lower right")
plt.tight_layout()

# Save figure
roc_plot_path = eval_dir / f"roc_curve_combined_{dataset}.png"
plt.savefig(roc_plot_path, dpi=300)
plt.show()

print(f"✅ Combined ROC curve saved to {roc_plot_path}")
