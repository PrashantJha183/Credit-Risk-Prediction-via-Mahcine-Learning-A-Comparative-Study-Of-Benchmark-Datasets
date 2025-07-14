import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

# --------------------------------------------
# List of feature names for UCI dataset
# --------------------------------------------
features = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6"
]

# --------------------------------------------
# Models to plot
# --------------------------------------------
model_files = {
    "Random Forest": "./models/rf_model_uci.pkl",
    "Logistic Regression": "./models/logreg_model_uci.pkl",
    "XGBoost": "./models/xgb_model_uci.pkl"
}

# Dictionary to hold importances for each model
importances_dict = {}

# --------------------------------------------
# Load models and extract importances
# --------------------------------------------
for model_name, path in model_files.items():
    if not os.path.exists(path):
        print(f"❌ Model not found: {path}. Skipping {model_name}.")
        continue

    print(f"✅ Loading model: {model_name}")
    model = joblib.load(path)

    # Get feature importances
    if model_name == "Logistic Regression":
        importances = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        print(f"⚠️ No feature importances for model: {model_name}")
        continue

    importances_dict[model_name] = importances

# --------------------------------------------
# Sort features by Random Forest importances
# --------------------------------------------
# Choose RF as reference for sorting
reference_model = "Random Forest"
if reference_model not in importances_dict:
    reference_model = list(importances_dict.keys())[0]
    print(f"ℹ️ Using {reference_model} for sorting instead.")

ref_importances = importances_dict[reference_model]
indices = np.argsort(ref_importances)[::-1]

# Top N features
N = 10
top_indices = indices[:N]
top_features = [features[i] for i in top_indices]

# --------------------------------------------
# Create bar plot
# --------------------------------------------
x = np.arange(len(top_features))  # label positions
bar_width = 0.25

plt.figure(figsize=(12,6))
colors = {
    "Random Forest": "skyblue",
    "Logistic Regression": "salmon",
    "XGBoost": "limegreen"
}

for i, (model_name, importances) in enumerate(importances_dict.items()):
    importances_top = importances[top_indices]
    plt.bar(
        x + i * bar_width,
        importances_top,
        width=bar_width,
        label=model_name,
        color=colors.get(model_name, None)
    )

plt.xticks(
    x + bar_width,
    top_features,
    rotation=45,
    ha="right"
)
plt.ylabel("Importance Score")
plt.title("Top 10 Feature Importances - UCI Credit Card Default Dataset")
plt.legend()
plt.tight_layout()

# Save figure
output_path = "./evaluation/feature_importance_uci_all_models.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300)
plt.show()

print(f"✅ Plot saved: {output_path}")
