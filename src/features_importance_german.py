import matplotlib.pyplot as plt
import numpy as np
import joblib
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------
# Setup
# ---------------------------------------------------------------
MODELS_DIR = Path("./models")
DATA_DIR = Path("./data")
EVAL_DIR = Path("./evaluation")
EVAL_DIR.mkdir(exist_ok=True)

# Models to process
model_names = ["rf", "logreg", "xgb"]  # skip 'svm'
model_labels = {
    "rf": "Random Forest",
    "logreg": "Logistic Regression",
    "xgb": "XGBoost"
}
colors = {
    "rf": "#1f77b4",
    "logreg": "#ff7f0e",
    "xgb": "#2ca02c"
}

# ---------------------------------------------------------------
# Load dataset to get consistent feature names
# ---------------------------------------------------------------
csv_path = DATA_DIR / "south_german_credit.csv"
df = pd.read_csv(csv_path)

if "Credit_Risk" not in df.columns:
    raise ValueError("Credit_Risk column missing in dataset.")

X = df.drop("Credit_Risk", axis=1)
X_encoded = pd.get_dummies(X)
features = list(X_encoded.columns)

# ---------------------------------------------------------------
# Collect feature importances
# ---------------------------------------------------------------
model_feature_importances = {}

for model_name in model_names:
    model_path = MODELS_DIR / f"{model_name}_model_german.pkl"
    if not model_path.exists():
        print(f"⚠️ Skipping {model_name.upper()}: model file not found.")
        continue

    model = joblib.load(model_path)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coefs = model.coef_
        importances = np.abs(coefs[0]) if coefs.ndim == 2 else np.mean(np.abs(coefs), axis=0)
    else:
        print(f"⚠️ Model {model_name.upper()} has no importances. Skipping.")
        continue

    if len(importances) != len(features):
        raise ValueError(f"Mismatch: {len(importances)} importances vs {len(features)} features")

    model_feature_importances[model_name] = importances

# ---------------------------------------------------------------
# Identify top 10 common features based on average importance
# ---------------------------------------------------------------
avg_importance = np.mean(
    np.array(list(model_feature_importances.values())),
    axis=0
)
top_indices = np.argsort(avg_importance)[::-1][:10]
top_features = [features[i] for i in top_indices]

# ---------------------------------------------------------------
# Create combined bar plot
# ---------------------------------------------------------------
x = np.arange(len(top_features))  # feature indices
width = 0.25

plt.figure(figsize=(12, 6))
for i, model_name in enumerate(model_names):
    importances = model_feature_importances[model_name]
    scores = [importances[features.index(f)] for f in top_features]
    plt.bar(x + i * width, scores, width=width, label=model_labels[model_name], color=colors[model_name])

plt.xticks(x + width, top_features, rotation=45, ha="right")
plt.ylabel("Importance Score")
plt.title("Top 10 Feature Importances (South German Credit Dataset)")
plt.legend()
plt.tight_layout()

# Save
combined_path = EVAL_DIR / "feature_importance_german_combined.png"
plt.savefig(combined_path, dpi=300)
plt.show()

print(f"✅ Combined feature importance plot saved to: {combined_path}")
