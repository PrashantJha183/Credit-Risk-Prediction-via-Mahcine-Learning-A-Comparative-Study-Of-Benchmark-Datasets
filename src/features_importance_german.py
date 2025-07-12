import matplotlib.pyplot as plt
import numpy as np
import joblib
import pandas as pd
import os

# -------------------------------------------------------------------
# Load trained RandomForest model for German dataset
model_path = "./models/rf_model_german.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = joblib.load(model_path)

# -------------------------------------------------------------------
# Load original German dataset and generate correct one-hot encoded feature names
csv_path = "./data/south_german_credit.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset file not found: {csv_path}")

df = pd.read_csv(csv_path)

# Drop target column
if "Credit_Risk" not in df.columns:
    raise ValueError("'Credit_Risk' column not found in the dataset.")

X = df.drop("Credit_Risk", axis=1)

# Apply the same one-hot encoding as during preprocessing
X_encoded = pd.get_dummies(X)

# Confirm the feature names
features = list(X_encoded.columns)

# -------------------------------------------------------------------
# Get feature importances
importances = model.feature_importances_

# Sanity check
if len(importances) != len(features):
    raise ValueError(
        f"Feature importances length ({len(importances)}) does not match "
        f"number of features ({len(features)})."
    )

# Sort indices by importance descending
indices = np.argsort(importances)[::-1]

# -------------------------------------------------------------------
# Plot top 10 features
plt.figure(figsize=(10, 6))
plt.title("Top 10 Features - South German Credit Dataset")

plt.bar(range(10), importances[indices[:10]], color="skyblue")
plt.xticks(
    range(10),
    [features[i] for i in indices[:10]],
    rotation=45,
    ha="right"
)
plt.ylabel("Importance Score")
plt.tight_layout()

# Save figure for paper inclusion
output_path = "./evaluation/feature_importance_german.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300)

print(f"✅ Feature importance plot saved to: {output_path}")

plt.show()
