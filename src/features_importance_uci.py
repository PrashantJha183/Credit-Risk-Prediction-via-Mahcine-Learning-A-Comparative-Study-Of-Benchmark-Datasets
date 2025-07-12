import matplotlib.pyplot as plt
import numpy as np
import joblib

# Load trained RandomForest model for UCI dataset
model = joblib.load("./models/rf_model_uci.pkl")

# List of all feature names in correct order from the dataset
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

# Get feature importances
importances = model.feature_importances_

# Sort indices by importance descending
indices = np.argsort(importances)[::-1]

# Plot top 10 features
plt.figure(figsize=(10,6))
plt.title("Top 10 Features - UCI Credit Card Default")
plt.bar(range(10), importances[indices[:10]], color='skyblue')
plt.xticks(
    range(10),
    [features[i] for i in indices[:10]],
    rotation=45,
    ha='right'
)
plt.ylabel("Importance Score")
plt.tight_layout()

# Save figure for paper inclusion
plt.savefig("evaluation/feature_importance_uci.png", dpi=300)
plt.show()
