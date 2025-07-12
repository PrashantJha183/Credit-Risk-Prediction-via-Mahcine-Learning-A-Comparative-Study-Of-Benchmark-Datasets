import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import os

# Load preprocessed arrays
X_path = os.path.join("data", "X_resampled_german.npy")
y_path = os.path.join("data", "y_resampled_german.npy")

X_resampled = np.load(X_path)
y_resampled = np.load(y_path)

print(f"✅ Loaded preprocessed data. Shape: {X_resampled.shape}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

print("✅ Data split into train and test.")

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("✅ Model training complete.")

# Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluation
report = classification_report(y_test, y_pred, output_dict=True)
roc_auc = roc_auc_score(y_test, y_proba)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC AUC: {roc_auc:.4f}")

# Save results
results_df = pd.DataFrame(report).transpose()
results_df["roc_auc"] = roc_auc

# Create evaluation folder if it doesn’t exist
os.makedirs("evaluation", exist_ok=True)

results_df.to_csv(os.path.join("evaluation", "results_german.csv"))
print("Results saved to evaluation/results_german.csv")
