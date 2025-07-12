import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# Load resampled data
X = np.load("data/X_resampled_german.npy")
y = np.load("data/y_resampled_german.npy")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load trained model (e.g. Random Forest)
model = joblib.load("models/rf_model_german.pkl")

# Predict probabilities
y_probs = model.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - South German Credit")
plt.legend(loc="lower right")
plt.tight_layout()

# Save figure
plt.savefig("evaluation/roc_curve_german.png", dpi=300)
plt.show()

print("ROC curve saved as evaluation/roc_curve_german.png")
