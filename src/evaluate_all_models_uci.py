import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ----------------------------------------------------
# Paths
# ----------------------------------------------------
X_path = "data/X_resampled_uci.npy"
y_path = "data/y_resampled_uci.npy"
output_csv = "evaluation/results_table_uci.csv"
output_plot = "evaluation/results_table_uci_plot.png"

# ----------------------------------------------------
# Load data
# ----------------------------------------------------
X = np.load(X_path)
y = np.load(y_path)

# ----------------------------------------------------
# Train-test split
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------
# Models to evaluate
# ----------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(solver="liblinear"),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    ),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

# ----------------------------------------------------
# Evaluate all models
# ----------------------------------------------------
results = []

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }
    results.append(metrics)

# ----------------------------------------------------
# Save results
# ----------------------------------------------------
df_results = pd.DataFrame(results)
df_results.to_csv(output_csv, index=False)
print(f"✅ Saved results table to: {output_csv}")

print("\n=== Table I Data ===")
print(df_results)

# ----------------------------------------------------
# Plot comparison chart
# ----------------------------------------------------

# Metrics to plot
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

# Create a figure
plt.figure(figsize=(10, 6))

# Plot each metric as a separate bar set
bar_width = 0.15
positions = np.arange(len(df_results["Model"]))

for idx, metric in enumerate(metrics_to_plot):
    plt.bar(
        positions + idx * bar_width,
        df_results[metric],
        width=bar_width,
        label=metric
    )

plt.xticks(
    positions + bar_width * (len(metrics_to_plot) - 1) / 2,
    df_results["Model"],
    rotation=45,
    ha="right"
)
plt.ylim(0, 1.1)
plt.ylabel("Score")
plt.title("Model Performance on UCI Dataset")
plt.legend()
plt.tight_layout()

# Save figure
plt.savefig(output_plot, dpi=300)
print(f"✅ Saved plot to: {output_plot}")

plt.show()
