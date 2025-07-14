import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from pathlib import Path
import time
import joblib
import sys

# Import XGBoost if available
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    print("XGBoost not installed. Install xgboost to enable that model.")

# ----------------------------------------------------------
# Function to train a single model
# ----------------------------------------------------------

def train_one_model(model_name, X_train, y_train, X_test, y_test, dataset, eval_dir):
    """
    Train one model and save:
      - CSV with metrics (including ROC AUC and time)
      - Model file
      - ROC curve data for plotting later
    """

    # ----------------------------------------------------------
    # Create model
    # ----------------------------------------------------------
    if model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    elif model_name == "logreg":
        model = LogisticRegression(
            solver="liblinear",
            C=1.0,
            max_iter=100
        )
    elif model_name == "xgb":
        if XGBClassifier is None:
            print("XGBoost not available. Skipping XGB.")
            return None
        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )
    elif model_name == "svm":
        print("⚠️ Training SVM with probability=True can be slow on large datasets.")
        model = SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        )
    else:
        print(f"Unknown model name: {model_name}")
        return None

    print(f"🔹 Training model: {model_name.upper()} on {dataset.upper()}")

    # ----------------------------------------------------------
    # Train
    # ----------------------------------------------------------
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()

    train_time_sec = end_time - start_time
    print(f"✅ Training time: {train_time_sec:.2f} seconds")

    # ----------------------------------------------------------
    # Predictions
    # ----------------------------------------------------------
    y_pred = model.predict(X_test)

    # Get probability scores for ROC AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = None

    # ----------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------
    report = classification_report(y_test, y_pred, output_dict=True)
    results_df = pd.DataFrame(report).transpose()

    if y_proba is not None:
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = None

    # Add a row for ROC AUC
    roc_row = pd.DataFrame({
        "precision": [np.nan],
        "recall": [np.nan],
        "f1-score": [np.nan],
        "support": [np.nan],
        "roc_auc": [roc_auc],
        "training_time_sec": [np.nan]
    })

    # Add a row for training time
    time_row = pd.DataFrame({
        "precision": [np.nan],
        "recall": [np.nan],
        "f1-score": [np.nan],
        "support": [np.nan],
        "roc_auc": [np.nan],
        "training_time_sec": [train_time_sec]
    })

    # Combine everything
    results_with_time = pd.concat([results_df, roc_row, time_row], ignore_index=True)

    # ----------------------------------------------------------
    # Save results CSV
    # ----------------------------------------------------------
    output_csv = eval_dir / f"results_{dataset}_{model_name}.csv"
    results_with_time.to_csv(output_csv, index=False)
    print(f"✅ Results saved to {output_csv}")

    # ----------------------------------------------------------
    # Save model
    # ----------------------------------------------------------
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / f"{model_name}_model_{dataset}.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")

    # ----------------------------------------------------------
    # Save ROC curve data
    # ----------------------------------------------------------
    if y_proba is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_data = pd.DataFrame({
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds
        })
        roc_csv_path = eval_dir / f"roc_curve_{dataset}_{model_name}.csv"
        roc_data.to_csv(roc_csv_path, index=False)
        print(f"✅ ROC curve data saved to {roc_csv_path}")

    # Print ROC AUC
    if roc_auc is not None:
        print(f"✅ ROC AUC for {model_name.upper()} on {dataset.upper()}: {roc_auc:.4f}")

    return model


# ----------------------------------------------------------
# Main script
# ----------------------------------------------------------

def main():
    data_dir = Path("data")
    eval_dir = Path("evaluation")
    eval_dir.mkdir(exist_ok=True)

    datasets = ["german", "uci"]
    model_names = ["rf", "logreg", "xgb", "svm"]

    for dataset in datasets:
        if dataset == "german":
            X_path = data_dir / "X_resampled_german.npy"
            y_path = data_dir / "y_resampled_german.npy"
        elif dataset == "uci":
            X_path = data_dir / "X_resampled_uci.npy"
            y_path = data_dir / "y_resampled_uci.npy"
        else:
            raise ValueError("Unknown dataset specified.")

        # Load data
        X = np.load(X_path)
        y = np.load(y_path)
        print(f"\n==============================")
        print(f"🔹 Loaded data for {dataset.upper()}: {X.shape}")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        for model_name in model_names:
            train_one_model(
                model_name,
                X_train, y_train,
                X_test, y_test,
                dataset,
                eval_dir
            )

if __name__ == "__main__":
    main()
