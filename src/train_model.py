import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    print("⚠️ XGBoost not installed. Install xgboost to enable that model.")

def train_and_evaluate(dataset="german", model_name="rf", **kwargs):
    """
    Train a ML model on preprocessed data and save results & plots
    """

    # Paths
    data_dir = Path("data")
    eval_dir = Path("evaluation")
    eval_dir.mkdir(exist_ok=True)

    if dataset == "german":
        X_path = data_dir / "X_resampled_german.npy"
        y_path = data_dir / "y_resampled_german.npy"
        output_csv = eval_dir / "results_german.csv"
        plot_path = eval_dir / "results_plot_german.png"
    elif dataset == "uci":
        X_path = data_dir / "X_resampled_uci.npy"
        y_path = data_dir / "y_resampled_uci.npy"
        output_csv = eval_dir / "results_uci.csv"
        plot_path = eval_dir / "results_plot_uci.png"
    else:
        raise ValueError("Unknown dataset specified.")

    # Load arrays
    X_resampled = np.load(X_path)
    y_resampled = np.load(y_path)

    print(f"✅ Loaded data from {X_path}. Shape: {X_resampled.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )
    print(f"✅ Data split: Train={X_train.shape}, Test={X_test.shape}")

    # Pick model
    if model_name.lower() == "rf":
        model = RandomForestClassifier(
            n_estimators=int(kwargs.get("n_estimators", 100)),
            random_state=42
        )
    elif model_name.lower() == "logreg":
        model = LogisticRegression(
            solver="liblinear",
            C=float(kwargs.get("C", 1.0)),
            max_iter=int(kwargs.get("max_iter", 100))
        )
    elif model_name.lower() == "xgb":
        if XGBClassifier is None:
            print("❌ XGBoost not available. Install it first.")
            sys.exit(1)
        model = XGBClassifier(
            n_estimators=int(kwargs.get("n_estimators", 100)),
            max_depth=int(kwargs.get("max_depth", 3)),
            learning_rate=float(kwargs.get("learning_rate", 0.1)),
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    print(f"✅ Training {model_name.upper()} model...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except:
        y_proba = None

    # Evaluation
    report = classification_report(y_test, y_pred, output_dict=True)
    if y_proba is not None:
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = None

    # Save results
    results_df = pd.DataFrame(report).transpose()
    results_df["roc_auc"] = roc_auc
    results_df.to_csv(output_csv)
    print(f"✅ Saved results to {output_csv}")

    # Print report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")

    # Plotting
    class_labels = [str(cls) for cls in sorted(np.unique(y_test))]
    classes_to_plot = [cls for cls in class_labels if cls in results_df.index]

    if classes_to_plot:
        filtered = results_df.loc[
            classes_to_plot,
            ["precision", "recall", "f1-score"]
        ]

        filtered.plot(
            kind="bar",
            figsize=(8, 5),
            title=f"Performance Metrics per Class ({dataset.title()}, {model_name.upper()})"
        )
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"✅ Plot saved to {plot_path}")
        plt.close()
    else:
        print("⚠️ No classes found for plotting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["german", "uci"], default="german", help="Dataset to use.")
    parser.add_argument("--model", choices=["rf", "xgb", "logreg"], default="rf", help="Model to train.")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees for RF/XGB.")
    parser.add_argument("--max_depth", type=int, default=3, help="Max tree depth for XGB.")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate for XGB.")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength for Logistic Regression.")
    parser.add_argument("--max_iter", type=int, default=100, help="Max iterations for Logistic Regression.")

    args = parser.parse_args()

    # Pass only relevant args
    train_and_evaluate(
        dataset=args.dataset,
        model_name=args.model,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        C=args.C,
        max_iter=args.max_iter
    )
