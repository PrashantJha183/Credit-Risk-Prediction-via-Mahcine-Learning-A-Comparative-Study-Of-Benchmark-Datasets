import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from pathlib import Path

def plot_roc_curves_uci():
    data_dir = Path("data")
    models_dir = Path("models")
    eval_dir = Path("evaluation")
    eval_dir.mkdir(exist_ok=True)

    # Load UCI data
    X_path = data_dir / "X_resampled_uci.npy"
    y_path = data_dir / "y_resampled_uci.npy"
    
    X = np.load(X_path)
    y = np.load(y_path)

    # Split test data (same split as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_names = ["rf", "logreg", "xgb", "svm"]
    colors = {
        "rf": "blue",
        "logreg": "darkorange",
        "xgb": "green",
        "svm": "purple"
    }

    plt.figure(figsize=(7, 6))

    for model_name in model_names:
        model_path = models_dir / f"{model_name}_model_uci.pkl"

        if not model_path.exists():
            print(f"⚠️ Model not found: {model_path}")
            continue

        print(f"🔹 Loading model: {model_name.upper()}")
        model = joblib.load(model_path)

        # Predict probabilities
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_probs = model.decision_function(X_test)
        else:
            print(f"⚠️ Model {model_name.upper()} does not support probability outputs. Skipping.")
            continue

        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr, tpr,
            label=f"{model_name.upper()} (AUC = {roc_auc:.3f})",
            color=colors[model_name],
            lw=2
        )

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - UCI Credit Card Default Dataset")
    plt.legend(loc="lower right")
    plt.tight_layout()

    plot_path = eval_dir / "roc_curve_uci.png"
    plt.savefig(plot_path, dpi=300)
    plt.show()

    print(f"✅ ROC curves saved to {plot_path}")

if __name__ == "__main__":
    plot_roc_curves_uci()
