import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path

# ---------------------------
# Model configs
# ---------------------------

models_config = {
    "Logistic Reg.": LogisticRegression(solver="liblinear", max_iter=100, C=1.0),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    ),
    "SVM": SVC(kernel='linear', probability=True, random_state=42, max_iter=1000)
}

# ---------------------------
# Datasets
# ---------------------------

datasets = {
    "UCI": {
        "X_path": "data/X_resampled_uci.npy",
        "y_path": "data/y_resampled_uci.npy"
    },
    "German": {
        "X_path": "data/X_resampled_german.npy",
        "y_path": "data/y_resampled_german.npy"
    }
}

# ---------------------------
# Measure training times
# ---------------------------

results = []

for dataset_name, paths in datasets.items():
    X = np.load(paths["X_path"])
    y = np.load(paths["y_path"])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n🔹 Dataset: {dataset_name} - Shape: {X.shape}")
    
    for model_name, model in models_config.items():
        print(f"Training {model_name}...")
        
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()
        
        elapsed = end - start
        print(f"✅ {model_name} trained in {elapsed:.3f} seconds")
        
        results.append({
            "Model": model_name,
            "Dataset": dataset_name,
            "Time (sec)": round(elapsed, 3)
        })

# Convert to DataFrame
df_results = pd.DataFrame(results)

# Pivot for nice table format
pivot_table = df_results.pivot(index="Model", columns="Dataset", values="Time (sec)")
pivot_table = pivot_table[["UCI", "German"]]  # enforce column order

# Print final table
print("\nFinal Table:")
print(pivot_table)

# Save CSV for paper
pivot_table.to_csv("evaluation/training_times_table.csv")
print("\n✅ Table saved to evaluation/training_times_table.csv")
