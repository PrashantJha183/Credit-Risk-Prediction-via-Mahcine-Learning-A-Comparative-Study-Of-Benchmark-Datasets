import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
import os
import argparse
import joblib  # <-- new

def preprocess_german():
    print("🔄 Preprocessing South German Credit Dataset...")
    data_path = os.path.join("data", "south_german_credit.csv")
    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.drop("Credit_Risk", axis=1)
    y = df["Credit_Risk"].replace({1: 0, 2: 1})


    # One-hot encoding for categorical features
    X_encoded = pd.get_dummies(X)

    # ✅ Save feature names
    feature_names = X_encoded.columns.tolist()
    joblib.dump(feature_names, "./models/feature_names_german.pkl")

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # Save preprocessed arrays
    np.save(os.path.join("data", "X_resampled_german.npy"), X_resampled)
    np.save(os.path.join("data", "y_resampled_german.npy"), y_resampled)

    print("✅ South German dataset preprocessing complete. Files saved.")


def preprocess_uci():
    print("🔄 Preprocessing UCI Credit Card Dataset...")
    uci_path = os.path.join("data", "default_of_credit_card_clients.xls")
    df = pd.read_excel(uci_path, header=1)

    # Drop ID column if present
    df = df.drop(columns=["ID"], errors="ignore")

    # Separate features and target
    X = df.drop("default payment next month", axis=1)
    y = df["default payment next month"]

    # ✅ Save feature names directly (already numerical)
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, "./models/feature_names_uci.pkl")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # Save arrays
    np.save(os.path.join("data", "X_resampled_uci.npy"), X_resampled)
    np.save(os.path.join("data", "y_resampled_uci.npy"), y_resampled)

    print("✅ UCI dataset preprocessing complete. Files saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess datasets")
    parser.add_argument("--dataset", type=str, choices=["german", "uci"], required=True, help="Dataset to preprocess")
    args = parser.parse_args()

    if args.dataset == "german":
        preprocess_german()
    elif args.dataset == "uci":
        preprocess_uci()
