import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
import os
import argparse


def preprocess_german():
    print("🔄 Preprocessing South German Credit Dataset...")
    data_path = os.path.join("data", "south_german_credit.csv")
    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.drop("Credit_Risk", axis=1)
    y = df["Credit_Risk"]

    # One-hot encoding for categorical features
    X_encoded = pd.get_dummies(X)

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # Save preprocessed arrays
    np.save(os.path.join("data", "X_resampled_german.npy"), X_resampled)
    np.save(os.path.join("data", "y_resampled_german.npy"), y_resampled)

    print("✅ South German dataset preprocessing complete. Files saved.")


def preprocess_uci():
    print("🔄 Preprocessing UCI Credit Card Dataset...")
    uci_path = os.path.join("data", "default of credit card clients.xls")
    df = pd.read_excel(uci_path, header=1)

    # Drop ID column if present
    df = df.drop(columns=["ID"], errors="ignore")

    # Separate features and target
    X = df.drop("default payment next month", axis=1)
    y = df["default payment next month"]

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
