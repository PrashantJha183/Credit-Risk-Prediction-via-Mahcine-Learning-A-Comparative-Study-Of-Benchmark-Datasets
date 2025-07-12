import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
import os

# Load CSV
data_path = os.path.join("data", "south_german_credit.csv")
df = pd.read_csv(data_path)

# Separate features and target
X = df.drop("Credit_Risk", axis=1)
y = df["Credit_Risk"]

# One-hot encoding
X_encoded = pd.get_dummies(X)

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# SMOTE oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Save preprocessed arrays (better names)
np.save(os.path.join("data", "X_resampled_german.npy"), X_resampled)
np.save(os.path.join("data", "y_resampled_german.npy"), y_resampled)

print("Preprocessing complete. Files saved.")
