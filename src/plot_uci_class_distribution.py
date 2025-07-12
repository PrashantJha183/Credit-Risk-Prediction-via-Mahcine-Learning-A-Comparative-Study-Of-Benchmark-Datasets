import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# --------------------------------------------------------
# Paths
# --------------------------------------------------------

BASE_DIR = os.getcwd()

data_excel_path = os.path.join(BASE_DIR, "data", "default_of_credit_card_clients.xls")
y_resampled_path = os.path.join(BASE_DIR, "data", "y_resampled_uci.npy")
eval_dir = os.path.join(BASE_DIR, "evaluation")

# Ensure output directory exists
os.makedirs(eval_dir, exist_ok=True)

# --------------------------------------------------------
# Plot BEFORE SMOTE
# --------------------------------------------------------

print("🔹 Loading original UCI dataset from:", data_excel_path)

if not os.path.exists(data_excel_path):
    raise FileNotFoundError(f"Excel file not found at {data_excel_path}")

# Read Excel (header starts at row 2 → index 1)
df_uci = pd.read_excel(data_excel_path, header=1)

if "default payment next month" not in df_uci.columns:
    print("Available columns:", df_uci.columns.tolist())
    raise ValueError("Could not find 'default payment next month' column.")

counts_before = df_uci["default payment next month"].value_counts().sort_index()
labels_before = ["No Default (0)", "Default (1)"]

plt.figure(figsize=(5, 4))
bars = plt.bar(labels_before, counts_before.values, color="#1f77b4")
plt.title("Class Distribution BEFORE SMOTE (UCI Data)")
plt.xlabel("Class")
plt.ylabel("Count")

# Add bar labels
for bar, value in zip(bars, counts_before.values):
    plt.text(bar.get_x() + bar.get_width() / 2, value + 100, f"{value:,}", 
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()

before_path = os.path.join(eval_dir, "class_dist_before_smote_uci.png")
plt.savefig(before_path, dpi=300)
print("✅ Saved:", before_path)
plt.show()

# --------------------------------------------------------
# Plot AFTER SMOTE
# --------------------------------------------------------

print("🔹 Loading resampled y labels from:", y_resampled_path)

if not os.path.exists(y_resampled_path):
    raise FileNotFoundError(f"Numpy file not found at {y_resampled_path}")

y_resampled = np.load(y_resampled_path)

counts_after = Counter(y_resampled)
labels_after = ["No Default (0)", "Default (1)"]
values_after = [counts_after[0], counts_after[1]]

plt.figure(figsize=(5, 4))
bars = plt.bar(labels_after, values_after, color="#2ca02c")
plt.title("Class Distribution AFTER SMOTE (UCI Data)")
plt.xlabel("Class")
plt.ylabel("Count")

# Add bar labels
for bar, value in zip(bars, values_after):
    plt.text(bar.get_x() + bar.get_width() / 2, value + 100, f"{value:,}",
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()

after_path = os.path.join(eval_dir, "class_dist_after_smote_uci.png")
plt.savefig(after_path, dpi=300)
print("✅ Saved:", after_path)
plt.show()

print("🎉 All plots generated successfully!")
