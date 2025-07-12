import pandas as pd
import matplotlib.pyplot as plt
import os

# Load saved results
results_path = os.path.join("evaluation", "results_german.csv")
results = pd.read_csv(results_path, index_col=0)

# Print the table
print("Classification Report Table:")
print(results)

# Ensure index is numeric if classes are integers
if results.index.dtype == "O":
    # e.g. convert "1" → 1
    try:
        results.index = results.index.astype(int)
    except:
        pass

# Only plot classes 1 and 2
classes_to_plot = [1, 2]
filtered_results = results.loc[
    results.index.isin(classes_to_plot),
    ["precision", "recall", "f1-score"]
]

filtered_results.plot(
    kind="bar",
    figsize=(8, 5),
    title="Performance Metrics per Class"
)

plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.tight_layout()

# Save the figure
plot_path = os.path.join("evaluation", "results_plot.png")
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")
plt.show()
