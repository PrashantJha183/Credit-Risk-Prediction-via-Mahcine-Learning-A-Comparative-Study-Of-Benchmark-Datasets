import pandas as pd
import matplotlib.pyplot as plt
import os

# Load saved results
results_path = os.path.join("evaluation", "results_german.csv")
results = pd.read_csv(results_path, index_col=0)

# Print table
print("Classification Report Table:")
print(results)

# Debug check
print("Index values in DataFrame:", results.index.tolist())

# Convert index to string for safety
results.index = results.index.astype(str)

classes_to_plot = ["1", "2"]

classes_present = [cls for cls in classes_to_plot if cls in results.index]

if len(classes_present) == 0:
    print("⚠️ No matching classes found for plotting. Skipping plot.")
else:
    filtered_results = results.loc[
        classes_present,
        ["precision", "recall", "f1-score"]
    ]

    if filtered_results.empty:
        print("⚠️ Filtered DataFrame empty. Nothing to plot.")
    else:
        ax = filtered_results.plot(
            kind="bar",
            figsize=(8, 5),
            title="Performance Metrics per Class (German Dataset)"
        )
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.legend(loc="lower right")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Optional: add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8)
        
        plt.tight_layout()

        plot_path = os.path.join("evaluation", "results_plot_german.png")
        plt.savefig(plot_path)
        print(f"✅ Plot saved to {plot_path}")
        plt.show()
