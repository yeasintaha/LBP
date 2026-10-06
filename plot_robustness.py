import os
import csv
import matplotlib.pyplot as plt
import numpy as np

csv_file = os.path.join("results", "robustness_improvements.csv")

methods = []
data = []
conditions = []

with open(csv_file, mode="r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    conditions = header[1:] 
    for row in reader:
        if row:
            methods.append(row[0])
            data.append([float(x) for x in row[1:]])

data = np.array(data)

x = np.arange(len(conditions))
width = 0.25

plt.figure(figsize=(14, 6))

colors = ["#4C72B0", "#55A868", "#C44E52"]

for i, method in enumerate(methods):
    offset = (i - 1) * width
    plt.bar(x + offset, data[i], width=width, label=method, color=colors[i % len(colors)], alpha=0.9, edgecolor="black", linewidth=0.5)

plt.ylabel("Recognition Rate", fontsize=12)
plt.title("Robustness Comparison: Baseline vs. Histogram Equalization vs. Tan-Triggs", fontsize=14, fontweight="bold")
plt.xticks(x, conditions, rotation=25, ha="right", fontsize=10)
plt.ylim(0.4, 1.05)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend(loc="lower left", fontsize=10, framealpha=0.9)

plt.tight_layout()

os.makedirs("results", exist_ok=True)
out_png = os.path.join("results", "figure_robustness_comparison.png")
plt.savefig(out_png, dpi=300)
plt.close()

print(f"Saved bar plot to: {out_png}")
