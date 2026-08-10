import pandas as pd
import ast

df = pd.read_csv("data/raw/full_df.csv")

# Parse the 'target' column into a real list
df["target_parsed"] = df["target"].apply(ast.literal_eval)

# Full original order, as defined by the CSV header - never change this list
ALL_CLASSES = ["N", "D", "G", "C", "A", "H", "M", "O"]
for i, name in enumerate(ALL_CLASSES):
    df[f"label_{name}"] = df["target_parsed"].apply(lambda x: x[i])

# Scope decision (Chapter 2, Phase 2.2): exclude H
TARGET_CLASSES = ["N", "D", "O", "C", "G", "A", "M"]

class_counts = df[[f"label_{n}" for n in TARGET_CLASSES]].sum()
class_counts.index = TARGET_CLASSES

print("Class distribution (7-class scope, H excluded):")
print(class_counts.sort_values(ascending=False))

# Check: how many rows have zero labels across our in-scope classes?
# (e.g., patients whose only condition was the excluded H)
zero_label_rows = (df[[f"label_{n}" for n in TARGET_CLASSES]].sum(axis=1) == 0).sum()
print(f"\nRows with zero in-scope labels (likely H-only patients): {zero_label_rows}")


df_filtered = df[df[[f"label_{n}" for n in TARGET_CLASSES]].sum(axis=1) > 0].copy()

print(f"\nOriginal row count: {len(df)}")
print(f"Filtered row count (H-only patients dropped): {len(df_filtered)}")
print(f"Rows dropped: {len(df) - len(df_filtered)}")


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Plot 1: Class distribution bar chart ---
final_counts = df_filtered[[f"label_{n}" for n in TARGET_CLASSES]].sum()
final_counts.index = TARGET_CLASSES
final_counts = final_counts.sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=final_counts.index, y=final_counts.values)
plt.title("Class Distribution (7-class scope, filtered)")
plt.xlabel("Disease Class")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.savefig("eda/class_distribution.png")
plt.close()
print("\nSaved: eda/class_distribution.png")

# --- Plot 2: Co-occurrence matrix ---
label_matrix = df_filtered[[f"label_{n}" for n in TARGET_CLASSES]].values  # shape: (rows, 7)
cooc = label_matrix.T @ label_matrix  # matrix multiply: counts how often each pair is BOTH 1
np.fill_diagonal(cooc, 0)  # zero out self-co-occurrence (meaningless)

plt.figure(figsize=(7, 6))
sns.heatmap(cooc, annot=True, fmt="d", xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES, cmap="Blues")
plt.title("Disease Co-occurrence Matrix")
plt.tight_layout()
plt.savefig("eda/cooccurrence_matrix.png")
plt.close()
print("Saved: eda/cooccurrence_matrix.png")


all_label_cols = [f"label_{n}" for n in ALL_CLASSES]
labels_per_row = df[all_label_cols].sum(axis=1)

print("\n--- Diagnostic: labels per row (full 8-class, unfiltered) ---")
print(labels_per_row.value_counts().sort_index())