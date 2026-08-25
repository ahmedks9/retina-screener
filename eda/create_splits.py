import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from pathlib import Path

df = pd.read_csv("data/raw/full_df.csv")
df["target_parsed"] = df["target"].apply(ast.literal_eval)

ALL_CLASSES = ["N", "D", "G", "C", "A", "H", "M", "O"]
for i, name in enumerate(ALL_CLASSES):
    df[f"label_{name}"] = df["target_parsed"].apply(lambda x: x[i])

TARGET_CLASSES = ["N", "D", "O", "C", "G", "A", "M"]

# Re-apply Phase 2.2 filtering: drop rows with zero in-scope labels
df_filtered = df[df[[f"label_{n}" for n in TARGET_CLASSES]].sum(axis=1) > 0].copy()

# Create a single-column class label (safe now - we confirmed exactly
# one label is active per row in Step 2.2.4)
def get_class(row):
    for name in TARGET_CLASSES:
        if row[f"label_{name}"] == 1:
            return name
df_filtered["class"] = df_filtered.apply(get_class, axis=1)

# First split: 70% train, 30% temp (val+test combined)
train_df, temp_df = train_test_split(
    df_filtered, test_size=0.30, stratify=df_filtered["class"], random_state=42
)

# Second split: split the 30% temp evenly into val (15%) and test (15%)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["class"], random_state=42
)

# Save
Path("data/processed/splits").mkdir(parents=True, exist_ok=True)
train_df[["filename", "class"]].to_csv("data/processed/splits/train.csv", index=False)
val_df[["filename", "class"]].to_csv("data/processed/splits/val.csv", index=False)
test_df[["filename", "class"]].to_csv("data/processed/splits/test.csv", index=False)

print(f"Train: {len(train_df)} rows")
print(f"Val:   {len(val_df)} rows")
print(f"Test:  {len(test_df)} rows")
print("\nClass distribution per split:")
for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    print(f"\n{name}:")
    print(split_df["class"].value_counts())