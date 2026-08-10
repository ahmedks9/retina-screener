import pandas as pd
import ast
from PIL import Image
from pathlib import Path

df = pd.read_csv("data/raw/full_df.csv")
df["target_parsed"] = df["target"].apply(ast.literal_eval)

IMAGE_DIR = Path("data/raw/ODIR-5K/ODIR-5K/Training Images")

# corrupt_files = []
# missing_files = []

# for fname in df["filename"]:
#     img_path = IMAGE_DIR / fname
#     if not img_path.exists():
#         missing_files.append(fname)
#         continue
#     try:
#         with Image.open(img_path) as img:
#             img.verify()  # checks integrity without fully decoding
#     except Exception as e:
#         corrupt_files.append((fname, str(e)))

# print(f"Total files checked: {len(df)}")
# print(f"Missing files: {len(missing_files)}")
# print(f"Corrupt files: {len(corrupt_files)}")

# if missing_files:
#     print("\nFirst 5 missing:", missing_files[:5])
# if corrupt_files:
#     print("\nFirst 5 corrupt:", corrupt_files[:5])


import hashlib
from collections import defaultdict

hash_to_files = defaultdict(list)
resolutions = []

for fname in df["filename"]:
    img_path = IMAGE_DIR / fname
    if not img_path.exists():
        continue

    # Duplicate detection: hash the raw file bytes
    with open(img_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    hash_to_files[file_hash].append(fname)

    # Resolution check: open and record width/height
    with Image.open(img_path) as img:
        resolutions.append(img.size)  # (width, height)

# Report duplicates
duplicate_groups = {h: files for h, files in hash_to_files.items() if len(files) > 1}
print(f"\nExact duplicate groups found: {len(duplicate_groups)}")
if duplicate_groups:
    for h, files in list(duplicate_groups.items())[:5]:
        print(f"  Duplicate set: {files}")

# Report resolution stats
widths = [r[0] for r in resolutions]
heights = [r[1] for r in resolutions]
print(f"\nWidth  - min: {min(widths)}, max: {max(widths)}, most common: {pd.Series(widths).mode()[0]}")
print(f"Height - min: {min(heights)}, max: {max(heights)}, most common: {pd.Series(heights).mode()[0]}")

aspect_ratios = [w/h for w, h in resolutions]
print(f"Aspect ratio - min: {min(aspect_ratios):.2f}, max: {max(aspect_ratios):.2f}, avg: {sum(aspect_ratios)/len(aspect_ratios):.2f}")