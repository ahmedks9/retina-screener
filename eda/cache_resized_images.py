import pandas as pd
from pathlib import Path
from PIL import Image

from src.transforms import resize_with_padding

RAW_IMAGE_DIR = Path("data/raw/ODIR-5K/ODIR-5K/Training Images")
CACHE_DIR = Path("data/processed/images_224")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

split_files = [
    "data/processed/splits/train.csv",
    "data/processed/splits/val.csv",
    "data/processed/splits/test.csv",
]

all_filenames = set()
for split_file in split_files:
    df = pd.read_csv(split_file)
    all_filenames.update(df["filename"].tolist())

print(f"Caching {len(all_filenames)} unique images...")

for i, fname in enumerate(all_filenames):
    out_path = CACHE_DIR / fname
    if out_path.exists():
        continue  # already cached, skip

    img = Image.open(RAW_IMAGE_DIR / fname).convert("RGB")
    resized = resize_with_padding(img)
    resized.save(out_path, quality=95)

    if (i + 1) % 500 == 0:
        print(f"  {i + 1}/{len(all_filenames)} done")

print("Done.")