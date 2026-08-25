import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from src.inference_utils import load_trained_model
from src.dataset import CLASS_TO_IDX
from src.transforms import preprocess_cached

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
IMAGE_DIR = "data/processed/images_224"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_trained_model("models/partial_ft_v2_raw_best.pt", device)

test_df = pd.read_csv("data/processed/splits/test.csv")

misclassified = []

with torch.no_grad():
    for _, row in test_df.iterrows():
        img = Image.open(f"{IMAGE_DIR}/{row['filename']}").convert("RGB")
        tensor = preprocess_cached(img).unsqueeze(0).to(device)
        output = model(tensor)
        pred_idx = output.argmax(dim=1).item()
        pred_class = IDX_TO_CLASS[pred_idx]

        true_class = row["class"]
        if true_class in ["G", "A", "O"] and pred_class == "N":
            misclassified.append((row["filename"], true_class, pred_class))

print(f"Found {len(misclassified)} G/A/O images misclassified as N")

num_to_show = min(6, len(misclassified))
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, (fname, true_c, pred_c) in enumerate(misclassified[:num_to_show]):
    img = Image.open(f"{IMAGE_DIR}/{fname}")
    ax = axes[i // 3, i % 3]
    ax.imshow(img)
    ax.set_title(f"True: {true_c}, Predicted: {pred_c}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("eda/misclassified_examples.png")
print("Saved eda/misclassified_examples.png")