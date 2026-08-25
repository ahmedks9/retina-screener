import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from src.transforms import preprocess_image, preprocess_image_train
from src.transforms import preprocess_cached, preprocess_cached_train

CLASS_TO_IDX = {
    "N": 0, "D": 1, "O": 2, "C": 3, "G": 4, "A": 5, "M": 6
}
IMAGE_DIR = Path("data/processed/images_224")

class RetinaDataset(Dataset):
    def __init__(self, csv_path: str, train: bool = False):
        self.df = pd.read_csv(csv_path)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = IMAGE_DIR / row["filename"]

        image = Image.open(img_path).convert("RGB")

        if self.train:
            tensor = preprocess_cached_train(image)
        else:
            tensor = preprocess_cached(image)

        label = CLASS_TO_IDX[row["class"]]

        return tensor, label
    
from torch.utils.data import DataLoader

def get_dataloader(csv_path: str, train: bool = False, batch_size: int = 32):
    dataset = RetinaDataset(csv_path, train=train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=14,
    )
import torch

# def compute_class_weights(csv_path: str = "data/processed/splits/train.csv"):
#     df = pd.read_csv(csv_path)
#     class_counts = df["class"].value_counts()

#     num_classes = len(CLASS_TO_IDX)
#     total_samples = len(df)

#     weights = torch.zeros(num_classes)
#     for class_name, idx in CLASS_TO_IDX.items():
#         count = class_counts[class_name]
#         weights[idx] = total_samples / (num_classes * count)

#     return weights
def compute_class_weights(csv_path: str = "data/processed/splits/train.csv", soften: bool = False):
    df = pd.read_csv(csv_path)
    class_counts = df["class"].value_counts()

    num_classes = len(CLASS_TO_IDX)
    total_samples = len(df)

    weights = torch.zeros(num_classes)
    for class_name, idx in CLASS_TO_IDX.items():
        count = class_counts[class_name]
        weights[idx] = total_samples / (num_classes * count)

    if soften:
        weights = torch.sqrt(weights)

    return weights