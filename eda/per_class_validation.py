import torch
import torch
from src.inference_utils import load_trained_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_trained_model("models/partial_ft_v2_raw_best.pt", device)
from src.dataset import get_dataloader, CLASS_TO_IDX

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

val_loader = get_dataloader("data/processed/splits/val.csv", train=False, batch_size=32)

model.eval()

correct_per_class = {name: 0 for name in CLASS_TO_IDX}
total_per_class = {name: 0 for name in CLASS_TO_IDX}

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)

        for true_label, pred_label in zip(labels, predicted):
            class_name = IDX_TO_CLASS[true_label.item()]
            total_per_class[class_name] += 1
            if true_label == pred_label:
                correct_per_class[class_name] += 1

print("Per-class validation accuracy:")
for name in CLASS_TO_IDX:
    total = total_per_class[name]
    correct = correct_per_class[name]
    acc = correct / total if total > 0 else 0
    print(f"  {name}: {correct}/{total} = {acc:.1%}")