import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

from src.inference_utils import load_trained_model
from src.dataset import get_dataloader, CLASS_TO_IDX

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
CLASS_NAMES = [IDX_TO_CLASS[i] for i in range(len(CLASS_TO_IDX))]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_trained_model("models/partial_ft_v2_raw_best.pt", device)

test_loader = get_dataloader("data/processed/splits/test.csv", train=False, batch_size=32)

all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

print("=== Classification Report (Precision / Recall / F1 per class) ===")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=3))

print("=== Per-class AUC-ROC (one-vs-rest) ===")
auc_scores = roc_auc_score(all_labels, all_probs, multi_class="ovr", average=None)
for name, score in zip(CLASS_NAMES, auc_scores):
    print(f"  {name}: {score:.3f}")

print(f"\nMacro-average AUC-ROC: {roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro'):.3f}")
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(all_labels, all_preds)
cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(8, 7))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (row-normalized) - Test Set")
plt.tight_layout()
plt.savefig("eda/confusion_matrix.png")
print("\nSaved eda/confusion_matrix.png")