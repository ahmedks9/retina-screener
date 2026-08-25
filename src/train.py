import torch
import torch.nn as nn

from src.model import build_model
from src.dataset import get_dataloader, compute_class_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# from src.model import build_model_partial_finetune
# model = build_model_partial_finetune().to(device)
from src.model import build_model_partial_finetune_v2
model = build_model_partial_finetune_v2().to(device)

class_weights = compute_class_weights(soften=False).to(device) 
criterion = nn.CrossEntropyLoss(weight=class_weights)

CLASSIFIER_LR = 1e-3
BACKBONE_LR = 1e-5

optimizer = torch.optim.AdamW([
    {"params": model.classifier.parameters(), "lr": CLASSIFIER_LR},
    {"params": list(model.features[6].parameters()) + list(model.features[7].parameters()) + list(model.features[8].parameters()), "lr": BACKBONE_LR},
])

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy

def validate(model, val_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy

# if __name__ == "__main__":
#     print("Class weights on device:", class_weights)
#     print("Loss function:", criterion)

# if __name__ == "__main__":
#     print("Class weights on device:", class_weights)
#     print("Loss function:", criterion)
#     print("Optimizer:", optimizer)

#     # Confirm optimizer only tracks trainable parameters
#     optimizer_param_count = sum(p.numel() for group in optimizer.param_groups for p in group["params"])
#     print(f"Parameters tracked by optimizer: {optimizer_param_count:,}")

# if __name__ == "__main__":
#     train_loader = get_dataloader("data/processed/splits/train.csv", train=True, batch_size=32)

#     print("Running one training epoch as a smoke test...")
#     loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
#     print(f"\nEpoch loss: {loss:.4f}")
#     print(f"Epoch accuracy: {acc:.4f}")
    
NUM_EPOCHS = 100
PATIENCE = 15

if __name__ == "__main__":
    train_loader = get_dataloader("data/processed/splits/train.csv", train=True, batch_size=32)
    val_loader = get_dataloader("data/processed/splits/val.csv", train=False, batch_size=32)

    history = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{NUM_EPOCHS} | "
              f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "models/partial_ft_raw_best.pt")
            print(f"  -> New best val loss ({val_loss:.4f}), model saved.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping triggered - no improvement for {PATIENCE} epochs.")
            break

    # import pandas as pd
    # import matplotlib.pyplot as plt
    # from pathlib import Path

    # Path("logs").mkdir(exist_ok=True)

    # history_df = pd.DataFrame(history)
    # lr_tag = f"lr{LEARNING_RATE}"
    # history_df.to_csv(f"logs/training_history_{lr_tag}.csv", index=False)
    # print(f"\nSaved logs/training_history_{lr_tag}.csv")

    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    # axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
    # axes[0].set_xlabel("Epoch")
    # axes[0].set_ylabel("Loss")
    # axes[0].set_title("Loss over Epochs")
    # axes[0].legend()

    # axes[1].plot(history_df["epoch"], history_df["train_acc"], label="Train Accuracy")
    # axes[1].plot(history_df["epoch"], history_df["val_acc"], label="Val Accuracy")
    # axes[1].set_xlabel("Epoch")
    # axes[1].set_ylabel("Accuracy")
    # axes[1].set_title("Accuracy over Epochs")
    # axes[1].legend()

    # plt.tight_layout()
    # plt.savefig(f"logs/training_curves_{lr_tag}.png")
    # print(f"Saved logs/training_curves_{lr_tag}.png")