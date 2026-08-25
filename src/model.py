import torchvision.models as models

def load_pretrained_backbone():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    return model

# if __name__ == "__main__":
#     model = load_pretrained_backbone()
#     print(model.classifier)

import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 7

def load_pretrained_backbone():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    return model

def build_model():
    model = load_pretrained_backbone()

    # Freeze the entire backbone (Step 4.3.2 baseline decision)
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the classifier head for our 7-class task
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=NUM_CLASSES),
    )

    return model

def build_model_partial_finetune():
    model = load_pretrained_backbone()

    # Freeze everything first
    for param in model.features.parameters():
        param.requires_grad = False

    # Then selectively unfreeze the last two stages
    for param in model.features[7].parameters():
        param.requires_grad = True
    for param in model.features[8].parameters():
        param.requires_grad = True

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=NUM_CLASSES),
    )

    return model

def build_model_partial_finetune_v2():
    model = load_pretrained_backbone()

    for param in model.features.parameters():
        param.requires_grad = False

    for stage_idx in [6, 7, 8]:
        for param in model.features[stage_idx].parameters():
            param.requires_grad = True

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=NUM_CLASSES),
    )

    return model

if __name__ == "__main__":
    model = build_model()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print(f"\nNew classifier:\n{model.classifier}")