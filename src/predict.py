import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from src.inference_utils import load_trained_model
from src.dataset import CLASS_TO_IDX
from src.transforms import preprocess_image

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "partial_ft_v2_raw_best.pt"

model = load_trained_model(str(CHECKPOINT_PATH), device)

_activations = {}
_gradients = {}

def _forward_hook(module, input, output):
    _activations["value"] = output.detach()

def _backward_hook(module, grad_input, grad_output):
    _gradients["value"] = grad_output[0].detach()

model.features[8].register_forward_hook(_forward_hook)
model.features[8].register_full_backward_hook(_backward_hook)


def predict(image_path: str):
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess_image(img).unsqueeze(0).to(device)
    tensor.requires_grad_()

    output = model(tensor)
    probs = F.softmax(output, dim=1)[0]
    pred_idx = output.argmax(dim=1).item()
    pred_class = IDX_TO_CLASS[pred_idx]

    model.zero_grad()
    output[0, pred_idx].backward()

    acts = _activations["value"][0]
    grads = _gradients["value"][0]
    weights = grads.mean(dim=(1, 2))

    cam = torch.zeros(acts.shape[1:], device=device)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = F.relu(cam)
    cam = cam / (cam.max() + 1e-8)
    cam = cam.cpu().numpy()
    cam_resized = np.array(Image.fromarray(cam).resize((224, 224), Image.BILINEAR))

    class_probs = {IDX_TO_CLASS[i]: round(probs[i].item(), 4) for i in range(len(CLASS_TO_IDX))}

    return {
        "predicted_class": pred_class,
        "class_probabilities": class_probs,
        "gradcam_heatmap": cam_resized,
        "preprocessed_image": preprocess_image(img),
    }
