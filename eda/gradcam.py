import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.inference_utils import load_trained_model
from src.dataset import CLASS_TO_IDX
from src.transforms import preprocess_cached

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_trained_model("models/partial_ft_v2_raw_best.pt", device)

activations = {}
gradients = {}

def forward_hook(module, input, output):
    activations["value"] = output.detach()

def backward_hook(module, grad_input, grad_output):
    gradients["value"] = grad_output[0].detach()

target_layer = model.features[8]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

def generate_gradcam(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess_cached(img).unsqueeze(0).to(device)
    tensor.requires_grad_()

    output = model(tensor)
    pred_class = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, pred_class].backward()

    acts = activations["value"][0]
    grads = gradients["value"][0]

    weights = grads.mean(dim=(1, 2))
    cam = torch.zeros(acts.shape[1:], device=device)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = F.relu(cam)
    cam = cam / (cam.max() + 1e-8)
    cam = cam.cpu().numpy()

    cam_resized = np.array(Image.fromarray(cam).resize((224, 224), Image.BILINEAR))

    return cam_resized, IDX_TO_CLASS[pred_class]