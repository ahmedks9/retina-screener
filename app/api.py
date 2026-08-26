import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import matplotlib.cm as cm

from src.predict import predict

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

TEMP_UPLOAD_DIR = APP_DIR / "temp_uploads"
TEMP_UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def make_overlay_image(preprocessed_tensor, heatmap):
    img_array = preprocessed_tensor.permute(1, 2, 0).cpu().numpy()
    img_array = img_array * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_array = np.clip(img_array, 0, 1)

    heatmap_colored = cm.jet(heatmap)[:, :, :3]
    overlay = 0.5 * img_array + 0.5 * heatmap_colored
    overlay = np.clip(overlay, 0, 1)

    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
    return overlay_img

from fastapi import HTTPException

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    temp_path = TEMP_UPLOAD_DIR / f"{uuid.uuid4()}.jpg"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        result = predict(str(temp_path))

        overlay_img = make_overlay_image(result["preprocessed_image"], result["gradcam_heatmap"])
        overlay_filename = f"{uuid.uuid4()}.png"
        overlay_img.save(STATIC_DIR / overlay_filename)

        return {
            "predicted_class": result["predicted_class"],
            "class_probabilities": result["class_probabilities"],
            "gradcam_overlay_url": f"/static/{overlay_filename}",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {str(e)}")
    finally:
        temp_path.unlink()