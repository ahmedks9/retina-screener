import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Retinal Disease Screening", layout="centered")
st.title("Retinal Disease Screening")
st.write("Upload a fundus image to get an AI-assisted screening prediction.")

uploaded_file = st.file_uploader("Upload a fundus image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Run Prediction"):
        with st.spinner("Analyzing image..."):
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()

            st.subheader(f"Predicted class: {result['predicted_class']}")

            st.write("Class probabilities:")
            st.bar_chart(result["class_probabilities"])

            overlay_url = f"http://127.0.0.1:8000{result['gradcam_overlay_url']}"
            overlay_response = requests.get(overlay_url)
            overlay_image = Image.open(io.BytesIO(overlay_response.content))

            st.write("Model attention (Grad-CAM):")
            st.image(overlay_image, use_container_width=True)
        else:
            error_detail = response.json().get("detail", "Unknown error")
            st.error(f"Prediction failed: {error_detail}")
