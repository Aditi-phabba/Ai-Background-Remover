import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import gdown  # ✅ Add this line
from model.u2net import U2NET


# ✅ Add this function to auto-download the model if missing
def download_model():
    model_dir = "model"
    model_path = os.path.join(model_dir, "u2net.pth")
    if not os.path.exists(model_path):
        st.info("📥 Downloading U²-Net model (first time only)...")
        url = "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ"
        os.makedirs(model_dir, exist_ok=True)
        gdown.download(url, model_path, quiet=False)
    else:
        print("✅ Model already exists.")


# Load model only once
@st.cache_resource
def load_model():
    download_model()  # ✅ Call download before loading
    model_path = os.path.join("model", "u2net.pth")
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()
    return net


def remove_bg(image, model):
    orig_size = image.size
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        d1, *_ = model(image_tensor)
        pred = d1[:, 0, :, :]
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        mask = pred.squeeze().cpu().numpy()

    mask = Image.fromarray((mask * 255).astype(np.uint8)).resize(orig_size)
    image = image.convert("RGBA")
    mask = np.array(mask) / 255.0
    img_np = np.array(image)
    img_np[..., 3] = (mask * 255).astype(np.uint8)
    return Image.fromarray(img_np)


# Streamlit UI
st.set_page_config(page_title="AI Background Remover", layout="centered")
st.title("🪄 AI Background Remover")
st.write("Upload an image and remove its background using U²-Net.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original", use_column_width=True)

    with st.spinner("Removing background..."):
        model = load_model()
        result = remove_bg(image, model)

    st.image(result, caption="Background Removed", use_column_width=True)

    st.download_button("📥 Download Result", data=result.tobytes(), file_name="no_bg.png", mime="image/png")
