import os
import torch
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image

# ⬇️ Import actual U2NETP architecture
from model.u2net import U2NETP

# Set paths
input_path = os.path.join("input", "input.jpg")
output_path = os.path.join("output", "output.png")
model_path = os.path.join("model", "u2netp.pth")

# Check input image exists
if not os.path.exists(input_path):
    print("❌ Put an image at input/input.jpg first!")
    exit()

# Load the model
print("🔄 Loading U2NETP model...")
net = U2NETP(3, 1)
net.load_state_dict(torch.load(model_path, map_location='cpu'))
net.eval()
print("✅ Model loaded.")

# Preprocess input image
print("🖼️ Preprocessing input image...")
image = Image.open(input_path).convert("RGB")
ori_size = image.size

transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])
image_tensor = transform(image).unsqueeze(0)
image_tensor = Variable(image_tensor)

# Run inference
print("🤖 Running inference...")
with torch.no_grad():
    d1, _, _, _, _, _, _ = net(image_tensor)
    pred = d1[:, 0, :, :]
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    pred_np = pred.squeeze().cpu().numpy()

# Resize and post-process mask
print("🎨 Applying mask...")
mask = Image.fromarray((pred_np * 255).astype(np.uint8)).resize(ori_size, Image.BILINEAR)
image = image.convert("RGBA")
mask = np.array(mask) / 255.0
image_np = np.array(image)

# Apply mask to alpha channel
image_np[..., 3] = (mask * 255).astype(np.uint8)
result = Image.fromarray(image_np)

# Save result
result.save(output_path)
print(f"✅ Background removed and saved to: {output_path}")
