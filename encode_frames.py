import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

FRAMES_DIR = "data/frames"

device = "cpu"

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    use_safetensors=True
).to(device)

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

embeddings = []

for filename in sorted(os.listdir(FRAMES_DIR)):
    if not filename.endswith(".jpg"):
        continue

    image_path = os.path.join(FRAMES_DIR, filename)
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    embeddings.append({
        "frame": filename,
        "embedding": image_features[0].tolist()
    })

print(f"Encoded {len(embeddings)} frames.")