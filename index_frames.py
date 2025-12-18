import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import chromadb
import gc

FRAMES_DIR = "data/frames"
CHROMA_DIR = "data/chroma"

# Read the collection name that was set by the main app
if not os.path.exists("data/.collection_name"):
    print("ERROR: No collection name file found!")
    exit(1)

with open("data/.collection_name", "r") as f:
    collection_name = f.read().strip()

# Get list of all frame files
frame_files = [f for f in sorted(os.listdir(FRAMES_DIR)) if f.endswith(".jpg")]

if len(frame_files) == 0:
    print("ERROR: No frames found!")
    exit(1)

# Make sure ChromaDB directory exists
os.makedirs(CHROMA_DIR, exist_ok=True)

# Load the CLIP model for generating embeddings
model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    use_safetensors=True
)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Connect to ChromaDB
client = chromadb.PersistentClient(path=CHROMA_DIR)

# Delete any existing collection with this name (in case of reprocessing)
try:
    client.delete_collection(name=collection_name)
except:
    pass

# Create a fresh collection for this video
collection = client.create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}
)

# Process each frame and add to collection
frame_count = 0
for filename in frame_files:
    # Load the image
    image_path = os.path.join(FRAMES_DIR, filename)
    image = Image.open(image_path).convert("RGB")
    
    # Generate CLIP embedding for this frame
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    # Normalize the embedding
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # Extract timestamp from filename (format: frame_00012345ms.jpg)
    ts_ms = int(filename.replace("frame_", "").replace("ms.jpg", ""))
    ts_sec = ts_ms / 1000.0
    
    # Add frame to ChromaDB collection
    collection.add(
        embeddings=[image_features[0].tolist()],
        metadatas=[{"timestamp": ts_sec, "frame": filename}],
        ids=[f"{collection_name}_{filename}"]
    )
    
    frame_count += 1

print(f"Successfully indexed {frame_count} frames into ChromaDB.")