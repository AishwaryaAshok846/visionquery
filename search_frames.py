import torch
from transformers import CLIPProcessor, CLIPModel
import chromadb

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    use_safetensors=True
)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_or_create_collection(name="video_frames")

query_text = input("Search query: ")

inputs = processor(text=query_text, return_tensors="pt")

with torch.no_grad():
    text_features = model.get_text_features(**inputs)

text_features = text_features / text_features.norm(dim=-1, keepdim=True)

results = collection.query(
    query_embeddings=[text_features[0].tolist()],
    n_results=5
)

print("\nRaw result object:")
print(results)

print("\nTop matches:")
for metadata in results["metadatas"][0]:
    print(f"Timestamp: {metadata['timestamp']}s")