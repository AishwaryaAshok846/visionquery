import os
import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel
import chromadb
import subprocess
import shutil
import sys
import time
import gc
import hashlib

st.title("VisionQuery: Semantic Video Search")

# Initialize session state variables
if "results" not in st.session_state:
    st.session_state.results = None
if "video_indexed" not in st.session_state:
    st.session_state.video_indexed = False
if "current_video_name" not in st.session_state:
    st.session_state.current_video_name = None
if "video_hash" not in st.session_state:
    st.session_state.video_hash = None
if "show_jump_video" not in st.session_state:
    st.session_state.show_jump_video = False
if "jump_time" not in st.session_state:
    st.session_state.jump_time = 0
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None


# Load CLIP model (cached so it only loads once)
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_model()

# Video upload
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_video:
    os.makedirs("data/videos", exist_ok=True)
    video_path = "data/videos/uploaded_video.mp4"
    
    # Create a unique hash for this video to detect when a new one is uploaded
    video_bytes = uploaded_video.getvalue()
    current_hash = hashlib.md5(video_bytes).hexdigest()[:8]
    
    # Check if this is a different video than what's currently loaded
    if st.session_state.video_hash != current_hash:
        # Store the old collection name so we can delete it
        old_collection = st.session_state.collection_name
        
        # Update session state with new video info
        st.session_state.video_hash = current_hash
        st.session_state.current_video_name = uploaded_video.name
        st.session_state.collection_name = f"video_{current_hash}"
        st.session_state.video_indexed = False
        st.session_state.results = None
        st.session_state.show_jump_video = False
        
        # Delete the old collection from ChromaDB
        if old_collection:
            try:
                client = chromadb.PersistentClient(path="data/chroma")
                client.delete_collection(name=old_collection)
                del client
            except:
                pass
        
        # Clean up memory and old files
        gc.collect()
        time.sleep(0.3)
        
        if os.path.exists("data/frames"):
            shutil.rmtree("data/frames")
            time.sleep(0.3)
        
        # Save the new video file
        with open(video_path, "wb") as f:
            f.write(video_bytes)
        
        # Save collection name to a file so index_frames.py can read it
        with open("data/.collection_name", "w") as f:
            f.write(st.session_state.collection_name)
        
        st.success(f"New video '{uploaded_video.name}' uploaded.")
        st.rerun()
    
    # Display the video (either from start or from a jumped-to timestamp)
    if st.session_state.show_jump_video:
        st.video(video_path, start_time=st.session_state.jump_time)
    else:
        st.video(video_path)
    
    # Index the video if it hasn't been indexed yet
    if not st.session_state.video_indexed:
        with st.spinner("Extracting and indexing frames... This may take a moment."):
            try:
                os.makedirs("data/frames", exist_ok=True)
                
                # Step 1: Extract frames from video
                result = subprocess.run(
                    [sys.executable, "extract_frames.py"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    st.error(f"Frame extraction failed: {result.stderr}")
                else:
                    # Step 2: Generate embeddings and index frames
                    result = subprocess.run(
                        [sys.executable, "index_frames.py"],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode != 0:
                        st.error(f"Indexing failed: {result.stderr}")
                    else:
                        st.session_state.video_indexed = True
                        st.success("Video indexed successfully!")
                        st.rerun()
            except Exception as e:
                st.error(f"Error during indexing: {str(e)}")

# Search interface (only show if video is indexed)
if st.session_state.video_indexed and uploaded_video and st.session_state.collection_name:
    # Connect to ChromaDB and get the collection for this video
    client = chromadb.PersistentClient(path="data/chroma")
    
    try:
        collection = client.get_collection(name=st.session_state.collection_name)
    except Exception as e:
        st.error(f"Could not load collection: {str(e)}")
        st.info("Try refreshing the page or re-uploading the video.")
        st.stop()

    # Search input
    query = st.text_input("Search for a moment in the video:")

    # Perform search when button is clicked
    if st.button("Search") and query:
        st.session_state.show_jump_video = False
        
        # Generate text embedding from search query
        inputs = processor(text=query, return_tensors="pt")
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Search the collection for similar frames (get top 5)
        st.session_state.results = collection.query(
            query_embeddings=[text_features[0].tolist()],
            n_results=5
        )

    results = st.session_state.results

    # Display results
    if results:
        metas = results["metadatas"][0]
        distances = results["distances"][0] if "distances" in results else None

        if metas:
            # Filter out poor matches (below 25% similarity)
            filtered_results = []
            for idx, meta in enumerate(metas):
                if distances and idx < len(distances):
                    similarity = 1 - distances[idx]
                    # Only include results with reasonable similarity
                    if similarity >= 0.25:
                        filtered_results.append((idx, meta, similarity))
            
            # Show results only if we have good matches
            if filtered_results:
                if len(filtered_results) > 1:
                    st.subheader(f"Found {len(filtered_results)} matches")
                else:
                    st.subheader("Top match")
                
                for result_idx, (idx, meta, similarity) in enumerate(filtered_results):
                    ts = float(meta["timestamp"])
                    jump_ts = max(0.0, ts - 0.3)
                    
                    col_a, col_b = st.columns([3, 1])
                    
                    with col_a:
                        st.write(f"**{result_idx + 1}. Timestamp:** {ts:.2f}s (similarity: {similarity:.2%})")
                    
                    with col_b:
                        if st.button(f"Jump to {ts:.2f}s", key=f"jump_{idx}"):
                            st.session_state.show_jump_video = True
                            st.session_state.jump_time = int(jump_ts)
                            st.rerun()
                    
                    # Show caption for the currently playing jump
                    if st.session_state.show_jump_video and result_idx == 0:
                        st.caption(f"Showing from ~{jump_ts:.2f}s (keyframe-safe). Exact match: {ts:.2f}s")
            else:
                st.write("No close matches found. Try a different search term.")
        else:
            st.write("No matches.")
elif uploaded_video and not st.session_state.video_indexed:
    st.info("Please wait while the video is being indexed...")
else:
    st.info("Please upload a video to begin.")