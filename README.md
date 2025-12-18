# VisionQuery: Semantic Video Search

VisionQuery is a video search tool enabling you to find specific moments in videos using natural language queries. Upload a video, type what you're looking for, and instantly jump to the matching timestamp.

---

## Features

- **Natural Language Search**: Search for moments using plain English (e.g., "person cycling", "someone drinking coffee").
- **Multiple Results**: Automatically finds and displays relevant matches (filtered by similarity).
- **Smart Filtering**: Displays results above a 25% similarity threshold to ensure accuracy.
- **Quick Navigation**: Jump directly to any matched timestamp with a single click.
- **Multi-Video Support**: Seamlessly switch between uploaded videos without restarting.
- **Automatic Cleanup**: Removes old data and ensures the latest video is efficiently indexed.

---

## Requirements

- Python 3.7 or higher
- At least 2GB RAM (4GB recommended for processing longer videos)
- Works on CPU—no GPU needed!

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AishwaryaAshok846/visionquery.git
cd visionquery
```

### 2. Install Dependencies

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

Dependencies:
- **`streamlit`**: Interactive web interface for video playback and querying.
- **`torch`**: PyTorch for deep learning (used for generating embeddings).
- **`transformers`**: Hugging Face toolkit to access the CLIP model.
- **`chromadb`**: Vector database for similarity search.
- **`opencv-python`**: For processing the video and extracting frames.
- **`pillow`**: Image processing during indexing.

**Note**: Installation might take 5-10 minutes, as the CLIP model is downloaded.

---

## Usage

### Step 1: Start the Application

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default browser at [http://localhost:8501](http://localhost:8501). If it doesn’t, manually navigate to this URL.

### Step 2: Upload Your Video

- Supported formats: **MP4, MOV, AVI**.
- The video will be processed automatically.

### Step 3: Search for Moments

1. Enter a descriptive query (e.g., "person typing on laptop").
2. Click "Search."
3. View the results ranked by similarity, along with timestamps.
4. Click "Jump to X.XXs" to navigate directly to the scene.

### Advanced Configurations

#### Adjusting Frame Sampling Rate
By default, frames are extracted at **2 FPS** in `extract_frames.py`. To change this:
1. Open `extract_frames.py`.
2. Modify the `TARGET_FPS` variable:
    ```python
    TARGET_FPS = <your_desired_fps>
    ```
3. Higher FPS results in better search precision but increases indexing time.

#### Optimizing Search Result Filtering
The similarity threshold for filtering search results is **25%**. To adjust this:
1. Open `app.py`.
2. Modify the filtering condition in the results section:
    ```python
    if similarity >= <your_desired_threshold>:
    ```

---

## How It Works

1. **Frame Extraction**:  
   - Extracts frames at regular intervals (default: 2 frames per second).  
   - Saves extracted frames in the `data/frames` directory.

2. **Embedding Generation**:  
   - Each frame is encoded into a vector using OpenAI CLIP.
   - Text queries are similarly encoded for comparison.

3. **Indexing and Storage**:  
   - Frame embeddings are stored in **ChromaDB**, which enables fast similarity search.

4. **Text-to-Video Search**:  
   - Queries are matched against the embedding database to identify the most relevant frames.

5. **Timestamp Navigation**:  
   - Results include timestamps. Users can jump directly to the relevant scene.

---

## Project Structure

```
visionquery/
├── app.py                 # Main Streamlit application
├── extract_frames.py      # Extracts frames from the video
├── index_frames.py        # Generates embeddings and indexes frames
├── requirements.txt       # Dependency list for Python
├── README.md              # Documentation
├── .gitignore             # Git ignore rules
├── data/                  # Generated at runtime (not in git)
│   ├── videos/            # Uploaded videos
│   ├── frames/            # Extracted frames
│   ├── chroma/            # ChromaDB database files
```

---

## Troubleshooting

### Common Issues

#### Error: "Frame extraction failed"
- Ensure that the uploaded video is not corrupted.
- Supported formats: MP4, MOV, AVI.
- Ensure sufficient disk space is available for saving frames.

#### Error: "Indexing failed"
- Verify that all dependencies are installed (`requirements.txt`).
- Restart the application if the issue persists.

#### Performance is slow
- Longer videos may take time to index (~1 min for 10 minutes of video).
- Reduce the **frames per second (FPS)** in `extract_frames.py` to speed up indexing.

#### Out of memory during processing
- Process shorter videos or reduce FPS.
- Close unnecessary applications.
- For large-scale processing, upgrade to a machine with more RAM.

---

## Limitations

- **Abstract Queries**: CLIP’s understanding of your query is limited to how well it aligns with frame content.
- **Short Moments**: Very brief actions (<0.5s) may not be picked up due to frame interval sampling.
- **Re-indexing Required**: A new video upload clears previous indexed data to maintain focus on the current video.

---

## Future Enhancements

- Audio transcription for speech-based queries.
- Batch video indexing to process multiple videos simultaneously.
- Fine-tuned CLIP models for specific domains (e.g., surveillance, sports).
- Export functionality for matched timestamps.
- Custom user-defined frame sampling rates.

---

## Credits

Built with:
- [Streamlit](https://streamlit.io/) - Interactive web application framework.
- [OpenAI CLIP](https://github.com/openai/CLIP) - Vision-language model for semantic understanding.
- [ChromaDB](https://www.trychroma.com/) - Fast and scalable vector database.
- [OpenCV](https://opencv.org/) - Library for video and image processing.

---

VisionQuery is an evolving project. Contributions and ideas are welcome! Fork the repository and submit a pull request to enhance its functionality.

---
