import cv2
import os

VIDEO_PATH = "data/videos/uploaded_video.mp4"
OUTPUT_DIR = "data/frames"
TARGET_FPS = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

frame_interval = max(1, int(round(fps / TARGET_FPS)))

frame_count = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    if frame_count % frame_interval == 0:
        timestamp_sec = frame_count / fps
        timestamp_ms = int(round(timestamp_sec * 1000))

        filename = f"frame_{timestamp_ms:08d}ms.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(filepath, frame)

    frame_count += 1

cap.release()
print("Done extracting frames.")