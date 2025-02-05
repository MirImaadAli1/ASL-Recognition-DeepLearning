import os
import json
import io
import cv2
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from PIL import Image
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor

# Google Cloud Storage bucket details
BUCKET_NAME = "clsr-storage"
GCS_OUTPUT_PATH = "processed_wlasl_data/"  # Path in your GCS bucket

# Initialize the Google Cloud Storage client
client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)

# Paths for data
data_dir = "wlasl/videos"
json_path = "wlasl/WLASL_v0.3.json" 

def load_wlasl_metadata(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    return frames

def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])
    tensor = transform(Image.fromarray(frame))
    tensor = (tensor * 255).byte()  # Convert to uint8
    return tensor

def upload_to_gcs(np_frames, video_id):
    # Create an in-memory byte stream to save the npz file
    byte_stream = io.BytesIO()
    
    # Save to the in-memory byte stream as .npz
    np.savez_compressed(byte_stream, frames=np_frames)
    
    # Move to the beginning of the byte stream before uploading
    byte_stream.seek(0)
    
    # Path for storing in GCS
    gcs_path = f"processed_wlasl_data/{video_id}.npz"  # Path in GCS for .npz
    
    # Upload the byte stream to GCS
    blob = bucket.blob(gcs_path)
    blob.upload_from_file(byte_stream)  # Upload from in-memory byte stream
    return gcs_path

def process_video_entry(entry):
    gloss = entry['gloss']
    dataset = []

    for instance in entry['instances']:
        video_id = instance['video_id']
        video_path = os.path.join(data_dir, f"{video_id}.mp4")
        
        if not os.path.exists(video_path):
            continue
        
        # Extract frames and preprocess
        frames = extract_frames(video_path, num_frames=16)
        processed_frames = [preprocess_frame(f) for f in frames]
        
        # Convert processed frames to numpy array
        np_frames = np.stack(processed_frames)
        
        # Upload to GCS and get the path
        gcs_path = upload_to_gcs(np_frames, video_id)
        
        dataset.append((gcs_path, gloss))
    
    return dataset

def process_videos_parallel(metadata, num_frames=16, max_workers=4):
    all_datasets = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Process each video entry in parallel
        futures = [executor.submit(process_video_entry, entry) for entry in metadata]
        
        for future in tqdm(futures, desc="Processing Videos", total=len(futures)):
            result = future.result()  # Get the result from each future
            all_datasets.extend(result)

    return all_datasets

def split_dataset(dataset, train_ratio=0.8):
    random.shuffle(dataset)
    split_idx = int(len(dataset) * train_ratio)
    train_set, test_set = dataset[:split_idx], dataset[split_idx:]
    
    with open("train_set.json", "w") as f:
        json.dump(train_set, f)
    with open("test_set.json", "w") as f:
        json.dump(test_set, f)
    
    print(f"Training set: {len(train_set)} samples")
    print(f"Testing set: {len(test_set)} samples")

if __name__ == "__main__":
    metadata = load_wlasl_metadata(json_path)
    dataset = process_videos_parallel(metadata, num_frames=16)
    split_dataset(dataset)
