import os
import cv2
import multiprocessing
from tqdm import tqdm

VIDEO_DIR = "./videos"  # Root directory where folders are located

def check_first_image(folder_name):
    """Check the first image in the folder. If corrupted, return folder_name."""
    folder_path = os.path.join(VIDEO_DIR, folder_name)

    if not os.path.isdir(folder_path):
        return None  # Skip if it's not a folder

    images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")])
    if not images:
        return folder_name  # No images found, mark as corrupted

    first_image_path = os.path.join(folder_path, images[0])
    img = cv2.imread(first_image_path)
    if img is None:
        return folder_name  # Mark as corrupted

    return None  # Folder is valid

if __name__ == "__main__":
    # Get all subfolders in ./videos
    folder_names = [f for f in os.listdir(VIDEO_DIR) if os.path.isdir(os.path.join(VIDEO_DIR, f))]

    # Use multiprocessing with tqdm progress bar
    corrupted_folders = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for result in tqdm(pool.imap(check_first_image, folder_names), total=len(folder_names), desc="Checking folders"):
            if result:
                corrupted_folders.append(result)

    # Save corrupted folders
    with open("corrupted_folders.txt", "w") as cf:
        cf.writelines("\n".join(corrupted_folders) + "\n")

    # Save valid folders
    valid_folders = set(folder_names) - set(corrupted_folders)
    with open("valid_folders.txt", "w") as vf:
        vf.writelines("\n".join(valid_folders) + "\n")

    print(f"✅ Done! Corrupted: {len(corrupted_folders)}, Valid: {len(valid_folders)}")
