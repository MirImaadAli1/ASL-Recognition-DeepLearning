import os
import shutil
from multiprocessing import Pool, Manager
from tqdm import tqdm

def get_folders(path):
    """Get a list of folder names in a directory."""
    return {folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))}

def is_folder_empty(path):
    """Check if a folder is empty."""
    return not any(os.scandir(path))

def move_files(source, destination, progress_queue):
    """Move files from source to destination if destination is empty."""
    if os.path.exists(source) and os.path.exists(destination) and is_folder_empty(destination):
        for file in os.listdir(source):
            shutil.move(os.path.join(source, file), os.path.join(destination, file))
        result = f"Moved contents from {source} to {destination}"
    else:
        result = f"Skipping {destination}, not empty or source missing."

    progress_queue.put(1)
    return result

def worker_init(q):
    global progress_queue
    progress_queue = q

def check_and_move(source_root, dest_root):
    source_folders = get_folders(source_root)
    dest_folders = get_folders(dest_root)
    
    tasks = [
        (os.path.join(source_root, folder), os.path.join(dest_root, folder))
        for folder in dest_folders if folder in source_folders
    ]

    with Manager() as manager:
        progress_queue = manager.Queue()
        with Pool(processes=8, initializer=worker_init, initargs=(progress_queue,)) as pool:
            results = pool.starmap_async(move_files, [(src, dst, progress_queue) for src, dst in tasks])

            with tqdm(total=len(tasks), desc="Moving Files", unit="folder") as pbar:
                for _ in range(len(tasks)):
                    progress_queue.get()
                    pbar.update(1)

            results = results.get()

    for res in results:
        print(res)

if __name__ == "__main__":
    SOURCE_DIR = "/mnt/c/videos"
    DEST_DIR = "./videos"
    check_and_move(SOURCE_DIR, DEST_DIR)
