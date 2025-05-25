import numpy as np
import os
import cv2
import torch
import pandas as pd
from utils import video_augmentation
from slr_network import SLRModel
from collections import OrderedDict
from decord import VideoReader, cpu
import argparse

VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]

# Define seen and unseen sentence sets
must_include = [
    "bHEw7t7tJsc_25-5-rgb_front", "1DgRPcY65Mk_9-5-rgb_front", "BbBdxWO88-I_27-8-rgb_front",
    "1WZGAIBhaPs_11-5-rgb_front", "3ntc51P5qjI_10-8-rgb_front", "3WEwTZHf8kI_5-5-rgb_front",
    "01m9yM04RwY_3-3-rgb_front", "3mjTbgSBlhE_7-5-rgb_front", "032PDxai5GE_13-8-rgb_front",
    "59xX5JTRuk4_1-8-rgb_front"
]

must_exclude = [
    "5dqnMgr62_w_14-8-rgb_front", "bbGZcyIMvCI_16-2-rgb_front", "bHEw7t7tJsc_6-5-rgb_front",
    "1A1mzX1-l9g_2-8-rgb_front", "5CGdJ5Cuv5M_0-8-rgb_front", "1IRJXMx85Rg_7-5-rgb_front",
    "1A1mzX1-l9g_2-8-rgb_front", "3TKkSL9OYAg_8-5-rgb_front", "05snMPwCV98_4-5-rgb_front",
    "0EW2ql96Ksc_9-8-rgb_front"
]

def load_video_frames(video_dir, max_frames_num=360):
    # Get the video name from the folder
    video_name = os.path.basename(video_dir)
    
    # Get all frame files (e.g., frame_0000.jpg, frame_0001.jpg, etc.)
    frame_files = [f for f in os.listdir(video_dir) if f.endswith(".jpg")]
    frame_files.sort()  # Sort files to ensure correct order
    
    total_frame_num = len(frame_files)
    if total_frame_num > max_frames_num:
        frame_idx = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int).tolist()
    else:
        frame_idx = np.linspace(0, total_frame_num - 1, dtype=int).tolist()
    
    frames = []
    for idx in frame_idx:
        frame_path = os.path.join(video_dir, frame_files[idx])
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    return frames

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="Path to pretrained model weights")
parser.add_argument("--video_dir", type=str, help="Path to directory containing videos")
parser.add_argument("--device", type=int, default=0, help="GPU device to use")
parser.add_argument("--max_frames_num", type=int, default=360, help="Max frames per video")

args = parser.parse_args()

device_id = args.device
video_dir = args.video_dir
model_path = args.model_path

# Load gloss dictionary
dict_path = './preprocess/how2sign/gloss_dict.npy'
gloss_dict = np.load(dict_path, allow_pickle=True).item()
# Load the labels CSV into a DataFrame
labels_df = pd.read_csv('filtered_labels.csv')

# Create a dictionary for easy lookup
label_dict = pd.Series(labels_df['label'].values, index=labels_df['sentence_name']).to_dict()


# Model Initialization (keeping kernel sizes and initialization)
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
model = SLRModel(
    num_classes=len(gloss_dict) + 1,
    c2d_type='resnet18',
    conv_type=2,
    use_bn=1,
    gloss_dict=gloss_dict,
    loss_weights={'ConvCTC': 1.0, 'SeqCTC': 1.0, 'Dist': 25.0},
).to(device)

state_dict = torch.load('./work_dir/baseline_res18/dev_100.00_epoch40_model.pt', map_location=device, weights_only=False)['model_state_dict']
state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
model.load_state_dict(state_dict, strict=True)
model.eval()

# Video preprocessing function
def preprocess_video(video_path):
    img_list = load_video_frames(video_path, args.max_frames_num)
    transform = video_augmentation.Compose([
        video_augmentation.CenterCrop(224),
        video_augmentation.Resize(1.0),
        video_augmentation.ToTensor(),
    ])
    
    vid, _ = transform(img_list, None, None)
    vid = vid.float() / 127.5 - 1
    vid = vid.unsqueeze(0)
    
    # Padding
    left_pad, last_stride, total_stride = 0, 1, 1
    kernel_sizes = ['K5', "P2", 'K5', "P2"]
    for ks in kernel_sizes:
        if ks[0] == 'K':
            left_pad = left_pad * last_stride
            left_pad += int((int(ks[1]) - 1) / 2)
        elif ks[0] == 'P':
            last_stride = int(ks[1])
            total_stride *= last_stride
    
    max_len = vid.size(1)
    video_length = torch.LongTensor([np.ceil(max_len / total_stride) * total_stride + 2 * left_pad])
    right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
    max_len = max_len + left_pad + right_pad
    vid = torch.cat((
        vid[0, 0][None].expand(left_pad, -1, -1, -1),
        vid[0],
        vid[0, -1][None].expand(max_len - vid.size(1) - left_pad, -1, -1, -1),
    ), dim=0).unsqueeze(0)
    
    return vid.to(device), video_length.to(device)


# Process only the specific videos in must_include and must_exclude
seen_results = []
unseen_results = []

# Combine the must_include and must_exclude lists for targeted processing
target_videos = must_include + must_exclude

for video_name in target_videos:
    print(f"Processing file: {video_name}")  # Print the current file being processed

    video_path = os.path.join(video_dir, video_name)
    print(f"Video path: {video_path}, Video name: {video_name}")  # Print the video path and name

    try:
        # Check if the directory for frames exists
        if not os.path.isdir(video_path):
            print(f"Directory {video_path} does not exist!")
            continue

        # Preprocess the video frames
        vid, vid_lgt = preprocess_video(video_path)
        print(f"Preprocessed video: {video_name}")  # Print after video is preprocessed
        
        # Predict using the model
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt, label=None, label_lgt=None)
        
        conv_predicted_sentence = ret_dict['conv_sents'][0]
        predicted_glosses = " ".join(conv_predicted_sentence)
        print(f"Predicted glosses: {predicted_glosses}")  
        # Check if the predicted gloss exceeds the length of the ground truth and trim if necessary
        ground_truth = label_dict.get(video_name, "UNKNOWN")  # Get ground truth for the video

        # Split the ground truth into words and ensure trimming
        ground_truth_words = ground_truth.split()  # Convert ground truth to list of words
        print(f"ground truth split", ground_truth_words)
        if len(predicted_glosses.split()) > len(ground_truth_words):
            predicted_glosses = " ".join(predicted_glosses.split()[:len(ground_truth_words)])

        result = {"video_name": video_name, 
                  "ground_truth": ground_truth, 
                  "prediction": predicted_glosses}

        # Categorize the results
        if video_name in must_include:
            seen_results.append(result)
            print(f"Added {video_name} to seen results")
        elif video_name in must_exclude:
            unseen_results.append(result)
            print(f"Added {video_name} to unseen results")

    except Exception as e:
        print(f"Error processing {video_name}: {e}")  


# Save results
seen_df = pd.DataFrame(seen_results)
unseen_df = pd.DataFrame(unseen_results)

seen_df.to_csv("seen_sentences.csv", index=False)
unseen_df.to_csv("unseen_sentences.csv", index=False)

print("Processing complete. Results saved.")
