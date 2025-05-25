import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_data(input_csv, output_folder, total_videos=288, train_ratio=0.7):
    # Load the labels CSV file
    df = pd.read_csv(input_csv)


    if len(df) < total_videos:
        raise ValueError(f"Dataset has only {len(df)} videos but needs at least {total_videos}.")

    # seen and unseen
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

    # Ensure must-have videos are in the dataset
    if not set(must_include).issubset(set(df["sentence_name"])):
        raise ValueError("Some must-have videos are missing from the dataset.")

    # Remove must-exclude videos
    df = df[~df["sentence_name"].isin(must_exclude)]

    # Separate must-include videos
    must_include_df = df[df["sentence_name"].isin(must_include)]

    # Select remaining videos excluding must-include ones
    remaining_df = df[~df["sentence_name"].isin(must_include)]

    # Ensure enough videos remain after exclusion
    remaining_needed = total_videos - len(must_include)
    if len(remaining_df) < remaining_needed:
        raise ValueError(f"Not enough remaining videos after exclusions. Needed {remaining_needed}, found {len(remaining_df)}.")

    # Randomly sample the remaining videos
    selected_remaining_df = remaining_df.sample(n=remaining_needed, random_state=42)

    # Combine must-include videos with selected remaining ones
    selected_df = pd.concat([must_include_df, selected_remaining_df])

    # Ensure must-include videos are correctly placed
    if not set(must_include).issubset(set(selected_df["sentence_name"])):
        raise ValueError("Must-include videos were not selected correctly.")

    # Allocate must-include videos to train set first
    train_must_include_df = must_include_df
    remaining_train_needed = int(total_videos * train_ratio) - len(train_must_include_df)

    if remaining_train_needed < 0:
        raise ValueError("Too many must-include videos for the train set. Adjust total_videos or must_include list.")

    # Split the remaining videos into train and remaining (dev + test)
    remaining_videos_df = selected_remaining_df.sample(n=remaining_train_needed, random_state=42)
    train_df = pd.concat([train_must_include_df, remaining_videos_df])

    # Split remaining videos into dev and test (15% each)
    dev_test_df = selected_remaining_df.drop(remaining_videos_df.index)
    dev_df, test_df = train_test_split(dev_test_df, test_size=0.5, random_state=42)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Save the splits
    train_df.to_csv(os.path.join(output_folder, 'train.csv'), index=False)
    dev_df.to_csv(os.path.join(output_folder, 'dev.csv'), index=False)
    test_df.to_csv(os.path.join(output_folder, 'test.csv'), index=False)

    print(f"Data split into train ({len(train_df)} videos), dev ({len(dev_df)} videos), and test ({len(test_df)} videos). Files saved to {output_folder}")

# Path to the labels.csv file
labels_csv_path = './filtered_labels.csv'
output_folder = './videos/splits'  # Directory where the split files will be saved

# Split the data
split_data(labels_csv_path, output_folder)
