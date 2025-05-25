import os
import pandas as pd

def generate_label_csv(video_folder, label_csv):
    # Load the existing label CSV file
    label_df = pd.read_csv(label_csv, delimiter='\t')  # for tab-separated files
    data = []

    # Traverse the directory structure to get sentence names
    for sentence_name in os.listdir(video_folder):
        sentence_folder = os.path.join(video_folder, sentence_name)

        if os.path.isdir(sentence_folder):
            # Find the corresponding label for the sentence_name
            label_row = label_df[label_df['SENTENCE_NAME'] == sentence_name]
            
            if not label_row.empty:
                label = label_row.iloc[0]['SENTENCE']  # Assuming the column is 'SENTENCE'
                
                # Filter based on the number of glosses (words)
                gloss_count = len(label.split())  # Counting words in the label
                if gloss_count < 10:  # Only include sentences with fewer than 10 glosses
                    data.append([label, sentence_name])
    
    # Create a DataFrame and save it as CSV
    df = pd.DataFrame(data, columns=['label', 'sentence_name'])
    df.to_csv('filtered_labels.csv', index=False)
    print("Generated filtered_labels.csv with sentences containing fewer than 10 glosses.")

video_folder = './videos'
label_csv = './how2sign_realigned_train.csv' 
generate_label_csv(video_folder, label_csv)
