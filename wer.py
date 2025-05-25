import pandas as pd
import jiwer

# Load CSV
df = pd.read_csv("seen_sentences.csv")

# Compute WER for each row
wer_scores = []
for _, row in df.iterrows():
    ground_truth = str(row["ground_truth"]).lower().strip()
    predicted = str(row["prediction"]).lower().strip()
    
    wer = jiwer.wer(ground_truth, predicted)  # Compute WER
    wer_scores.append(wer)

# Compute overall average WER
average_wer = sum(wer_scores) / len(wer_scores)
print(f"Average WER: {average_wer * 100:.2f}%")

