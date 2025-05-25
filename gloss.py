import numpy as np

# Load the original gloss dictionary
gloss_dict = np.load('./preprocess/how2sign/gloss_dict.npy', allow_pickle=True).item()

# Print its size
print("Gloss dictionary size:", len(gloss_dict))

# Print sample entries
print("Sample keys:", list(gloss_dict.keys()))
