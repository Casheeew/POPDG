import os
import random
import pandas as pd

CSV_FILE = "analysis_results_english.csv"  # your CSV file
OUTPUT_DIR = "splits"
TRAIN_RATIO = 0.95  # 80% train, 20% test

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load CSV
df = pd.read_csv(CSV_FILE)

# Extract all video patterns
video_patterns = df[["Strategy_Number", "Video_Pattern", "Selected_Segment"]].drop_duplicates().values.tolist()

# Shuffle and split
random.shuffle(video_patterns)
split_idx = int(len(video_patterns) * TRAIN_RATIO)
train_patterns = video_patterns[:split_idx]
test_patterns = video_patterns[split_idx:]

# Convert to paths relative to dataset (dataset/{strat}/{dance})
def to_rel_path(strat, pattern, segment):
    strat = str(strat)
    if segment == "Full":
        f = f"{pattern}_strategy_{strat}"
    else:
        f = f"{pattern}_segment_{segment}_strategy_{strat}"

    return os.path.join(strat, f)

train_lines = [to_rel_path(strat, pattern, segment) for strat, pattern, segment in train_patterns]
test_lines = [to_rel_path(strat, pattern, segment) for strat, pattern, segment in test_patterns]

# Save to text files
with open(os.path.join(OUTPUT_DIR, "train.txt"), "w") as f:
    f.write("\n".join(train_lines))

with open(os.path.join(OUTPUT_DIR, "test.txt"), "w") as f:
    f.write("\n".join(test_lines))

print(f"Total: {len(video_patterns)}, Train: {len(train_patterns)}, Test: {len(test_patterns)}")
print("Splits saved in 'splits/' folder.")
