import os
import shutil
from glob import glob

# Source and destination
source_root = "data/test"
destination_dir = 'eval_data'

# # Uncomment if reference data (simplified ground truth)
# destination_dir = "eval_data_ref"

# Ensure destination directory exists
os.makedirs(destination_dir, exist_ok=True)


# Define search patterns
file_patterns = [
    # # Uncomment if reference data (simplified ground truth)
    # ("motions", "*_simplified.npy"),
    ("motions", "*_original.npy"),
    ("wavs", "*_original.wav"),
]

total_copied = 0

for subdir, pattern in file_patterns:
    search_path = os.path.join(source_root, "*", subdir, pattern)
    files = glob(search_path)
    
    print(f"Found {len(files)} files in {subdir}...")

    for filepath in files:
        # Extract strat and original filename
        parts = filepath.split(os.sep)
        strat = parts[1]  # test/{strat}/...
        filename = os.path.basename(filepath)

                
        # # Uncomment if reference data (simplified ground truth)
                
        # # Modify filename for .wav files
        # if filename.endswith("_original.wav"):
        #     filename = filename.replace("_original.wav", "_simplified.wav")


        # Compose a unique filename: {strat}_{filename}
        new_filename = f"{strat}_{filename}"
        destination_path = os.path.join(destination_dir, new_filename)

        # Copy the file
        shutil.copy2(filepath, destination_path)
        print(f"Copied: {filepath} → {destination_path}")
        total_copied += 1

print(f"✅ Done. Total files copied: {total_copied} [target: 44 (22 .npy, 22 .wav)]")
