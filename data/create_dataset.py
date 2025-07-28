import argparse
import os
from pathlib import Path

from audio_extraction.baseline_features import process_audio_directory as baseline_extract
from audio_extraction.jukebox_features import process_audio_directory as jukebox_extract
from filter_split_data import split_data
from slice import slice_and_clean

def create_dataset(options):
    """
    Creates a dataset by splitting the data, slicing it into training and testing sets,
    and extracting audio features as specified.
    
    Parameters:
    options (Namespace): Command line arguments specifying dataset operations.
    """
    dataset_path = Path(options.dataset_folder)
    slice_path = Path(__file__).parent
    print("Creating train / test split")
    split_data(dataset_path)

    for strat in ["1", "2", "3", "4", "5", "6"]:
        # Define paths for sliced data
        train_motions = slice_path / f"train/{strat}/motions"
        train_wavs = slice_path / f"train/{strat}/wavs"
        test_motions = slice_path / f"test/{strat}/motions"
        test_wavs = slice_path / f"test/{strat}/wavs"

        print("Slicing train data")
        slice_and_clean(train_motions, train_wavs)

        print("Slicing test data")
        slice_and_clean(test_motions, test_wavs)

        # Extracting audio features
        if options.extract_baseline:
            print("Extracting baseline features")
            # baseline_extract("train/wavs_sliced", "train/baseline_feats")
            baseline_extract(test_wavs, f"test/{strat}/baseline_feats")

        # Don't extract on slices
        if options.extract_jukebox:
            print("Extracting jukebox features")
            jukebox_extract(f"train/{strat}/wavs_sliced", f"train/{strat}/jukebox_feats")
            jukebox_extract(f"test/{strat}/wavs_sliced", f"test/{strat}/jukebox_feats")

def parse_options():
    parser = argparse.ArgumentParser(description="Create dataset for audio and motion processing.")
    parser.add_argument("--dataset_folder", type=str, default="popdanceset",
                        help="Folder containing motions and music.")
    parser.add_argument("--extract-baseline", action="store_true",
                        help="Flag to extract baseline audio features.")
    parser.add_argument("--extract-jukebox", action="store_true",
                        help="Flag to extract jukebox audio features.")
    return parser.parse_args()

if __name__ == "__main__":
    options = parse_options()
    create_dataset(options)
