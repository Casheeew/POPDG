import glob
import os
import pickle
import shutil
from pathlib import Path
import numpy as np

def file_to_list(filepath):
    with open(filepath, "r") as file:
        lines = file.readlines()
    return [line.strip() for line in lines if line.strip()]

def prepare_directories(base_dir, subfolders):
    for folder in subfolders:
        Path(base_dir / folder).mkdir(parents=True, exist_ok=True)

def copy_and_process_files(dataset_path, split_name, split_list):
    base_path = Path(dataset_path)

    for sequence in split_list:
        strat, seq = sequence.split("/")
        motion_path = base_path / f"{strat}" / 'motions' / f"{seq}_simplified.npy"
        original_path = base_path / f"{strat}" / 'motions' / f"{seq}_original.npy"
        wav_path = base_path / f"{strat}" / 'wavs' / f"{seq}_original.wav"

        if not motion_path.exists() or not wav_path.exists() or not original_path.exists():
            print(f"File missing: {strat}/{motion_path} or {strat}/{wav_path} or {strat}/{original_path}")
            continue

        # with open(motion_path, "rb") as f:
        #     motion_data = np.load(f, allow_pickle=True)

        # print(motion_data)
        
        # trans = motion_data["transl"]
        # pose = motion_data["pred_thetas"].reshape(motion_data["pred_thetas"].shape[0], -1)
        # scale = motion_data["pred_camera"]

        # out_data = {"pos": trans, "q": pose, "scale": scale}
        # with open(f"{split_name}" / f"{strat}" / 'motions' / f"{seq}.npy", "wb") as f:
        #     pickle.dump(out_data, f)

        shutil.copyfile(motion_path, f"{split_name}/{strat}/motions/{seq}_simplified.npy")
        shutil.copyfile(original_path, f"{split_name}/{strat}/motions/{seq}_original.npy")
        shutil.copyfile(wav_path, f"{split_name}/{strat}/wavs/{seq}_original.wav")

def split_data(dataset_path):
    train_list = set(file_to_list("splits/train.txt"))
    test_list = set(file_to_list("splits/test.txt"))

    dataset_path = Path(dataset_path)
    train_path = Path(__file__).parent
    prepare_directories(train_path, [f"{split}/{s}/{typ}" for s in ["1", "2", "3", "4", "5", "6"] for split in ["train", "test"] for typ in ["motions", "wavs"]])

    for split_list, split_name in zip([train_list, test_list], ["train", "test"]):
        copy_and_process_files(dataset_path, split_name, split_list)

