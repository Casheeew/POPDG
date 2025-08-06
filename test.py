import glob
import os
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
import tempfile
import random

import numpy as np
import torch
from tqdm import tqdm

from args import parse_test_opt
from data.slice import slice_audio, slice_motion
from POPDG import POPDG
from data.audio_extraction.baseline_features import extract_audio_features as baseline_extract
from data.audio_extraction.jukebox_features import extract_audio_features as juke_extract

from vis import SMPLSkeleton
from pytorch3d.transforms import (RotateAxisAngle, axis_angle_to_quaternion, matrix_to_axis_angle,
                                  quaternion_multiply,quaternion_to_axis_angle)
from dataset.quaternion import ax_to_6v

from dataset.preprocess import Normalizer, vectorize_many

random.seed(123)

def process_dataset(root_pos, local_q, normalizer):
    # FK skeleton
    smpl = SMPLSkeleton()
    # to Tensor
    root_pos = torch.Tensor(root_pos)
    local_q = torch.Tensor(local_q)
    # to ax

    # Our local_q is already in the right dimensions

    # print("local_q.shape", local_q.shape)
    # bs, sq, c = local_q.shape
    # local_q = local_q.reshape((bs, sq, -1, 3))

    # After the extraction of PoPDanceSet using HyBrIK, 
    # rotation adjustments need to be made to ensure that the human body's z-axis is oriented upwards.
    root_q = local_q[:, :, :1, :]  
    root_q_quat = axis_angle_to_quaternion(root_q)
    rotation = torch.Tensor(
        [0.7071068, -0.7071068, 0, 0]
    ) 
    root_q_quat = quaternion_multiply(rotation, root_q_quat)
    root_q = quaternion_to_axis_angle(root_q_quat)
    local_q[:, :, :1, :] = root_q

    # The coordinates of the human body's root joint 
    # need to be shifted and scaled to fall within the range of [-1, 1].
    pos_rotation = RotateAxisAngle(-90, axis="X", degrees=True)
    root_pos = pos_rotation.transform_points(
        root_pos
    )
    root_pos[:, :, 2] += 1
    root_pos[:, :, 1] -= 5
    
    # do FK
    positions = smpl.forward(local_q, root_pos)  # batch x sequence x 24 x 3
    body = positions[:, :, (7, 8, 10, 11, 15, 20, 21, 22, 23)]
    bodyv = torch.zeros(body.shape[:3])
    bodyv[:, :-1] = (body[:, 1:] - body[:, :-1]).norm(dim=-1)
    contacts = (bodyv < 0.01).to(local_q)  # cast to right dtype

    # to 6d
    local_q = ax_to_6v(local_q)

    # now, flatten everything into: batch x sequence x [...]
    l = [root_pos, local_q, contacts]
    global_pose_vec_input = vectorize_many(l).float().detach()

    # # normalize the data. Both train and test need the same normalizer.
    # if self.train:
    #     self.normalizer = Normalizer(global_pose_vec_input)
    # else:
    #     assert self.normalizer is not None
    global_pose_vec_input = normalizer.normalize(global_pose_vec_input)

    assert not torch.isnan(global_pose_vec_input).any()
    # data_name = "Train" if self.train else "Test"

    # # cut the dataset
    # if self.data_len > 0:
    #     global_pose_vec_input = global_pose_vec_input[: self.data_len]

    global_pose_vec_input = global_pose_vec_input

    print(f"Dataset Motion Features Dim: {global_pose_vec_input.shape}")

    return global_pose_vec_input


def extract_slice_number(filename):
    """Extract the numeric part of the slice from a filename."""
    return int(Path(filename).stem.split("_slice")[-1])

def compare_filenames(a, b):
    """Compare two filenames based on their base name and slice number."""
    base_name_a, slice_number_a = Path(a).stem.rsplit('_', 1)
    base_name_b, slice_number_b = Path(b).stem.rsplit('_', 1)

    # Compare the base names
    if base_name_a != base_name_b:
        return -1 if base_name_a < base_name_b else 1

    # Compare the slice numbers
    slice_a = extract_slice_number(a)
    slice_b = extract_slice_number(b)
    return -1 if slice_a < slice_b else (1 if slice_a > slice_b else 0)

# Convert comparison function to a key function for sorting
sort_key = cmp_to_key(compare_filenames)

def test(opt):
    feature_func = juke_extract if opt.feature_type == "jukebox" else baseline_extract
    sample_length = opt.out_length
    sample_size = int(sample_length / 2.5) - 1
    # sample_size = 1

    checkpoint_path = opt.checkpoint
    checkpoint = torch.load(
        checkpoint_path, map_location="cuda", weights_only=False
    )
    normalizer = checkpoint["normalizer"]

    temp_dir_list = []
    all_cond = []
    all_filenames = []
    
    all_pos = []
    all_q = []
    all_motions = []

    if opt.use_cached_features:
        print("Using precomputed features")
        # all subdirectories
        dir_list = glob.glob(os.path.join(opt.feature_cache_dir, "*/"))
        for dir in dir_list:
            file_list = sorted(glob.glob(f"{dir}/*.wav"), key=sort_key)
            juke_file_list = sorted(glob.glob(f"{dir}/*.npy"), key=sort_key)
            assert len(file_list) == len(juke_file_list)
            # random chunk after sanity check
            rand_idx = random.randint(0, len(file_list) - sample_size)
            file_list = file_list[rand_idx : rand_idx + sample_size]
            juke_file_list = juke_file_list[rand_idx : rand_idx + sample_size]
            cond_list = [np.load(x) for x in juke_file_list]
            all_filenames.append(file_list)
            all_cond.append(torch.from_numpy(np.array(cond_list)))
    else:
        print("Computing features for input music")
        for wav_file in glob.glob(os.path.join(opt.music_dir, "*.wav")):
            motion_file = wav_file.replace(".wav", ".npy")

            # create temp folder (or use the cache folder if specified)
            if opt.cache_features:
                songname = os.path.splitext(os.path.basename(wav_file))[0]
                save_dir = os.path.join(opt.feature_cache_dir, songname)
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                dirname = save_dir
            else:
                # temp_dir = TemporaryDirectory()
                temp_dir = tempfile.mkdtemp()
                temp_dir_list.append(temp_dir)
                # dirname = temp_dir.name
                dirname = temp_dir
            # slice the audio file
            print(f"Slicing audio: {wav_file}")
            audio_slices = slice_audio(wav_file, 2.5, 5.0, os.path.join(dirname, "wavs_sliced"))

            # slice the motion file

            print(f"Slicing motion: {motion_file}")
            slice_motion(motion_file, 2.5, 5.0, audio_slices, os.path.join(dirname, "motions_sliced"))

            file_list = sorted(glob.glob(f"{dirname}/wavs_sliced/*.wav"), key=sort_key)
            rand_idx = random.randint(0, len(file_list) - sample_size)
            cond_list = []

            pos_list = []
            q_list = []

            # generate juke representations
            print(f"Computing features for {wav_file}")
            for idx, file in enumerate(tqdm(file_list)):

                # if not caching then only calculate for the interested range
                if (not opt.cache_features) and (not (rand_idx <= idx < rand_idx + sample_size)):
                    continue

                print(idx)
                # Pre-process audio
                reps, _ = feature_func(file)
                # save reps
                if opt.cache_features:
                    featurename = os.path.splitext(file)[0] + ".npy"
                    np.save(featurename, reps)
                # if in the random range, put it into the list of reps we want
                # to actually use for generation
                if rand_idx <= idx < rand_idx + sample_size:
                    cond_list.append(reps)

                
                # Pre-process motions

                file = file.replace("wavs_sliced", "motions_sliced").replace(".wav", ".npy")

                motion_data = np.load(open(file, 'rb'), allow_pickle=True).item()

                root_pos = motion_data["pos"]
                local_q = motion_data["q"]

                if rand_idx <= idx < rand_idx + sample_size:
                    pos_list.append(root_pos)
                    q_list.append(local_q)

            
            # Downsample the motions to match fps: todo!


            cond_list = torch.from_numpy(np.array(cond_list))
            pos_list = torch.from_numpy(np.array(pos_list))
            q_list = torch.from_numpy(np.array(q_list))

            

            all_cond.append(cond_list)
            all_pos.append(pos_list)
            all_q.append(q_list)

            all_motions.append(process_dataset(pos_list, q_list, normalizer))

            all_filenames.append(file_list[rand_idx : rand_idx + sample_size])

    model = POPDG(opt.feature_type, opt.checkpoint)
    model.eval()

    # directory for optionally saving the dances for eval
    fk_out = None
    if opt.save_motions:
        fk_out = opt.motion_save_dir

    print("Generating dances")
    for i in range(len(all_cond)):
        data_tuple = None, all_cond[i], all_motions[i], all_filenames[i]
        model.render_sample(
            data_tuple, "test", opt.render_dir, render_count=-1, fk_out=fk_out, render=not opt.no_render, ref_fk_out='eval/reference_motions'
        )
    print("Done")
    torch.cuda.empty_cache()
    # for temp_dir in temp_dir_list:
    #     temp_dir.cleanup()


if __name__ == "__main__":
    opt = parse_test_opt()
    test(opt)