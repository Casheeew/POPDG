import glob
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pytorch3d.transforms import (RotateAxisAngle, axis_angle_to_quaternion, matrix_to_axis_angle,
                                  quaternion_multiply,quaternion_to_axis_angle)
from torch.utils.data import Dataset

from dataset.preprocess import Normalizer, vectorize_many
from dataset.quaternion import ax_to_6v
from vis import SMPLSkeleton

class PopDanceSet(Dataset):
    def __init__(
        self,
        data_path: str,
        backup_path: str,
        train: bool,
        feature_type: str = "jukebox",
        # feature_type: str = "baseline",
        normalizer: Any = None,
        data_len: int = -1,
        include_contacts: bool = True,
        force_reload: bool = False,
    ):
        self.data_path = Path(data_path)
        self.backup_path = Path(backup_path)
        self.backup_path.mkdir(parents=True, exist_ok=True)
        self.input_video_fps = 30
        self.data_fps = 30
        assert self.data_fps <= self.input_video_fps
        self.data_stride = self.input_video_fps // self.data_fps

        self.train = train
        self.name = "Train" if self.train else "Test"
        self.feature_type = feature_type

        self.normalizer = normalizer
        self.data_len = data_len

        pickle_name = f"processed_{self.name.lower()}_data.pkl"

        # save normalizer
        if not train:
            pickle.dump(
                normalizer, open(os.path.join(backup_path, "normalizer.pkl"), "wb")
            )
        
        pickle_path = self.backup_path / pickle_name
        if not force_reload and pickle_path.exists():
            print(f"Using cached dataset from {pickle_path}")
            self.data = self.load_pickle(pickle_path)
        else:
            print("Loading dataset...")
            self.data = self.load_popdanceset()
            self.save_pickle(pickle_path, self.data)

        self.process_data()

    def save_pickle(self, path, data):
        with open(path.as_posix(), 'wb') as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

    def load_pickle(self, path):
        with open(path.as_posix(), 'rb') as file:
            return pickle.load(file)

    def process_data(self):
        print(f"Loaded {self.name} ({self.train}) Dataset With Dimensions: Pos: {self.data['pos'].shape}, Q: {self.data['q'].shape}")
        pose_input = self.process_dataset(self.data["pos"], self.data["q"])
        self.data.update({
            "pose": pose_input,
            "filenames": self.data["filenames"],
            "wavs": self.data["wavs"],
        })
        assert len(pose_input) == len(self.data["filenames"])
        self.length = len(pose_input)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        filename_ = self.data["filenames"][idx]
        feature = torch.from_numpy(np.load(filename_))
        return self.data["pose"][idx], feature, filename_, self.data["wavs"][idx]

    def load_popdanceset(self):
        # open data path
        split_data_path = os.path.join(
            self.data_path, "train" if self.train else "test"
        )

        # The dance clips in the 'high_quality_dataset' directory have a higher quality 
        # than other dance clips in the dataset.
        high_quality_dataset_path = "high_quality_dataset"
        with open(high_quality_dataset_path, 'r') as file:
            high_quality_dataset = set(file.read().splitlines())

        motion_path = os.path.join(split_data_path, "motions_sliced")
        sound_path = os.path.join(split_data_path, f"{self.feature_type}_feats")
        wav_path = os.path.join(split_data_path, f"wavs_sliced")
        # sort motions and sounds
        motions = sorted(glob.glob(os.path.join(motion_path, "*.pkl")))
        features = sorted(glob.glob(os.path.join(sound_path, "*.npy")))
        wavs = sorted(glob.glob(os.path.join(wav_path, "*.wav")))

        # stack the motions and features together
        all_pos = []
        all_q = []
        all_names = []
        all_wavs = []
        assert len(motions) == len(features)

        grouped_data = {}

        for motion, feature, wav in zip(motions, features, wavs):
            # make sure name is matching
            m_name = os.path.splitext(os.path.basename(motion))[0]
            f_name = os.path.splitext(os.path.basename(feature))[0]
            w_name = os.path.splitext(os.path.basename(wav))[0]
            assert m_name == f_name == w_name, str((motion, feature, wav))
            
            base_name = "_".join(os.path.splitext(os.path.basename(motion))[0].split('_')[:-1])
            if base_name in high_quality_dataset:
                if base_name not in grouped_data:
                    grouped_data[base_name] = []
                grouped_data[base_name].append((motion,feature,wav))
        for base_name, files in grouped_data.items():
            for motion, feature, wav in files:
    
                data = pickle.load(open(motion, "rb"))
                pos = data["pos"]
                q = data["q"]

                # The data augmentation can be determined by setting the repeat_count number. 
                if base_name in high_quality_dataset:
                    repeat_count = 0
                
                for _ in range(repeat_count):
                    all_pos.append(pos.copy())
                    all_q.append(q.copy())
                    all_names.append(feature)
                    all_wavs.append(wav)

        # Ensure all data is included, especially when not in high_quality_dataset
        for motion, feature, wav in zip(motions, features, wavs):
            if motion not in sum((files for files in grouped_data.values()), []):
                data = pickle.load(open(motion, 'rb'))
                all_pos.append(data["pos"])
                all_q.append(data["q"])
                all_names.append(feature)
                all_wavs.append(wav)

        all_pos = np.array(all_pos)  # N x seq x 3
        all_q = np.array(all_q)  # N x seq x (joint * 3)
        # downsample the motions to the data fps
        print(all_pos.shape)
        all_pos = all_pos[:, :: self.data_stride, :]
        all_q = all_q[:, :: self.data_stride, :]
        data = {"pos": all_pos, "q": all_q, "filenames": all_names, "wavs": all_wavs}
        return data

    def process_dataset(self, root_pos, local_q):
        # FK skeleton
        smpl = SMPLSkeleton()
        # to Tensor
        root_pos = torch.Tensor(root_pos)
        local_q = torch.Tensor(local_q)
        # to ax
        bs, sq, c = local_q.shape
        local_q = local_q.reshape((bs, sq, -1, 3))

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

        # normalize the data. Both train and test need the same normalizer.
        if self.train:
            self.normalizer = Normalizer(global_pose_vec_input)
        else:
            assert self.normalizer is not None
        global_pose_vec_input = self.normalizer.normalize(global_pose_vec_input)

        assert not torch.isnan(global_pose_vec_input).any()
        data_name = "Train" if self.train else "Test"

        # cut the dataset
        if self.data_len > 0:
            global_pose_vec_input = global_pose_vec_input[: self.data_len]

        global_pose_vec_input = global_pose_vec_input

        print(f"{data_name} Dataset Motion Features Dim: {global_pose_vec_input.shape}")

        return global_pose_vec_input

class SimplifieDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        backup_path: str,
        train: bool = True,
        feature_type: str = "jukebox",
        # feature_type: str = "baseline",
        normalizer: Any = None,
        data_len: int = -1,
        include_contacts: bool = True,
        force_reload: bool = False,
    ):
        self.data_path = Path(data_path)
        self.backup_path = Path(backup_path)
        self.backup_path.mkdir(parents=True, exist_ok=True)
        self.input_video_fps = 30
        self.data_fps = 30
        assert self.data_fps <= self.input_video_fps
        self.data_stride = self.input_video_fps // self.data_fps

        self.train = train
        self.name = "Simplifie"
        self.feature_type = feature_type

        self.normalizer = normalizer
        self.data_len = data_len

        pickle_name = f"processed_{self.name.lower()}_data.pkl"

        # save normalizer
        if not train:
            pickle.dump(
                normalizer, open(os.path.join(backup_path, "normalizer.pkl"), "wb")
            )
        
        pickle_path = self.backup_path / pickle_name
        if not force_reload and pickle_path.exists():
            print(f"Using cached dataset from {pickle_path}")
            self.data = self.load_pickle(pickle_path)
        else:
            print("Loading simplifie dataset...")
            self.data = self.load_simplifie_dataset()
            self.save_pickle(pickle_path, self.data)

        print("")

        self.process_data()

    def save_pickle(self, path, data):
        with open(path.as_posix(), 'wb') as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

    def load_pickle(self, path):
        with open(path.as_posix(), 'rb') as file:
            return pickle.load(file)

    def process_data(self):
        print(f"Loaded {self.name} Dataset With Dimensions:\n \
               Original Pos: {self.data['original_pos'].shape}, Original Q: {self.data['original_q'].shape}, \n \
               Simplified Pos: {self.data['simplified_pos'].shape}, Simplified Q: {self.data['simplified_q'].shape}")
        
        print('**' * 10)

        original_pose_input = self.process_dataset(self.data["original_pos"], self.data["original_q"])
        simplified_pose_input = self.process_dataset(self.data["simplified_pos"], self.data["simplified_q"])
        self.data.update({
            "original_pose": original_pose_input,
            "simplified_pose": simplified_pose_input,
            "filenames": self.data["filenames"],
            "wavs": self.data["wavs"],
        })
        assert len(original_pose_input) == len(simplified_pose_input) == len(self.data["filenames"])
        self.length = len(original_pose_input)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        filename_ = self.data["filenames"][idx]
        feature = torch.from_numpy(np.load(filename_))
        return self.data["original_pose"][idx], self.data["simplified_pose"][idx], feature, filename_, self.data["wavs"][idx]

    def load_simplifie_dataset(self):
        # open data path
        split_data_path = os.path.join(
            self.data_path, "train" if self.train else "test"
        )   

        # The dance clips in the 'high_quality_dataset' directory have a higher quality 
        # than other dance clips in the dataset.
        # high_quality_dataset_path = "high_quality_dataset"
        # with open(high_quality_dataset_path, 'r') as file:
        #     high_quality_dataset = set(file.read().splitlines())

        motion_path = os.path.join(split_data_path, "*", "motions_sliced")
        sound_path = os.path.join(split_data_path, "*", f"{self.feature_type}_feats")
        wav_path = os.path.join(split_data_path, "*", f"wavs_sliced")

        # sort motions and sounds
        original_motions = sorted(glob.glob(os.path.join(motion_path, "*_original_*.npy")))
        simplified_motions = sorted(glob.glob(os.path.join(motion_path, "*_simplified_*.npy")))
        features = sorted(glob.glob(os.path.join(sound_path, "*.npy")))
        wavs = sorted(glob.glob(os.path.join(wav_path, "*.wav")))

        # Helper: get base names without extension
        def get_basenames(file_list):
            names = set()
            for f in file_list:
                base = os.path.splitext(os.path.basename(f))[0]
                base = base.replace("_original_", "_").replace("_simplified_", "_")
                names.add(base)
            return names

        orig_set = get_basenames(original_motions)
        simp_set = get_basenames(simplified_motions)
        feat_set = get_basenames(features)
        # wav_set = get_basenames(wavs)

        if not (len(orig_set) == len(simp_set) == len(feat_set)):
            print(f"Original motions: {len(orig_set)}, Simplified motions: {len(simp_set)}, Features: {len(feat_set)}")

            print("\nIn original but not in simplified:", sorted(orig_set - simp_set))
            print("\nIn simplified but not in original:", sorted(simp_set - orig_set))
            # print("\nIn original but not in features:", sorted(orig_set - feat_set))
            # print("\nIn features but not in original:", sorted(feat_set - orig_set))

            raise AssertionError("motions and features have incompatible length")


        # stack the motions and features together
        all_original_pos = []
        all_simplified_pos = []
        all_original_q = []
        all_simplified_q = []
        all_names = []
        all_wavs = []

        assert len(original_motions) == len(simplified_motions) == len(features), f"motions and features have incompatible length: {len(original_motions)} and {len(simplified_motions)} and {len(features)}"

        # grouped_data = {}

        # for motion, feature, wav in zip(motions, features, wavs):
        #     # make sure name is matching
        #     m_name = os.path.splitext(os.path.basename(motion))[0]
        #     f_name = os.path.splitext(os.path.basename(feature))[0]
        #     w_name = os.path.splitext(os.path.basename(wav))[0]
        #     assert m_name == f_name == w_name, f"names don't match: {str((motion, feature, wav))}"
            
        #     base_name = "_".join(os.path.splitext(os.path.basename(motion))[0].split('_')[:-1])
        #     if base_name in high_quality_dataset:
        #         if base_name not in grouped_data:
        #             grouped_data[base_name] = []
        #         grouped_data[base_name].append((motion,feature,wav))
        # for base_name, files in grouped_data.items():
        #     for motion, feature, wav in files:
    
        #         data = pickle.load(open(motion, "rb"))
        #         pos = data["pos"]
        #         q = data["q"]

        #         # # The data augmentation can be determined by setting the repeat_count number. 
        #         # if base_name in high_quality_dataset:
        #         #     repeat_count = 0
                
        #         # for _ in range(repeat_count):
        #         #     all_pos.append(pos.copy())
        #         #     all_q.append(q.copy())
        #         #     all_names.append(feature)
        #         #     all_wavs.append(wav)

        # Ensure all data is included, especially when not in high_quality_dataset
        for original_motion, simplified_motion, feature, wav in zip(original_motions, simplified_motions, features, wavs):
            # if motion not in sum((files for files in grouped_data.values()), []):

            # make sure name is matching
            o_name = os.path.splitext(os.path.basename(original_motion))[0].replace("original", "").replace("simplified", "")
            s_name = os.path.splitext(os.path.basename(simplified_motion))[0].replace("original", "").replace("simplified", "")
            f_name = os.path.splitext(os.path.basename(feature))[0].replace("original", "").replace("simplified", "")
            w_name = os.path.splitext(os.path.basename(wav))[0].replace("original", "").replace("simplified", "")

            assert o_name == s_name == f_name == w_name, str((original_motion, simplified_motion, feature, wav))

            original_data = np.load(open(original_motion, 'rb'), allow_pickle=True).item()
            simplified_data = np.load(open(simplified_motion, 'rb'), allow_pickle=True).item()

            # print(original_data)

            original_root_pos = original_data["pos"]
            # original_root_pos = original_root_pos.squeeze(1)
            original_local_q = original_data["q"]
            # original_local_q = original_local_q.reshape(original_local_q.shape[0], -1)

            # all_pos.append(data["pos"])
            # all_q.append(data["q"])
            all_original_pos.append(original_root_pos)
            all_original_q.append(original_local_q)

            simplified_root_pos = simplified_data["pos"]
            # simplified_root_pos = simplified_root_pos.squeeze(1)
            simplified_local_q = simplified_data["q"]
            # simplified_local_q = simplified_local_q.reshape(simplified_local_q.shape[0], -1)

            # all_pos.append(data["pos"])
            # all_q.append(data["q"])
            all_simplified_pos.append(simplified_root_pos)
            all_simplified_q.append(simplified_local_q)

            all_names.append(feature)
            all_wavs.append(wav)

        all_original_pos = np.array(all_original_pos)  # N x seq x 3
        all_original_q = np.array(all_original_q)  # N x seq x (joint * 3)

        all_simplified_pos = np.array(all_simplified_pos)  # N x seq x 3
        all_simplified_q = np.array(all_simplified_q)  # N x seq x (joint * 3)

        # downsample the motions to the data fps
        # print(all_original_pos.shape)
        all_original_pos = all_original_pos[:, :: self.data_stride, :]
        all_original_q = all_original_q[:, :: self.data_stride, :]

        all_simplified_pos = all_simplified_pos[:, :: self.data_stride, :]
        all_simplified_q = all_simplified_q[:, :: self.data_stride, :]


        data = {"original_pos": all_original_pos, "original_q": all_original_q, "simplified_pos": all_simplified_pos, "simplified_q": all_simplified_q, "filenames": all_names, "wavs": all_wavs}
        return data

    def process_dataset(self, root_pos, local_q):
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

        # normalize the data. Both train and test need the same normalizer.
        if self.train:
            self.normalizer = Normalizer(global_pose_vec_input)
        else:
            assert self.normalizer is not None
        global_pose_vec_input = self.normalizer.normalize(global_pose_vec_input)

        assert not torch.isnan(global_pose_vec_input).any()
        data_name = "Train" if self.train else "Test"

        # cut the dataset
        if self.data_len > 0:
            global_pose_vec_input = global_pose_vec_input[: self.data_len]

        global_pose_vec_input = global_pose_vec_input

        print(f"{data_name} Dataset Motion Features Dim: {global_pose_vec_input.shape}")

        return global_pose_vec_input
