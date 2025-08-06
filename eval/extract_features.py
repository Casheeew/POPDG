import os
import pickle
from vis import SMPLSkeleton
import numpy as np
from eval.features.kinetic import extract_kinetic_features
from eval.features.manual import extract_manual_features
import torch
from pytorch3d.transforms import (RotateAxisAngle, axis_angle_to_quaternion, matrix_to_axis_angle,
                                  quaternion_multiply,
                                  quaternion_to_axis_angle)
import argparse


# please remember to replace 'train' with 'test' after extracting training data.
# folder_path = 'data/train/motions'
# save_dir = 'data/features'

train_path = 'data/test'

for strategy in os.listdir(train_path):
    folder_path = os.path.join(train_path, strategy, 'motions')
    save_dir = os.path.join('data/features', strategy)

    print("Processing: ", folder_path)

    for filename in os.listdir(folder_path):
        # if filename.endswith('.pkl'):  
        if filename.endswith('_simplified.npy'):  
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'rb') as file:
                # data = pickle.load(file)
                data = np.load(file, allow_pickle=True).item()

            # print(data.keys())
            # print('----------------------------')



            smpl = SMPLSkeleton()
            # pos, q = data["pos"], data["q"]
            # scale = data["scale"]
            # pos *= scale

            pos = data["pred_trans"].squeeze(1)
            q = matrix_to_axis_angle(data["pred_rotmat"])

            pos = pos.unsqueeze(0)
            
            q = q.unsqueeze(0).view(1,-1,24,3)


            root_q = q[:, :, :1, :]  
            root_q_quat = axis_angle_to_quaternion(root_q)
            rotation = torch.Tensor(
                [0.7071068, -0.7071068, 0, 0]
            )  
            root_q_quat = quaternion_multiply(rotation, root_q_quat)
            root_q = quaternion_to_axis_angle(root_q_quat)
            q[:, :, :1, :] = root_q

            pos_rotation = RotateAxisAngle(-90, axis="X", degrees=True)
            pos = pos_rotation.transform_points(
                pos
            ) 
            pos[:, :, 2] += 1
            pos[:, :, 1] -= 5

            keypoints3d = smpl.forward(q, pos).detach().cpu().numpy().squeeze()  # b, s, 24, 3

            features_manual = extract_manual_features(keypoints3d)
            features_kinetic = extract_kinetic_features(keypoints3d)

            os.makedirs(save_dir, exist_ok=True)

            manual_feature_filename = os.path.splitext(filename)[0] + "_manual.npy"
            kinetic_feature_filename = os.path.splitext(filename)[0] + "_kinetic.npy"

            np.save(os.path.join(save_dir, manual_feature_filename), features_manual)
            np.save(os.path.join(save_dir, kinetic_feature_filename), features_kinetic)

            print(f"processed files: {filename}")