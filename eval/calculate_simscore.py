import argparse
import glob
import os
import pickle

import numpy as np
from tqdm import tqdm
import random
from complexity_functions_revised import compute_all_complexities_revised

def calc_physical_score(dir):
    # scores = []

    c1 = []
    c2 = []
    c3 = []
    c4 = []
    c5 = []
    
    names = []
    up_dir = 2  # z is up
    # flat_dirs = [i for i in range(3) if i != up_dir]
    # cube_dirs = [i for i in range(3)]
    # DT = 1 / 30

    it = glob.glob(os.path.join(dir, "*_simplified.pkl"))
    if len(it) > 1000:
        it = random.sample(it, 1000)
    for pkl in tqdm(it):
        info = pickle.load(open(pkl, "rb"))
        joint3d = info["full_pose"]

        # print("Joint 3D shape:", joint3d.shape)
        # print("----------------")

        # Foot labels: [7, 8, 10, 11]

        body = joint3d[:, (7, 8, 10, 11)]
        bodyv = np.zeros((joint3d.shape[0], 4))
        bodyv[:-1] = np.linalg.norm(body[1:] - body[:-1], axis=-1)


        # Foot contact label is 0.01 in POPDG but 0.05 in MotionLCM

        contacts = (bodyv < 0.05)      

        # scores.append(compute_all_complexities_revised(joint3d, contacts, fps=30))

        complexities = compute_all_complexities_revised(joint3d, contacts, fps=30)

        for metric_name, metric_data in complexities.items():
            simscore = metric_data['complexity']

            if metric_name.startswith('C1'):
                c1.append(simscore)
            elif metric_name.startswith('C2'):
                c2.append(simscore)
            elif metric_name.startswith('C3'):
                c3.append(simscore)
            elif metric_name.startswith('C4'):
                c4.append(simscore)
            elif metric_name.startswith('C5'):
                c5.append(simscore)

            # print(f"{metric_name}: {simscore:.3f}")

        names.append(pkl)

    c1 = np.mean(c1)
    c2 = np.mean(c2)
    c3 = np.mean(c3)
    c4 = np.mean(c4)
    c5 = np.mean(c5)

    print(f"C1 Foot movement: {c1:.3f}")
    print(f"C2 Movement density: {c2:.3f}")
    print(f"C3 Rotation: {c3:.3f}")
    print(f"C4 Coordination: {c4:.3f}")
    print(f"C5 Asymmetry: {c5:.3f}")

    # out = np.mean(scores) * 1
    # print(f"{dir} has a mean simplification score of {out}")

    print("Overall complexity score:", np.mean([c1, c2, c3, c4, c5]))


def parse_eval_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_path",
        type=str,
        default="eval/motions",
        help="Where to load saved motions",
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_eval_opt()
    calc_physical_score(opt.motion_path)
