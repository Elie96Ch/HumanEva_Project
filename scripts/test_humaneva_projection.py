# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:20:43 2026

@author: user
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy.io

from src.utils.camera_projection import load_humaneva_calibration, project_world_to_image


def main():
    pose_path = Path("data/humaneva/processed/body_pose_matlab/S1/Walking_1.mat")
    cal_path = Path("data/humaneva/raw/S1/Calibration_Data/C1.cal")

    pose_data = scipy.io.loadmat(pose_path, squeeze_me=True, struct_as_record=False)
    pose_3d = np.asarray(pose_data["pose_3d"], dtype=np.float32)
    valid = np.asarray(pose_data["valid"]).astype(np.uint8)

    first_valid = int(np.where(valid == 1)[0][0])
    pts3d = pose_3d[first_valid]   # (20, 3)

    cam = load_humaneva_calibration(cal_path)
    pts2d, pts_cam = project_world_to_image(
        pts3d,
        K=cam["K"],
        R=cam["R"],
        t=cam["t"],
        dist=cam["dist"],
    )

    print("frame:", first_valid)
    print("3d shape:", pts3d.shape)
    print("2d shape:", pts2d.shape)
    print("2d min:", pts2d.min(axis=0))
    print("2d max:", pts2d.max(axis=0))
    print("camera z min/max:", pts_cam[:, 2].min(), pts_cam[:, 2].max())
    print("first 5 projected points:\n", pts2d[:5])


if __name__ == "__main__":
    main()