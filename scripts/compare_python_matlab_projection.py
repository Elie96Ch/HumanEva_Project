# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:43:13 2026

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
    debug_path = PROJECT_ROOT / "outputs" / "debug" / "official_project2d_one_frame.mat"

    data = scipy.io.loadmat(debug_path, squeeze_me=True, struct_as_record=False)

    pose_3d = np.asarray(data["pose_3d"], dtype=np.float32)
    pose_2d_matlab = np.asarray(data["pose_2d_matlab"], dtype=np.float32)

    subject = str(data["subject"])
    camera = str(data["camera"])

    cal_path = PROJECT_ROOT / "data" / "humaneva" / "raw" / subject / "Calibration_Data" / f"{camera}.cal"

    cam = load_humaneva_calibration(cal_path)

    pose_2d_py_nodist, cam_nodist = project_world_to_image(
        pose_3d,
        K=cam["K"],
        R=cam["R"],
        t=cam["t"],
        dist=None,
    )

    pose_2d_py_dist, cam_dist = project_world_to_image(
        pose_3d,
        K=cam["K"],
        R=cam["R"],
        t=cam["t"],
        dist=cam["dist"],
    )

    diff_nodist = np.linalg.norm(pose_2d_py_nodist - pose_2d_matlab, axis=1)
    diff_dist = np.linalg.norm(pose_2d_py_dist - pose_2d_matlab, axis=1)

    print("MATLAB 2D range:")
    print("x:", pose_2d_matlab[:, 0].min(), pose_2d_matlab[:, 0].max())
    print("y:", pose_2d_matlab[:, 1].min(), pose_2d_matlab[:, 1].max())
    print()

    print("Python no distortion range:")
    print("x:", pose_2d_py_nodist[:, 0].min(), pose_2d_py_nodist[:, 0].max())
    print("y:", pose_2d_py_nodist[:, 1].min(), pose_2d_py_nodist[:, 1].max())
    print("diff to MATLAB no distortion mean/max:", diff_nodist.mean(), diff_nodist.max())
    print()

    print("Python with distortion range:")
    print("x:", pose_2d_py_dist[:, 0].min(), pose_2d_py_dist[:, 0].max())
    print("y:", pose_2d_py_dist[:, 1].min(), pose_2d_py_dist[:, 1].max())
    print("diff to MATLAB with distortion mean/max:", diff_dist.mean(), diff_dist.max())
    print()

    print("First 5 MATLAB:")
    print(pose_2d_matlab[:5])
    print("First 5 Python no distortion:")
    print(pose_2d_py_nodist[:5])
    print("First 5 Python with distortion:")
    print(pose_2d_py_dist[:5])


if __name__ == "__main__":
    main()