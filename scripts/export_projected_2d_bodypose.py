# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:22:32 2026

@author: user
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy.io

from src.utils.camera_projection import load_humaneva_calibration, project_world_to_image


CAMERAS = ["C1", "C2", "C3", "BW1", "BW2", "BW3", "BW4"]


def main():
    bodypose_root = Path("data/humaneva/processed/body_pose_matlab")
    raw_root = Path("data/humaneva/raw")
    out_root = Path("data/humaneva/processed/projected_bodypose_2d")
    out_root.mkdir(parents=True, exist_ok=True)

    subjects = sorted([p.name for p in bodypose_root.iterdir() if p.is_dir()])

    for subject in subjects:
        subject_pose_dir = bodypose_root / subject
        subject_raw_cal_dir = raw_root / subject / "Calibration_Data"

        if not subject_raw_cal_dir.exists():
            print("Skipping {}, no calibration folder".format(subject))
            continue

        for mat_path in sorted(subject_pose_dir.glob("*.mat")):
            data = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

            pose_3d = np.asarray(data["pose_3d"], dtype=np.float32)   # (T, 20, 3)
            valid = np.asarray(data["valid"]).astype(np.uint8)
            point_names = data["point_names"]

            action = mat_path.stem

            for camera_name in CAMERAS:
                cal_path = subject_raw_cal_dir / "{}.cal".format(camera_name)
                if not cal_path.exists():
                    continue

                cam = load_humaneva_calibration(cal_path)

                T = pose_3d.shape[0]
                pose_2d = np.zeros((T, 20, 2), dtype=np.float32)
                pose_cam = np.zeros((T, 20, 3), dtype=np.float32)

                for t in range(T):
                    pts2d, ptscam = project_world_to_image(
                        pose_3d[t],
                        K=cam["K"],
                        R=cam["R"],
                        t=cam["t"],
                        dist=None,#dist=cam["dist"],
                    )
                    pose_2d[t] = pts2d
                    pose_cam[t] = ptscam

                out_dir = out_root / subject / camera_name
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "{}.npz".format(action)

                np.savez_compressed(
                    out_path,
                    pose_2d=pose_2d,
                    pose_3d=pose_3d,
                    pose_cam=pose_cam,
                    valid=valid,
                    point_names=point_names,
                    subject=subject,
                    action=action,
                    camera_name=camera_name,
                )

                print(
                    "Saved {} | pose_2d={} pose_3d={}".format(
                        out_path, pose_2d.shape, pose_3d.shape
                    )
                )

    print("Done.")


if __name__ == "__main__":
    main()