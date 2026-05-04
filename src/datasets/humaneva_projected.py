# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:26:35 2026

@author: user
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def root_center_pose(pose_3d, root_index=1):
    root = pose_3d[root_index:root_index + 1]
    return pose_3d - root


def normalize_2d_points(points_2d, root_index=1):
    root = points_2d[root_index:root_index + 1]
    centered = points_2d - root
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale < 1e-8:
        scale = 1.0
    return centered / scale

def normalize_3d_points(points_3d):
    scale = np.max(np.linalg.norm(points_3d, axis=1))
    if scale < 1e-8:
        scale = 1.0
    return points_3d / scale

def compute_3d_scale(points_3d):
    scale = np.max(np.linalg.norm(points_3d, axis=1))
    if scale < 1e-8:
        scale = 1.0
    return scale

class HumanEvaProjectedFrameDataset(Dataset):
    def __init__(
        self,
        root,
        subjects=None,
        cameras=None,
        actions=None,
        valid_only=True,
        root_center_3d=True,
        normalize_2d=True,
        root_index=1,
        flatten=True,
        dtype=torch.float32,
    ):
        self.root = Path(root)
        self.subjects = subjects
        self.cameras = cameras
        self.actions = actions
        self.valid_only = valid_only
        self.root_center_3d = root_center_3d
        self.normalize_2d = normalize_2d
        self.root_index = root_index
        self.flatten = flatten
        self.dtype = dtype
        self.samples = []
        self._load_all()

    def _load_all(self):
        if not self.root.exists():
            raise FileNotFoundError("Dataset root not found: {}".format(self.root))

        subjects = self.subjects
        if subjects is None:
            subjects = sorted([p.name for p in self.root.iterdir() if p.is_dir()])

        for subject in subjects:
            subject_dir = self.root / subject
            if not subject_dir.exists():
                continue

            camera_dirs = self.cameras
            if camera_dirs is None:
                camera_dirs = sorted([p.name for p in subject_dir.iterdir() if p.is_dir()])

            for camera_name in camera_dirs:
                camera_dir = subject_dir / camera_name
                if not camera_dir.exists():
                    continue

                for npz_path in sorted(camera_dir.glob("*.npz")):
                    if self.actions is not None and npz_path.stem not in self.actions:
                        continue

                    data = np.load(npz_path, allow_pickle=True)
                    pose_2d = np.asarray(data["pose_2d"], dtype=np.float32)
                    pose_3d = np.asarray(data["pose_3d"], dtype=np.float32)
                    valid = np.asarray(data["valid"]).astype(np.uint8)

                    for i in range(len(pose_2d)):
                        if self.valid_only and valid[i] == 0:
                            continue

                        x2d = pose_2d[i].copy()
                        y3d = pose_3d[i].copy()

                        if self.normalize_2d:
                            x2d = normalize_2d_points(x2d, root_index=self.root_index)

                        if self.root_center_3d:
                            y3d = root_center_pose(y3d, root_index=self.root_index)

                        self.samples.append(
                            {
                                "pose_2d": x2d,
                                "pose_3d": y3d,
                                "subject": subject,
                                "camera_name": camera_name,
                                "action": npz_path.stem,
                                "frame_idx": int(i),
                                "valid": int(valid[i]),
                                "source_file": str(npz_path),
                            }
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
    
        x = item["pose_2d"].copy()
        y = item["pose_3d"].copy()
    
        if self.normalize_2d:
            x = normalize_2d_points(x, root_index=self.root_index)
    
        if self.root_center_3d:
            y = root_center_pose(y, root_index=self.root_index)
    
        scale_3d = compute_3d_scale(y)
        y = y / scale_3d
    
        if self.flatten:
            x = x.reshape(-1)
            y = y.reshape(-1)
    
        x = torch.tensor(x, dtype=self.dtype)
        y = torch.tensor(y, dtype=self.dtype)
    
        meta = {
            "subject": item["subject"],
            "camera_name": item["camera_name"],
            "action": item["action"],
            "frame_idx": item["frame_idx"],
            "valid": item["valid"],
            "source_file": item["source_file"],
            "scale_3d": float(scale_3d),
        }
        return x, y, meta