# -*- coding: utf-8 -*-
"""
Dataset loader for projected HumanEva-I 2D-to-3D body-pose lifting.

This loader expects .npz files produced by scripts/export_projected_2d_bodypose.py
with fields:

    pose_2d      (T, 20, 2)
    pose_3d      (T, 20, 3)
    pose_cam     (T, 20, 3)
    valid        (T,)
    point_names
    subject
    action
    camera_name

Important design choice:
    _load_all() stores raw 2D and raw 3D poses only.
    Normalization/root-centering/scaling are applied exactly once in __getitem__().
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def root_center_pose(pose_3d, root_index=1):
    """
    Root-center a 3D pose.

    Parameters
    ----------
    pose_3d : np.ndarray
        Pose array with shape (J, 3).
    root_index : int
        Joint index used as root.

    Returns
    -------
    np.ndarray
        Root-centered pose with shape (J, 3).
    """
    root = pose_3d[root_index:root_index + 1]
    return pose_3d - root


def normalize_2d_points(points_2d, root_index=1):
    """
    Root-center and scale-normalize 2D points.

    Scale is the maximum Euclidean distance from the root joint.
    """
    root = points_2d[root_index:root_index + 1]
    centered = points_2d - root

    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale < 1e-8:
        scale = 1.0

    return centered / scale


def compute_3d_scale(points_3d):
    """
    Compute 3D pose scale as the maximum Euclidean distance from origin.

    This should be called after optional root-centering when using normalized
    3D targets.
    """
    scale = np.max(np.linalg.norm(points_3d, axis=1))
    if scale < 1e-8:
        scale = 1.0

    return scale


def normalize_3d_points(points_3d):
    """
    Scale-normalize 3D points by max Euclidean distance.

    Kept as a helper for experiments, but __getitem__ uses compute_3d_scale()
    so the scale can be returned in meta.
    """
    scale = compute_3d_scale(points_3d)
    return points_3d / scale


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
        normalize_3d=True,
        root_index=1,
        flatten=True,
        dtype=torch.float32,
    ):
        """
        Parameters
        ----------
        root : str or Path
            Root directory of projected dataset.
            Example: data/humaneva/processed/projected_bodypose_2d

        subjects : list[str] or None
            Subject names to include, e.g. ["S1", "S2"].

        cameras : list[str] or None
            Camera names to include, e.g. ["C1"] or ["C1", "C2", "C3"].

        actions : list[str] or None
            Action names to include, e.g. ["Walking_1", "Jog_1"].

        valid_only : bool
            If True, skip frames marked invalid.

        root_center_3d : bool
            If True, subtract root joint from 3D target.

        normalize_2d : bool
            If True, root-center and scale-normalize 2D input.

        normalize_3d : bool
            If True, scale-normalize 3D target and return scale_3d in meta.

        root_index : int
            Root joint index. Current project uses torsoDistal = 1.

        flatten : bool
            If True:
                x shape: (40,)
                y shape: (60,)
            If False:
                x shape: (20, 2)
                y shape: (20, 3)

        dtype : torch.dtype
            Tensor dtype.
        """
        self.root = Path(root)
        self.subjects = subjects
        self.cameras = cameras
        self.actions = actions
        self.valid_only = valid_only
        self.root_center_3d = root_center_3d
        self.normalize_2d = normalize_2d
        self.normalize_3d = normalize_3d
        self.root_index = root_index
        self.flatten = flatten
        self.dtype = dtype

        self.samples = []
        self._load_all()

    def _load_all(self):
        """
        Load sample metadata and raw poses.

        Do not normalize or root-center here.
        That must happen exactly once in __getitem__().
        """
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

                    if "pose_2d" not in data or "pose_3d" not in data or "valid" not in data:
                        raise KeyError(
                            "Expected pose_2d, pose_3d, and valid in {}".format(npz_path)
                        )

                    pose_2d = np.asarray(data["pose_2d"], dtype=np.float32)
                    pose_3d = np.asarray(data["pose_3d"], dtype=np.float32)
                    valid = np.asarray(data["valid"]).astype(np.uint8)

                    if pose_2d.ndim != 3 or pose_2d.shape[1:] != (20, 2):
                        raise ValueError(
                            "Unexpected pose_2d shape in {}: {}".format(
                                npz_path,
                                pose_2d.shape,
                            )
                        )

                    if pose_3d.ndim != 3 or pose_3d.shape[1:] != (20, 3):
                        raise ValueError(
                            "Unexpected pose_3d shape in {}: {}".format(
                                npz_path,
                                pose_3d.shape,
                            )
                        )

                    if len(valid) != len(pose_2d):
                        raise ValueError(
                            "valid length {} does not match pose_2d length {} in {}".format(
                                len(valid),
                                len(pose_2d),
                                npz_path,
                            )
                        )

                    for i in range(len(pose_2d)):
                        if self.valid_only and valid[i] == 0:
                            continue

                        self.samples.append(
                            {
                                "pose_2d_raw": pose_2d[i].copy(),
                                "pose_3d_raw": pose_3d[i].copy(),
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

        x = item["pose_2d_raw"].copy()
        y = item["pose_3d_raw"].copy()

        root_3d = y[self.root_index:self.root_index + 1].copy()

        if self.normalize_2d:
            x = normalize_2d_points(x, root_index=self.root_index)

        if self.root_center_3d:
            y = root_center_pose(y, root_index=self.root_index)

        scale_3d = 1.0
        if self.normalize_3d:
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
            "root_3d": root_3d.astype(np.float32),
        }

        return x, y, meta