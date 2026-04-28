from pathlib import Path

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset


POINT_NAMES = [
    "torsoProximal",
    "torsoDistal",
    "upperLArmProximal",
    "upperLArmDistal",
    "lowerLArmProximal",
    "lowerLArmDistal",
    "upperRArmProximal",
    "upperRArmDistal",
    "lowerRArmProximal",
    "lowerRArmDistal",
    "upperLLegProximal",
    "upperLLegDistal",
    "lowerLLegProximal",
    "lowerLLegDistal",
    "upperRLegProximal",
    "upperRLegDistal",
    "lowerRLegProximal",
    "lowerRLegDistal",
    "headProximal",
    "headDistal",
]


def root_center_pose(pose_3d, root_index=1):
    root = pose_3d[root_index:root_index + 1]
    return pose_3d - root


class HumanEvaBodyPoseFrameDataset(Dataset):
    def __init__(
        self,
        root,
        subjects=None,
        actions=None,
        valid_only=True,
        root_center=True,
        root_index=1,
        flatten=False,
        dtype=torch.float32,
    ):
        self.root = Path(root)
        self.subjects = subjects
        self.actions = actions
        self.valid_only = valid_only
        self.root_center = root_center
        self.root_index = root_index
        self.flatten = flatten
        self.dtype = dtype
        self.samples = []
        self._load_all()

    def _match_action(self, stem):
        if self.actions is None:
            return True
        return stem in self.actions

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

            for mat_path in sorted(subject_dir.glob("*.mat")):
                if not self._match_action(mat_path.stem):
                    continue

                data = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

                if "pose_3d" not in data or "valid" not in data:
                    raise KeyError("Missing pose_3d/valid in {}".format(mat_path))

                pose_3d = np.asarray(data["pose_3d"], dtype=np.float32)
                valid = np.asarray(data["valid"]).astype(np.uint8)

                if pose_3d.ndim != 3 or pose_3d.shape[1:] != (20, 3):
                    raise ValueError(
                        "Unexpected pose shape in {}: {}".format(mat_path, pose_3d.shape)
                    )

                for i in range(len(pose_3d)):
                    if self.valid_only and valid[i] == 0:
                        continue

                    pose = pose_3d[i].copy()
                    if self.root_center:
                        pose = root_center_pose(pose, root_index=self.root_index)

                    self.samples.append(
                        {
                            "pose_3d": pose,
                            "subject": subject,
                            "action": mat_path.stem,
                            "frame_idx": int(i),
                            "valid": int(valid[i]),
                            "source_file": str(mat_path),
                        }
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        pose = item["pose_3d"]

        if self.flatten:
            pose = pose.reshape(-1)

        x = torch.tensor(pose, dtype=self.dtype)

        meta = {
            "subject": item["subject"],
            "action": item["action"],
            "frame_idx": item["frame_idx"],
            "valid": item["valid"],
            "source_file": item["source_file"],
        }
        return x, meta