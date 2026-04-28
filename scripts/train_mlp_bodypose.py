# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 13:49:45 2026

@author: user
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

from src.datasets.humaneva_bodypose import HumanEvaBodyPoseFrameDataset
from src.models.mlp import PoseMLP, mpjpe_loss


def project_to_2d_simple(pose_3d):
    # simple orthographic baseline: keep x,y only
    return pose_3d[:, :, :2]


class Pose2DTo3DDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        pose_3d, meta = self.base_dataset[idx]  # (20, 3)
        pose_3d = pose_3d.float()
        pose_2d = pose_3d[:, :2].contiguous()

        x = pose_2d.view(-1)   # 20*2 = 40
        y = pose_3d.view(-1)   # 20*3 = 60
        return x, y, meta


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_mpjpe = 0.0
    total_count = 0

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            mpjpe = mpjpe_loss(pred, y)

            batch_size = x.shape[0]
            total_loss += loss.item() * batch_size
            total_mpjpe += mpjpe.item() * batch_size
            total_count += batch_size

    return total_loss / total_count, total_mpjpe / total_count


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    data_root = "data/humaneva/processed/body_pose_matlab"

    train_base = HumanEvaBodyPoseFrameDataset(
        root=data_root,
        subjects=["S1", "S2"],
        valid_only=True,
        root_center=True,
        flatten=False,
    )

    val_base = HumanEvaBodyPoseFrameDataset(
        root=data_root,
        subjects=["S3"],
        valid_only=True,
        root_center=True,
        flatten=False,
    )

    train_ds = Pose2DTo3DDataset(train_base)
    val_ds = Pose2DTo3DDataset(val_base)

    print("train samples:", len(train_ds))
    print("val samples:", len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    model = PoseMLP(
        input_dim=40,
        output_dim=60,
        hidden_dim=256,
        num_layers=3,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    num_epochs = 20

    best_val_mpjpe = float("inf")
    out_dir = Path("outputs/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "mlp_bodypose_best.pt"

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0

        pbar = tqdm(train_loader, desc="Epoch {}".format(epoch))
        for x, y, _ in pbar:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = x.shape[0]
            running_loss += loss.item() * batch_size
            running_count += batch_size

            pbar.set_postfix(train_loss=running_loss / running_count)

        train_loss = running_loss / running_count
        val_loss, val_mpjpe = evaluate(model, val_loader, device)

        print(
            "Epoch {:02d} | train_loss={:.6f} | val_loss={:.6f} | val_mpjpe={:.4f}".format(
                epoch, train_loss, val_loss, val_mpjpe
            )
        )

        if val_mpjpe < best_val_mpjpe:
            best_val_mpjpe = val_mpjpe
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mpjpe": val_mpjpe,
                },
                best_path,
            )
            print("Saved best model to:", best_path)

    print("Best val MPJPE:", best_val_mpjpe)


if __name__ == "__main__":
    main()