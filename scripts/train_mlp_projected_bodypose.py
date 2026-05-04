# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:29:30 2026

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

from src.datasets.humaneva_projected import HumanEvaProjectedFrameDataset
from src.models.mlp import PoseMLP, mpjpe_loss


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_mpjpe_norm = 0.0
    total_mpjpe_mm = 0.0
    total_count = 0

    with torch.no_grad():
        for x, y, meta in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            mpjpe_norm = mpjpe_loss(pred, y)

            pred_3d = pred.view(pred.shape[0], -1, 3)
            y_3d = y.view(y.shape[0], -1, 3)

            scales = meta["scale_3d"]
            if not torch.is_tensor(scales):
                scales = torch.tensor(scales, dtype=torch.float32)
            scales = scales.to(device).view(-1, 1, 1)

            pred_mm = pred_3d * scales
            y_mm = y_3d * scales

            mpjpe_mm = torch.norm(pred_mm - y_mm, dim=-1).mean()

            batch_size = x.shape[0]
            total_loss += loss.item() * batch_size
            total_mpjpe_norm += mpjpe_norm.item() * batch_size
            total_mpjpe_mm += mpjpe_mm.item() * batch_size
            total_count += batch_size

    return (
        total_loss / total_count,
        total_mpjpe_norm / total_count,
        total_mpjpe_mm / total_count,
    )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    data_root = "data/humaneva/processed/projected_bodypose_2d"

    train_ds = HumanEvaProjectedFrameDataset(
    root=data_root,
    subjects=["S1", "S2"],
    cameras=["C1"],
    valid_only=True,
    root_center_3d=True,
    normalize_2d=True,
    flatten=True,
    )
    
    val_ds = HumanEvaProjectedFrameDataset(
        root=data_root,
        subjects=["S3"],
        cameras=["C1"],
        valid_only=True,
        root_center_3d=True,
        normalize_2d=True,
        flatten=True,
    )

    print("train samples:", len(train_ds))
    print("val samples:", len(val_ds))
    print("train cameras:", sorted(set([s["camera_name"] for s in train_ds.samples])))
    print("val cameras:", sorted(set([s["camera_name"] for s in val_ds.samples])))


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

    best_val_mpjpe_mm = float("inf")
    out_dir = Path("outputs/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "mlp_projected_bodypose_best.pt"

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
        val_loss, val_mpjpe_norm, val_mpjpe_mm = evaluate(model, val_loader, device)

        print(
            "Epoch {:02d} | train_loss={:.6f} | val_loss={:.6f} | val_mpjpe_norm={:.4f} | val_mpjpe_mm={:.4f}".format(
                epoch, train_loss, val_loss, val_mpjpe_norm, val_mpjpe_mm
            )
        )

        if val_mpjpe_mm < best_val_mpjpe_mm:
            best_val_mpjpe_mm = val_mpjpe_mm
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mpjpe_norm": val_mpjpe_norm,
                    "val_mpjpe_mm": val_mpjpe_mm,
                },
                best_path,
            )
            print("Saved best model to:", best_path)

    print("Best val MPJPE (mm):", best_val_mpjpe_mm)

if __name__ == "__main__":
    main()