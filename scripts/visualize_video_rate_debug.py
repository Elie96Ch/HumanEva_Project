# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:25:50 2026

@author: user
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def read_video_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx}")

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def draw_pose(ax, pts, edges, color="yellow", label=None):
    pts = np.asarray(pts)
    ax.scatter(pts[:, 0], pts[:, 1], s=20, c=color, label=label)
    for i, j in edges:
        if np.all(np.isfinite(pts[[i, j]])):
            ax.plot(
                [pts[i, 0], pts[j, 0]],
                [pts[i, 1], pts[j, 1]],
                color=color,
                linewidth=2,
            )


def main():
    debug_path = PROJECT_ROOT / "outputs" / "debug" / "video_rate_pose_debug.mat"
    raw_root = PROJECT_ROOT / "data" / "humaneva" / "raw"

    data = scipy.io.loadmat(debug_path, squeeze_me=True, struct_as_record=False)

    pose_2d = np.asarray(data["pose_2d"], dtype=np.float32)
    valid = np.asarray(data["valid"]).reshape(-1)
    frame_ids = np.asarray(data["frame_ids"]).reshape(-1).astype(int)

    subject = str(data["subject"])
    action = str(data["action"])
    camera = str(data["camera"])

    video_path = raw_root / subject / "Image_Data" / f"{action}_({camera}).avi"

    out_dir = PROJECT_ROOT / "outputs" / "figures" / "video_rate_pose_debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    edges = [
        (1, 0),
        (0, 18),
        (18, 19),
        (0, 2), (2, 3), (3, 4), (4, 5),
        (0, 6), (6, 7), (7, 8), (8, 9),
        (1, 10), (10, 11), (11, 12), (12, 13),
        (1, 14), (14, 15), (15, 16), (16, 17),
    ]

    for i, frame_id in enumerate(frame_ids):
        if valid[i] != 1:
            print("[WARN] invalid frame:", frame_id)

        frame = read_video_frame(video_path, frame_id)
        pts = pose_2d[i]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(frame)
        draw_pose(ax, pts, edges, color="yellow", label="MATLAB project2d, scaling=2")

        ax.set_title(f"{subject} {action} {camera} | video frame {frame_id}")
        ax.axis("off")
        ax.legend(loc="upper right")

        save_path = out_dir / f"{subject}_{action}_{camera}_v{frame_id:06d}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

        print("saved:", save_path)


if __name__ == "__main__":
    main()