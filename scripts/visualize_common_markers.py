#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def find_nonempty_frame(markers):
    for i in range(len(markers)):
        pts = markers[i]
        valid = ~(np.all(pts == 0, axis=1))
        if np.any(valid):
            return i
    return None


def main():
    parser = argparse.ArgumentParser(description="Visualize one frame of processed HumanEva common markers")
    parser.add_argument("--npz", type=Path, required=True, help="Path to npz file")
    parser.add_argument("--frame", type=int, default=None, help="Frame index")
    args = parser.parse_args()

    if not args.npz.exists():
        raise FileNotFoundError(f"File not found: {args.npz}")

    data = np.load(args.npz, allow_pickle=True)
    markers = data["markers_3d"]
    labels = data["marker_labels"]

    if args.frame is None:
        frame_idx = find_nonempty_frame(markers)
        if frame_idx is None:
            raise RuntimeError("No non-empty frame found.")
    else:
        frame_idx = max(0, min(args.frame, len(markers) - 1))

    pts = markers[frame_idx]
    valid = ~(np.all(pts == 0, axis=1))
    pts_valid = pts[valid]
    labels_valid = labels[valid]

    print("markers shape:", markers.shape)
    print("selected frame:", frame_idx)
    print("visible markers:", len(pts_valid))

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts_valid[:, 0], pts_valid[:, 1], pts_valid[:, 2], s=12)

    for p, lbl in zip(pts_valid, labels_valid):
        ax.text(p[0], p[1], p[2], str(lbl), fontsize=7)

    ax.set_title(f"{args.npz.name} | frame {frame_idx}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


if __name__ == "__main__":
    main()