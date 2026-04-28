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
    parser = argparse.ArgumentParser(description="Visualize one frame of HumanEva markers")
    parser.add_argument("--npz", type=Path, required=True, help="Path to npz file")
    parser.add_argument("--frame", type=int, default=None, help="Frame index; if omitted, first non-empty frame is used")
    args = parser.parse_args()

    if not args.npz.exists():
        raise FileNotFoundError(f"File not found: {args.npz}")

    data = np.load(args.npz, allow_pickle=True)
    markers = data["markers_3d"]

    print("markers shape:", markers.shape)
    print("global min/max:", markers.min(), markers.max())

    all_zero_frames = np.all(markers == 0, axis=(1, 2))
    print("all-zero frames:", int(all_zero_frames.sum()), "/", len(markers))

    if args.frame is None:
        frame_idx = find_nonempty_frame(markers)
        if frame_idx is None:
            raise RuntimeError("No non-empty frame found in this file.")
    else:
        frame_idx = max(0, min(args.frame, len(markers) - 1))

    pts = markers[frame_idx]
    valid = ~(np.all(pts == 0, axis=1))
    pts = pts[valid]

    if len(pts) == 0:
        alt = find_nonempty_frame(markers)
        if alt is None:
            raise RuntimeError("No non-empty frame found in this file.")
        print(f"Frame {frame_idx} is empty, switching to frame {alt}")
        frame_idx = alt
        pts = markers[frame_idx]
        valid = ~(np.all(pts == 0, axis=1))
        pts = pts[valid]

    print("selected frame:", frame_idx)
    print("visible markers:", len(pts))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=10)

    ax.set_title(f"{args.npz.name} | frame {frame_idx} | visible markers={len(pts)}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


if __name__ == "__main__":
    main()