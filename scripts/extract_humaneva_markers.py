# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:41:34 2026

@author: user
"""

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
import scipy.io


SUBJECTS = ["S1", "S2", "S3", "S4"]


def parse_action_name(path: Path) -> str:
    return path.stem


def load_mat_markers(mat_path: Path):
    data = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

    if "Markers" not in data:
        raise KeyError(f"'Markers' not found in {mat_path}")

    markers = np.asarray(data["Markers"], dtype=np.float32)

    video_frame_rate = data.get("VideoFrameRate", None)
    analog_frame_rate = data.get("AnalogFrameRate", None)

    return {
        "markers_3d": markers,  # (T, M, 3)
        "video_frame_rate": int(video_frame_rate) if video_frame_rate is not None else -1,
        "analog_frame_rate": int(analog_frame_rate) if analog_frame_rate is not None else -1,
    }


def infer_subject_from_path(path: Path) -> str:
    for part in path.parts:
        if part in SUBJECTS:
            return part
    raise ValueError(f"Could not infer subject from path: {path}")


def save_sequence(out_path: Path, subject: str, action: str, payload: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        subject=subject,
        action=action,
        markers_3d=payload["markers_3d"],
        video_frame_rate=payload["video_frame_rate"],
        analog_frame_rate=payload["analog_frame_rate"],
    )


def main():
    parser = argparse.ArgumentParser(description="Extract HumanEva marker sequences from MAT files")
    parser.add_argument("--root", type=Path, required=True, help="Path to HumanEva raw root")
    parser.add_argument("--out", type=Path, required=True, help="Output processed folder")
    args = parser.parse_args()

    if not args.root.exists():
        raise FileNotFoundError(f"Root folder not found: {args.root}")

    mat_files = []
    for subject in SUBJECTS:
        mocap_dir = args.root / subject / "Mocap_Data"
        if mocap_dir.exists():
            mat_files.extend(sorted(mocap_dir.glob("*.mat")))

    if not mat_files:
        raise RuntimeError("No .mat files found under Mocap_Data")

    print(f"Found {len(mat_files)} MAT files")

    for mat_path in mat_files:
        subject = infer_subject_from_path(mat_path)
        action = parse_action_name(mat_path)

        payload = load_mat_markers(mat_path)

        out_path = args.out / subject / f"{action}.npz"
        save_sequence(out_path, subject, action, payload)

        print(
            f"Saved {out_path} | shape={payload['markers_3d'].shape} "
            f"| video_fps={payload['video_frame_rate']}"
        )

    print("Done.")


if __name__ == "__main__":
    main()