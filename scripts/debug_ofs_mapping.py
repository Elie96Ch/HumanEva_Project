# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:38:52 2026

@author: user
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def read_ofs(path):
    values = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values.extend([float(x) for x in line.split()])

    if len(values) != 3:
        raise ValueError(f"Expected 3 values in {path}, got {len(values)}: {values}")

    im_st, mc_st, mc_sc = values
    return im_st, mc_st, mc_sc


def image_to_mocap(image_idx, im_st, mc_st, mc_sc):
    return mc_st + (image_idx - im_st) * mc_sc


def mocap_to_image(mocap_idx, im_st, mc_st, mc_sc):
    return im_st + (mocap_idx - mc_st) / mc_sc


def main():
    subject = "S1"
    action = "Walking_1"
    camera = "C1"

    ofs_path = (
        PROJECT_ROOT
        / "data"
        / "humaneva"
        / "raw"
        / subject
        / "Sync_Data"
        / f"{action}_({camera}).ofs"
    )

    im_st, mc_st, mc_sc = read_ofs(ofs_path)

    print("OFS:", ofs_path)
    print("im_st:", im_st)
    print("mc_st:", mc_st)
    print("mc_sc:", mc_sc)
    print()

    print("Image/video frame -> mocap frame")
    for image_idx in [1, 50, 100, 200, 300, 400, 500]:
        mocap_idx = image_to_mocap(image_idx, im_st, mc_st, mc_sc)
        print(f"image {image_idx:5d} -> mocap {mocap_idx:.3f}")

    print()

    print("Mocap frame -> image/video frame")
    for mocap_idx in [1, 100, 200, 400, 705, 1000]:
        image_idx = mocap_to_image(mocap_idx, im_st, mc_st, mc_sc)
        print(f"mocap {mocap_idx:5d} -> image {image_idx:.3f}")


if __name__ == "__main__":
    main()