#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import scipy.io


def main():
    parser = argparse.ArgumentParser(description="Inspect MATLAB-exported HumanEva body pose file")
    parser.add_argument("--mat", type=Path, required=True, help="Path to exported .mat file")
    args = parser.parse_args()

    if not args.mat.exists():
        raise FileNotFoundError(f"File not found: {args.mat}")

    data = scipy.io.loadmat(args.mat, squeeze_me=True, struct_as_record=False)

    print(f"File: {args.mat}\n")
    print("Keys:")
    for k, v in data.items():
        if k.startswith("__"):
            continue
        shape = getattr(v, "shape", None)
        dtype = getattr(v, "dtype", None)
        print(f"  - {k}: type={type(v)}, shape={shape}, dtype={dtype}")

    pose_3d = data["pose_3d"]
    valid = data["valid"]
    point_names = data["point_names"]

    print("\nSummary:")
    print("  pose_3d shape:", pose_3d.shape)
    print("  valid shape  :", valid.shape)
    print("  valid frames :", int(np.sum(valid)))
    print("  invalid frames:", int(len(valid) - np.sum(valid)))

    print("\nPoint names:")
    if isinstance(point_names, np.ndarray):
        for i, name in enumerate(point_names.tolist()):
            print(f"  [{i}] {name}")
    else:
        print(point_names)


if __name__ == "__main__":
    main()