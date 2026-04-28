# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:43:26 2026

@author: user
"""

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Inspect a processed HumanEva NPZ file")
    parser.add_argument("--npz", type=Path, required=True, help="Path to npz file")
    args = parser.parse_args()

    if not args.npz.exists():
        raise FileNotFoundError(f"File not found: {args.npz}")

    data = np.load(args.npz, allow_pickle=True)

    print(f"File: {args.npz}\n")
    print("Keys:")
    for key in data.files:
        arr = data[key]
        print(f"  - {key}: type={type(arr)}, shape={getattr(arr, 'shape', None)}, dtype={getattr(arr, 'dtype', None)}")

    print("\nSample values:")
    for key in data.files:
        arr = data[key]
        if np.isscalar(arr) or arr.shape == ():
            print(f"  {key} = {arr.item() if hasattr(arr, 'item') else arr}")


if __name__ == "__main__":
    main()