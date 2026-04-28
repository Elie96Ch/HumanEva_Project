# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:51:27 2026

@author: user
"""

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

import scipy.io


def describe_value(name, value, indent=0):
    prefix = " " * indent
    try:
        shape = value.shape
    except AttributeError:
        shape = None

    print(f"{prefix}- {name}: type={type(value)}", end="")
    if shape is not None:
        print(f", shape={shape}", end="")
    print()

    # If this is a MATLAB struct-like ndarray/object, try to inspect fields
    if hasattr(value, "dtype") and value.dtype.names is not None:
        print(f"{prefix}  fields: {value.dtype.names}")


def main():
    parser = argparse.ArgumentParser(description="Inspect a HumanEva .mat mocap file")
    parser.add_argument("--mat", type=Path, required=True, help="Path to .mat file")
    args = parser.parse_args()

    if not args.mat.exists():
        raise FileNotFoundError(f"File not found: {args.mat}")

    print(f"Loading: {args.mat}")
    data = scipy.io.loadmat(args.mat, struct_as_record=False, squeeze_me=True)

    print("\n=== Top-level keys ===")
    keys = [k for k in data.keys() if not k.startswith("__")]
    pprint(keys)

    print("\n=== Key details ===")
    for key in keys:
        try:
            describe_value(key, data[key])
        except Exception as e:
            print(f"- {key}: [could not inspect: {e}]")

    print("\n=== Deeper inspection ===")
    for key in keys:
        value = data[key]

        # MATLAB structs loaded by scipy often appear as mat_struct objects
        if hasattr(value, "_fieldnames"):
            print(f"\nStruct: {key}")
            print(f"  fields: {value._fieldnames}")
            for field in value._fieldnames:
                try:
                    field_value = getattr(value, field)
                    describe_value(field, field_value, indent=4)
                except Exception as e:
                    print(f"    - {field}: [could not inspect: {e}]")

        elif hasattr(value, "dtype") and value.dtype.names is not None:
            print(f"\nStructured array: {key}")
            print(f"  fields: {value.dtype.names}")

    print("\nDone.")


if __name__ == "__main__":
    main()