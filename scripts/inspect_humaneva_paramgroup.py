#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import scipy.io


def safe_getattr(obj, name, default=None):
    return getattr(obj, name, default) if hasattr(obj, name) else default


def print_basic(obj, name="obj", indent=0):
    prefix = " " * indent
    print(f"{prefix}{name}: type={type(obj)}")
    if isinstance(obj, np.ndarray):
        print(f"{prefix}  shape={obj.shape}, dtype={obj.dtype}")


def decode_matlab_strings(value):
    if isinstance(value, str):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, np.ndarray):
        # char array or object array
        if value.dtype.kind in {"U", "S"}:
            if value.ndim == 1:
                return "".join(map(str, value.tolist()))
            if value.ndim == 2:
                rows = []
                for i in range(value.shape[0]):
                    rows.append("".join(map(str, value[i].tolist())).strip())
                return rows
        if value.dtype == object:
            out = []
            for x in value.flat:
                out.append(decode_matlab_strings(x))
            return out

    return value


def get_group_by_name(parameter_group, group_name):
    for i, group in enumerate(parameter_group):
        name = safe_getattr(group, "name", None)
        if name == group_name:
            return i, group
    return None, None


def get_parameter_by_name(group, param_name):
    params = safe_getattr(group, "Parameter", None)
    if params is None:
        return None, None

    if not isinstance(params, np.ndarray):
        params = np.array([params], dtype=object)

    for i, param in enumerate(params.flat):
        name = safe_getattr(param, "name", None)
        if name == param_name:
            return i, param
    return None, None


def inspect_parameter(param, title):
    print(f"\n=== {title} ===")
    print_basic(param, "parameter")

    for field in ["name", "datatype", "dim", "data"]:
        value = safe_getattr(param, field, None)
        print(f"\nField: {field}")
        if value is None:
            print("  <missing>")
            continue

        print_basic(value, field, indent=2)

        if field == "data":
            decoded = decode_matlab_strings(value)
            print("\nDecoded data preview:")
            if isinstance(decoded, list):
                for i, item in enumerate(decoded[:50]):
                    print(f"  [{i}] {item}")
                if len(decoded) > 50:
                    print(f"  ... and {len(decoded) - 50} more")
            else:
                print(decoded)


def main():
    parser = argparse.ArgumentParser(description="Inspect HumanEva POINT/LABELS parameter")
    parser.add_argument("--mat", type=Path, required=True, help="Path to HumanEva .mat file")
    args = parser.parse_args()

    if not args.mat.exists():
        raise FileNotFoundError(f"File not found: {args.mat}")

    print(f"Loading: {args.mat}\n")
    data = scipy.io.loadmat(args.mat, struct_as_record=False, squeeze_me=True)

    if "ParameterGroup" not in data:
        raise KeyError("'ParameterGroup' not found")

    parameter_group = data["ParameterGroup"]
    print_basic(parameter_group, "ParameterGroup")

    if not isinstance(parameter_group, np.ndarray):
        parameter_group = np.array([parameter_group], dtype=object)

    group_idx, point_group = get_group_by_name(parameter_group.flat, "POINT")
    if point_group is None:
        raise RuntimeError("POINT group not found")

    print(f"\nFound POINT group at index {group_idx}")

    param_idx, labels_param = get_parameter_by_name(point_group, "LABELS")
    if labels_param is None:
        raise RuntimeError("LABELS parameter not found inside POINT group")

    print(f"Found LABELS parameter at index {param_idx}")
    inspect_parameter(labels_param, "POINT -> LABELS")

    param_idx2, desc_param = get_parameter_by_name(point_group, "DESCRIPTIONS")
    if desc_param is not None:
        print(f"\nFound DESCRIPTIONS parameter at index {param_idx2}")
        inspect_parameter(desc_param, "POINT -> DESCRIPTIONS")


if __name__ == "__main__":
    main()