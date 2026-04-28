#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter
import re

import numpy as np
import scipy.io


SUBJECTS = ["S1", "S2", "S3", "S4"]
PLACEHOLDER_RE = re.compile(r"^\*\d+$")


def safe_getattr(obj, name, default=None):
    return getattr(obj, name, default) if hasattr(obj, name) else default


def decode_matlab_strings(value):
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S"}:
            if value.ndim == 1:
                return "".join(map(str, value.tolist()))
            if value.ndim == 2:
                return ["".join(map(str, value[i].tolist())).strip() for i in range(value.shape[0])]
        if value.dtype == object:
            return [decode_matlab_strings(x) for x in value.flat]

    return value


def get_group_by_name(parameter_group, group_name):
    if not isinstance(parameter_group, np.ndarray):
        parameter_group = np.array([parameter_group], dtype=object)

    for i, group in enumerate(parameter_group.flat):
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


def extract_labels_from_mat(mat_path: Path):
    data = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    pg = data["ParameterGroup"]

    _, point_group = get_group_by_name(pg, "POINT")
    if point_group is None:
        raise RuntimeError(f"POINT group not found in {mat_path}")

    _, labels_param = get_parameter_by_name(point_group, "LABELS")
    if labels_param is None:
        raise RuntimeError(f"LABELS param not found in {mat_path}")

    raw = safe_getattr(labels_param, "data", None)
    labels = decode_matlab_strings(raw)

    if not isinstance(labels, list):
        raise RuntimeError(f"Unexpected LABELS format in {mat_path}: {type(labels)}")

    return [str(x).strip() for x in labels]


def normalize_label(label: str) -> str | None:
    if not label:
        return None
    if PLACEHOLDER_RE.match(label):
        return None

    # remove prefix like RL: or LS:
    if ":" in label:
        label = label.split(":", 1)[1].strip()

    # discard derived angle/features rather than physical markers
    bad_keywords = ["Angles", "Progress"]
    if any(k.lower() in label.lower() for k in bad_keywords):
        return None

    return label


def main():
    parser = argparse.ArgumentParser(description="Find normalized common HumanEva marker labels")
    parser.add_argument("--root", type=Path, required=True, help="Path to HumanEva raw root")
    args = parser.parse_args()

    mat_files = []
    for subject in SUBJECTS:
        mocap_dir = args.root / subject / "Mocap_Data"
        if mocap_dir.exists():
            mat_files.extend(sorted(mocap_dir.glob("*.mat")))

    if not mat_files:
        raise RuntimeError("No MAT files found")

    label_sets = {}
    label_counts = Counter()

    for mat_path in mat_files:
        labels = extract_labels_from_mat(mat_path)
        norm_labels = [normalize_label(x) for x in labels]
        norm_labels = [x for x in norm_labels if x is not None]
        norm_set = set(norm_labels)

        rel_name = str(mat_path.relative_to(args.root))
        label_sets[rel_name] = norm_set

        for lbl in norm_set:
            label_counts[lbl] += 1

        print(f"{rel_name}: normalized_labels={len(norm_set)}")

    common_labels = set.intersection(*label_sets.values())
    union_labels = set.union(*label_sets.values())

    print("\n=== Summary ===")
    print(f"Sequences scanned: {len(label_sets)}")
    print(f"Union of normalized labels: {len(union_labels)}")
    print(f"Common normalized labels across all sequences: {len(common_labels)}")

    print("\n=== Common normalized labels across all sequences ===")
    for lbl in sorted(common_labels):
        print(lbl)

    print("\n=== Most frequent normalized labels ===")
    for lbl, count in label_counts.most_common(100):
        print(f"{lbl:20s} {count}")

    print("\n=== Labels missing from some sequences ===")
    partial = [(lbl, count) for lbl, count in label_counts.items() if count < len(label_sets)]
    partial = sorted(partial, key=lambda x: (-x[1], x[0]))
    for lbl, count in partial[:100]:
        print(f"{lbl:20s} {count}/{len(label_sets)}")


if __name__ == "__main__":
    main()