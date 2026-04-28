#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import List, Dict

import numpy as np
import scipy.io


SUBJECTS = ["S1", "S2", "S3", "S4"]
PLACEHOLDER_RE = re.compile(r"^\*\d+$")

COMMON_LABELS = [
    "C7", "CLAV", "HEDA", "HEDL", "HEDO", "HEDP",
    "LANK", "LASI", "LBHD", "LCLA", "LCLL", "LCLO", "LCLP",
    "LELB", "LFEA", "LFEL", "LFEO", "LFEP", "LFHD", "LFIN",
    "LFOA", "LFOL", "LFOO", "LFOP", "LFRA", "LHEE",
    "LHNA", "LHNL", "LHNO", "LHNP", "LHUA", "LHUL", "LHUO", "LHUP",
    "LKNE", "LPSI", "LRAA", "LRAL", "LRAO", "LRAP",
    "LSHO", "LTHI", "LTIA", "LTIB", "LTIL", "LTIO", "LTIP",
    "LTOA", "LTOE", "LTOL", "LTOO", "LTOP", "LUPA", "LWRA", "LWRB",
    "PELA", "PELL", "PELO", "PELP",
    "RANK", "RASI", "RBAK", "RBHD", "RCLA", "RCLL", "RCLO", "RCLP",
    "RELB", "RFHD", "RFIN", "RFRA", "RHEE", "RKNE", "RPSI",
    "RSHO", "RTIB", "RTOE", "RUPA", "RWRA",
    "STRN", "T10", "TRXA", "TRXL", "TRXO", "TRXP",
]


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

    for _, group in enumerate(parameter_group.flat):
        name = safe_getattr(group, "name", None)
        if name == group_name:
            return group
    return None


def get_parameter_by_name(group, param_name):
    params = safe_getattr(group, "Parameter", None)
    if params is None:
        return None

    if not isinstance(params, np.ndarray):
        params = np.array([params], dtype=object)

    for param in params.flat:
        name = safe_getattr(param, "name", None)
        if name == param_name:
            return param
    return None


def normalize_label(label: str) -> str | None:
    if not label:
        return None
    if PLACEHOLDER_RE.match(label):
        return None

    if ":" in label:
        label = label.split(":", 1)[1].strip()

    bad_keywords = ["Angles", "Progress", "Force", "Moment", "Power"]
    if any(k.lower() in label.lower() for k in bad_keywords):
        return None

    return label


def extract_labels_and_markers(mat_path: Path):
    data = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

    markers = np.asarray(data["Markers"], dtype=np.float32)
    parameter_group = data["ParameterGroup"]

    point_group = get_group_by_name(parameter_group, "POINT")
    if point_group is None:
        raise RuntimeError(f"POINT group not found in {mat_path}")

    labels_param = get_parameter_by_name(point_group, "LABELS")
    if labels_param is None:
        raise RuntimeError(f"LABELS parameter not found in {mat_path}")

    raw_labels = safe_getattr(labels_param, "data", None)
    labels = decode_matlab_strings(raw_labels)
    labels = [str(x).strip() for x in labels]
    norm_labels = [normalize_label(x) for x in labels]

    return markers, labels, norm_labels


def build_label_index(norm_labels: List[str]) -> Dict[str, int]:
    out = {}
    for i, lbl in enumerate(norm_labels):
        if lbl is None:
            continue
        if lbl not in out:
            out[lbl] = i
    return out


def main():
    parser = argparse.ArgumentParser(description="Extract fixed common HumanEva marker subset")
    parser.add_argument("--root", type=Path, required=True, help="Path to HumanEva raw root")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    args = parser.parse_args()

    mat_files = []
    for subject in SUBJECTS:
        mocap_dir = args.root / subject / "Mocap_Data"
        if mocap_dir.exists():
            mat_files.extend(sorted(mocap_dir.glob("*.mat")))

    if not mat_files:
        raise RuntimeError("No MAT files found")

    args.out.mkdir(parents=True, exist_ok=True)

    for mat_path in mat_files:
        subject = mat_path.parts[-3]
        action = mat_path.stem

        markers, raw_labels, norm_labels = extract_labels_and_markers(mat_path)
        label_to_idx = build_label_index(norm_labels)

        missing = [lbl for lbl in COMMON_LABELS if lbl not in label_to_idx]
        if missing:
            print(f"[WARN] {mat_path.name}: missing {len(missing)} common labels")

        selected = np.zeros((markers.shape[0], len(COMMON_LABELS), 3), dtype=np.float32)
        present_mask = np.zeros((len(COMMON_LABELS),), dtype=np.uint8)

        for j, lbl in enumerate(COMMON_LABELS):
            idx = label_to_idx.get(lbl, None)
            if idx is not None and idx < markers.shape[1]:
                selected[:, j, :] = markers[:, idx, :]
                present_mask[j] = 1

        out_dir = args.out / subject
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{action}.npz"

        np.savez_compressed(
            out_path,
            subject=subject,
            action=action,
            marker_labels=np.array(COMMON_LABELS, dtype=object),
            markers_3d=selected,
            present_mask=present_mask,
        )

        print(f"Saved {out_path} | shape={selected.shape}")

    print("Done.")


if __name__ == "__main__":
    main()