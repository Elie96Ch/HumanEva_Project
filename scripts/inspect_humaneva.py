# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 23:48:29 2026

@author: user
"""

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter, defaultdict


TEXT_EXTENSIONS = {
    ".txt", ".xml", ".csv", ".json", ".yaml", ".yml", ".mat",
    ".cdf", ".amc", ".asf", ".bvh"
}

MEDIA_EXTENSIONS = {
    ".avi", ".mp4", ".mov", ".mkv", ".jpg", ".jpeg", ".png", ".bmp"
}


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def print_tree(root: Path, max_depth: int = 3, max_entries_per_dir: int = 20) -> None:
    print("\n=== Folder tree ===")
    print(root.resolve())

    def _walk(current: Path, prefix: str = "", depth: int = 0) -> None:
        if depth > max_depth:
            return

        try:
            entries = sorted(
                current.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower())
            )
        except PermissionError:
            print(f"{prefix}[Permission denied] {current.name}")
            return

        shown = 0
        for entry in entries:
            if shown >= max_entries_per_dir:
                remaining = len(entries) - max_entries_per_dir
                print(f"{prefix}└── ... ({remaining} more entries)")
                break

            connector = "└── " if shown == len(entries) - 1 else "├── "
            print(f"{prefix}{connector}{entry.name}")
            shown += 1

            if entry.is_dir():
                extension = "    " if connector == "└── " else "│   "
                _walk(entry, prefix + extension, depth + 1)

    _walk(root)


def scan_files(root: Path) -> dict:
    stats = {
        "total_files": 0,
        "total_size": 0,
        "extensions": Counter(),
        "top_level_dirs": Counter(),
        "possible_annotations": [],
        "possible_media": [],
        "possible_calibration": [],
        "possible_mocap": [],
        "subjects": defaultdict(list),
    }

    calibration_keywords = {"calib", "camera", "intrinsic", "extrinsic", "projection"}
    mocap_keywords = {"mocap", "pose", "3d", "skeleton", "joint", "c3d"}

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        stats["total_files"] += 1
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        stats["total_size"] += size

        suffix = path.suffix.lower()
        stats["extensions"][suffix] += 1

        relative_parts = path.relative_to(root).parts
        if relative_parts:
            stats["top_level_dirs"][relative_parts[0]] += 1

        lower_name = path.name.lower()
        lower_path = str(path).lower()

        if suffix in TEXT_EXTENSIONS:
            stats["possible_annotations"].append(path)

        if suffix in MEDIA_EXTENSIONS:
            stats["possible_media"].append(path)

        if any(k in lower_name or k in lower_path for k in calibration_keywords):
            stats["possible_calibration"].append(path)

        if any(k in lower_name or k in lower_path for k in mocap_keywords):
            stats["possible_mocap"].append(path)

        for subject in ["s1", "s2", "s3", "s4"]:
            if subject in lower_path:
                stats["subjects"][subject.upper()].append(path)
                break

    return stats


def print_summary(stats: dict, show_examples: int = 10) -> None:
    print("\n=== Summary ===")
    print(f"Total files: {stats['total_files']}")
    print(f"Total size : {human_size(stats['total_size'])}")

    print("\nTop file extensions:")
    for ext, count in stats["extensions"].most_common(15):
        label = ext if ext else "[no extension]"
        print(f"  {label:12s} {count}")

    print("\nTop-level folders by file count:")
    for folder, count in stats["top_level_dirs"].most_common():
        print(f"  {folder:20s} {count}")

    print("\nSubject presence:")
    for subject in ["S1", "S2", "S3", "S4"]:
        print(f"  {subject}: {len(stats['subjects'].get(subject, []))} files")

    def _print_examples(title: str, items: list[Path]) -> None:
        print(f"\n{title} ({len(items)} found):")
        for item in items[:show_examples]:
            print(f"  - {item}")
        if len(items) > show_examples:
            print(f"  ... and {len(items) - show_examples} more")

    _print_examples("Possible annotation files", stats["possible_annotations"])
    _print_examples("Possible media files", stats["possible_media"])
    _print_examples("Possible calibration files", stats["possible_calibration"])
    _print_examples("Possible mocap / 3D files", stats["possible_mocap"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect an extracted HumanEva dataset folder."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Path to extracted HumanEva root folder"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Max depth for folder tree printing"
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=20,
        help="Max entries shown per directory in tree"
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=10,
        help="How many example file paths to print per category"
    )
    args = parser.parse_args()

    if not args.root.exists():
        raise FileNotFoundError(f"Path does not exist: {args.root}")
    if not args.root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {args.root}")

    print_tree(args.root, max_depth=args.max_depth, max_entries_per_dir=args.max_entries)
    stats = scan_files(args.root)
    print_summary(stats, show_examples=args.show_examples)

    print("\n=== Next step ===")
    print("Use the printed candidate files to identify:")
    print("  1) where 3D pose annotations are stored")
    print("  2) where image/video data are stored")
    print("  3) where camera calibration files are stored")
    print("Then we can write preprocess_humaneva.py.")


if __name__ == "__main__":
    main()