# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:26:43 2026

@author: user
"""

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Inspect a HumanEva .mp file")
    parser.add_argument("--mp", type=Path, required=True, help="Path to .mp file")
    parser.add_argument("--max-lines", type=int, default=200, help="Number of lines to print")
    args = parser.parse_args()

    if not args.mp.exists():
        raise FileNotFoundError(f"File not found: {args.mp}")

    print(f"Reading: {args.mp}\n")
    with open(args.mp, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            print(f"{i:04d}: {line.rstrip()}")
            if i >= args.max_lines:
                break

    print("\nDone.")


if __name__ == "__main__":
    main()