from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description="Inspect a HumanEva calibration file")
    parser.add_argument("--cal", type=Path, required=True, help="Path to .cal file")
    parser.add_argument("--max-lines", type=int, default=200, help="Number of lines to print")
    args = parser.parse_args()

    if not args.cal.exists():
        raise FileNotFoundError("File not found: {}".format(args.cal))

    print("Reading:", args.cal)
    print()

    with open(args.cal, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            print("{:04d}: {}".format(i, line.rstrip()))
            if i >= args.max_lines:
                break


if __name__ == "__main__":
    main()