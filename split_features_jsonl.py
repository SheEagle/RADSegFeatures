import argparse
import math
import os
from pathlib import Path


def count_lines(input_path: Path) -> int:
    total = 0
    with input_path.open("r", encoding="utf-8") as handle:
        for _ in handle:
            total += 1
    return total


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_shard_path(output_dir: Path, stem: str, shard_index: int, shard_count: int) -> Path:
    width = max(3, len(str(shard_count)))
    filename = f"{stem}.part_{shard_index:0{width}d}_of_{shard_count:0{width}d}.jsonl"
    return output_dir / filename


def split_jsonl(input_path: Path, output_dir: Path, lines_per_shard: int) -> None:
    total_lines = count_lines(input_path)
    if total_lines == 0:
        raise ValueError(f"Input file is empty: {input_path}")

    shard_count = math.ceil(total_lines / lines_per_shard)
    stem = input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input file: {input_path}")
    print(f"Total records: {total_lines}")
    print(f"Lines per shard: {lines_per_shard}")
    print(f"Output shards: {shard_count}")
    print(f"Output directory: {output_dir}")

    current_shard_index = 1
    current_line_in_shard = 0
    current_shard_path = build_shard_path(output_dir, stem, current_shard_index, shard_count)
    ensure_parent(current_shard_path)
    current_handle = current_shard_path.open("w", encoding="utf-8")

    try:
        with input_path.open("r", encoding="utf-8") as input_handle:
            for line_index, line in enumerate(input_handle, start=1):
                if current_line_in_shard >= lines_per_shard:
                    current_handle.close()
                    print(
                        f"Finished shard {current_shard_index}/{shard_count}: "
                        f"{current_shard_path.name}"
                    )
                    current_shard_index += 1
                    current_line_in_shard = 0
                    current_shard_path = build_shard_path(output_dir, stem, current_shard_index, shard_count)
                    current_handle = current_shard_path.open("w", encoding="utf-8")

                current_handle.write(line)
                current_line_in_shard += 1

                if line_index % 1000 == 0:
                    print(f"Processed {line_index}/{total_lines} records...")
    finally:
        current_handle.close()

    print(f"Finished shard {current_shard_index}/{shard_count}: {current_shard_path.name}")
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a large features JSONL file into smaller JSONL shards.")
    parser.add_argument("input_file", type=str, help="Path to the source JSONL file")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for shard files. Defaults to '<input_stem>_shards' beside the input file.",
    )
    parser.add_argument(
        "--lines_per_shard",
        type=int,
        default=None,
        help="Number of image records per shard",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="Split into this many shards instead of specifying lines per shard",
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if (args.lines_per_shard is None) == (args.num_shards is None):
        raise ValueError("Specify exactly one of --lines_per_shard or --num_shards.")

    if args.lines_per_shard is not None and args.lines_per_shard <= 0:
        raise ValueError("--lines_per_shard must be > 0.")

    if args.num_shards is not None and args.num_shards <= 0:
        raise ValueError("--num_shards must be > 0.")

    output_dir = Path(args.output_dir) if args.output_dir else input_path.with_name(f"{input_path.stem}_shards")

    if args.num_shards is not None:
        total_lines = count_lines(input_path)
        if total_lines == 0:
            raise ValueError(f"Input file is empty: {input_path}")
        lines_per_shard = math.ceil(total_lines / args.num_shards)
    else:
        lines_per_shard = args.lines_per_shard

    split_jsonl(input_path=input_path, output_dir=output_dir, lines_per_shard=lines_per_shard)


if __name__ == "__main__":
    main()
