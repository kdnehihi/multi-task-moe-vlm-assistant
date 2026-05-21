"""Build a processed multi-task VQA JSONL file from raw dataset samples."""

from argparse import ArgumentParser
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.preprocessing import DEFAULT_DATASETS, build_multitask_jsonl


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--split", default="validation")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--output-path", default=None)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=DEFAULT_DATASETS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = (
        args.output_path
        or f"data/processed/multitask/{args.split}.jsonl"
    )

    count = build_multitask_jsonl(
        raw_root=args.raw_root,
        output_path=output_path,
        split=args.split,
        datasets=tuple(args.datasets),
    )

    print(f"Saved {count} processed examples to {output_path}", flush=True)


if __name__ == "__main__":
    main()
