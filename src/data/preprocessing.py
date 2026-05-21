"""Preprocessing utilities for normalized multi-task VQA records."""

from pathlib import Path
import json


DEFAULT_DATASETS = ("docvqa", "chartqa", "textvqa")

SELECTED_FIELDS = (
    "dataset",
    "split",
    "task_type",
    "question_id",
    "question",
    "answers",
    "image_path",
)

REQUIRED_FIELDS = {
    "dataset",
    "split",
    "task_type",
    "question",
    "answers",
    "image_path",
}


def get_raw_metadata_paths(
    raw_root: str,
    split: str,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
) -> list[Path]:
    """Return existing raw JSONL metadata paths for the requested split."""
    raw_root_path = Path(raw_root)
    paths = []

    for dataset_name in datasets:
        metadata_path = raw_root_path / dataset_name / "sample" / f"{split}.jsonl"
        if metadata_path.exists():
            paths.append(metadata_path)

    return paths


def normalize_record(record: dict, line_number: int, source_path: Path) -> dict:
    """Select core fields and validate one normalized VQA record."""
    missing_fields = REQUIRED_FIELDS - set(record)

    if missing_fields:
        raise ValueError(
            f"Missing fields in {source_path} at line {line_number}: "
            f"{sorted(missing_fields)}"
        )

    normalized = {field: record.get(field) for field in SELECTED_FIELDS}
    normalized["answers"] = [str(answer) for answer in normalized["answers"]]
    normalized["question"] = str(normalized["question"])
    normalized["image_path"] = str(normalized["image_path"])

    return normalized


def read_jsonl(path: Path) -> list[dict]:
    """Read normalized VQA records from a JSONL file."""
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            record = json.loads(line)
            records.append(normalize_record(record, line_number, path))

    return records


def write_jsonl(records: list[dict], path: str) -> None:
    """Write records to a JSONL file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def merge_metadata_files(input_paths: list[Path]) -> list[dict]:
    """Merge normalized records from multiple dataset metadata files."""
    merged_records = []

    for input_path in input_paths:
        merged_records.extend(read_jsonl(input_path))

    return merged_records


def build_multitask_jsonl(
    raw_root: str,
    output_path: str,
    split: str,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
) -> int:
    """Merge raw dataset JSONL files into one processed multi-task JSONL file."""
    input_paths = get_raw_metadata_paths(raw_root, split, datasets)

    if not input_paths:
        raise FileNotFoundError(
            f"No raw metadata files found under {raw_root} for split '{split}'."
        )

    records = merge_metadata_files(input_paths)
    write_jsonl(records, output_path)

    return len(records)
