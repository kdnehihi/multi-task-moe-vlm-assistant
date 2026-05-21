"""Tests for preprocessing utilities."""

import json
from pathlib import Path

from src.data.preprocessing import build_multitask_jsonl, normalize_record


def make_record(dataset_name: str, task_type: str, image_path: Path) -> dict:
    return {
        "dataset": dataset_name,
        "split": "validation",
        "question_id": f"{dataset_name}-1",
        "question": "What is shown?",
        "answers": ["answer"],
        "image_path": str(image_path),
        "task_type": task_type,
        "unused_field": "drop me",
    }


def test_normalize_record_selects_core_fields(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    record = make_record("docvqa", "document_qa", image_path)

    normalized = normalize_record(record, line_number=1, source_path=tmp_path)

    assert set(normalized) == {
        "dataset",
        "split",
        "task_type",
        "question_id",
        "question",
        "answers",
        "image_path",
    }
    assert normalized["dataset"] == "docvqa"
    assert normalized["task_type"] == "document_qa"
    assert "unused_field" not in normalized


def test_build_multitask_jsonl_merges_datasets(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    output_path = tmp_path / "processed" / "validation.jsonl"

    docvqa_path = raw_root / "docvqa" / "sample"
    chartqa_path = raw_root / "chartqa" / "sample"
    docvqa_path.mkdir(parents=True)
    chartqa_path.mkdir(parents=True)

    doc_record = make_record("docvqa", "document_qa", tmp_path / "doc.png")
    chart_record = make_record("chartqa", "chart_qa", tmp_path / "chart.png")

    (docvqa_path / "validation.jsonl").write_text(
        json.dumps(doc_record) + "\n",
        encoding="utf-8",
    )
    (chartqa_path / "validation.jsonl").write_text(
        json.dumps(chart_record) + "\n",
        encoding="utf-8",
    )

    count = build_multitask_jsonl(
        raw_root=str(raw_root),
        output_path=str(output_path),
        split="validation",
        datasets=("docvqa", "chartqa"),
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]

    assert count == 2
    assert len(records) == 2
    assert records[0]["dataset"] == "docvqa"
    assert records[1]["dataset"] == "chartqa"
