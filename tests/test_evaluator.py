"""Tests for evaluator helpers."""

import pytest

from src.evaluation.evaluator import evaluate_predictions, evaluate_routing


def test_evaluate_predictions_computes_exact_match() -> None:
    references = [
        {"answers": ["2021"], "task_type": "chart_qa"},
        {"answers": ["yes"], "task_type": "document_qa"},
    ]

    result = evaluate_predictions(["2021.", "no"], references)

    assert result["num_examples"] == 2
    assert result["exact_match"] == pytest.approx(0.5)


def test_evaluate_predictions_rejects_length_mismatch() -> None:
    references = [{"answers": ["2021"], "task_type": "chart_qa"}]

    with pytest.raises(ValueError):
        evaluate_predictions(["2021", "2022"], references)


def test_evaluate_routing_computes_accuracy() -> None:
    references = [
        {"answers": ["2021"], "task_type": "chart_qa"},
        {"answers": ["yes"], "task_type": "document_qa"},
        {"answers": ["cat"], "task_type": "image_vqa"},
    ]

    result = evaluate_routing(
        ["chart_qa", "document_qa", "chart_qa"],
        references,
    )

    assert result["num_examples"] == 3
    assert result["routing_accuracy"] == pytest.approx(2 / 3)


def test_evaluate_routing_rejects_length_mismatch() -> None:
    references = [
        {"answers": ["2021"], "task_type": "chart_qa"},
        {"answers": ["yes"], "task_type": "document_qa"},
    ]

    with pytest.raises(ValueError):
        evaluate_routing(["chart_qa"], references)

