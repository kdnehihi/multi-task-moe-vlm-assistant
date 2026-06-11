"""Tests for evaluator helpers."""

import pytest

from src.evaluation.evaluator import (
    build_prediction_records,
    evaluate_predictions,
    evaluate_predictions_by_task,
    evaluate_routing,
)


def test_evaluate_predictions_computes_exact_match() -> None:
    references = [
        {"answers": ["2021"], "task_type": "chart_qa"},
        {"answers": ["yes"], "task_type": "document_qa"},
    ]

    result = evaluate_predictions(["2021.", "no"], references)

    assert result["num_examples"] == 2
    assert result["exact_match"] == pytest.approx(0.5)
    assert result["normalized_exact_match"] == pytest.approx(0.5)
    assert result["raw_exact_match"] == pytest.approx(0.0)


def test_evaluate_predictions_rejects_length_mismatch() -> None:
    references = [{"answers": ["2021"], "task_type": "chart_qa"}]

    with pytest.raises(ValueError):
        evaluate_predictions(["2021", "2022"], references)


def test_evaluate_predictions_by_task_reports_task_metrics() -> None:
    references = [
        {"answers": ["2021"], "task_type": "chart_qa"},
        {"answers": ["yes"], "task_type": "document_qa"},
        {"answers": ["no"], "task_type": "document_qa"},
    ]

    result = evaluate_predictions_by_task(["2021.", "yes", "maybe"], references)

    assert result["overall"]["num_examples"] == 3
    assert result["overall"]["exact_match"] == pytest.approx(2 / 3)
    assert result["by_task"]["chart_qa"]["num_examples"] == 1
    assert result["by_task"]["chart_qa"]["exact_match"] == 1.0
    assert result["by_task"]["document_qa"]["num_examples"] == 2
    assert result["by_task"]["document_qa"]["exact_match"] == pytest.approx(0.5)
    assert result["overall"]["token_f1"] > 0.0
    assert "avg_output_token_length" in result["overall"]


def test_build_prediction_records_includes_quality_fields() -> None:
    references = [
        {
            "question": "Where?",
            "answers": ["New York"],
            "task_type": "textvqa",
            "chosen_training_answer": "New York",
        }
    ]

    records = build_prediction_records(["The answer is New York."], references)

    assert records[0]["question"] == "Where?"
    assert records[0]["chosen_training_answer"] == "New York"
    assert records[0]["raw_prediction"] == "The answer is New York."
    assert records[0]["cleaned_prediction"] == "The answer is New York."
    assert records[0]["normalized_em"] == 0.0
    assert records[0]["containment"] == 1.0
    assert records[0]["output_length"] == 5


def test_evaluate_predictions_by_task_rejects_length_mismatch() -> None:
    references = [{"answers": ["2021"], "task_type": "chart_qa"}]

    with pytest.raises(ValueError):
        evaluate_predictions_by_task(["2021", "2022"], references)


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
