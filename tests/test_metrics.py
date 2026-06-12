"""Tests for evaluation metrics."""

import pytest

from src.evaluation.metrics import (
    anls,
    containment,
    exact_match,
    mean_score,
    normalize_answer,
    output_token_length,
    raw_exact_match,
    relaxed_numeric_accuracy,
    routing_accuracy,
    strict_containment,
    token_f1,
    vqa_soft_score,
)


def test_normalize_answer() -> None:
    assert normalize_answer(" The Revenue, 2021! ") == "revenue 2021"


def test_exact_match_accepts_any_ground_truth_answer() -> None:
    assert exact_match("2021.", ["2020", "2021"]) == 1.0


def test_raw_exact_match_preserves_surface_form() -> None:
    assert raw_exact_match("New York", ["New York"]) == 1.0
    assert raw_exact_match("new york", ["New York"]) == 0.0


def test_token_f1_uses_best_reference() -> None:
    assert token_f1("new york city", ["new york"]) == pytest.approx(0.8)


def test_strict_containment_detects_answer_span_without_substring_matches() -> None:
    assert strict_containment("the answer is New York City", ["New York"]) == 1.0
    assert strict_containment("CMROJOURNAL", ["CMRO"]) == 0.0
    assert strict_containment("notebook", ["no"]) == 0.0
    assert strict_containment("Gray (light grey)", ["gray"]) == 1.0
    assert strict_containment("64%", ["64"]) == 0.0
    assert strict_containment("19346.08", ["19346"]) == 0.0
    assert containment("CMROJOURNAL", ["CMRO"]) == 0.0


def test_relaxed_numeric_accuracy() -> None:
    assert relaxed_numeric_accuracy("19346.08", ["19346"]) == 1.0
    assert relaxed_numeric_accuracy("15.08647", ["15"]) == 1.0
    assert relaxed_numeric_accuracy("64%", ["64"]) == 1.0
    assert relaxed_numeric_accuracy("2.465833", ["10.13"]) == 0.0


def test_anls_thresholds_low_similarity() -> None:
    assert anls("Bausch Lomb", ["Bausch & Lomb"]) > 0.5
    assert anls("zzzz", ["Bausch & Lomb"]) == 0.0


def test_vqa_soft_score_counts_repeated_references() -> None:
    assert vqa_soft_score("beauregard", ["beauregard", "beauregard", "chablis"]) == pytest.approx(2 / 3)
    assert vqa_soft_score("unanswerable", ["no answer", "cannot be determined", "blue"]) == pytest.approx(2 / 3)


def test_output_token_length() -> None:
    assert output_token_length("New York City") == 3


def test_exact_match_rejects_wrong_answer() -> None:
    assert exact_match("2022", ["2021"]) == 0.0


def test_mean_score() -> None:
    assert mean_score([1.0, 0.0, 1.0]) == pytest.approx(2 / 3)


def test_mean_score_empty_list() -> None:
    assert mean_score([]) == 0.0


def test_routing_accuracy() -> None:
    predicted_tasks = ["chart_qa", "document_qa", "image_vqa"]
    target_tasks = ["chart_qa", "document_qa", "document_qa"]

    assert routing_accuracy(predicted_tasks, target_tasks) == pytest.approx(2 / 3)


def test_routing_accuracy_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        routing_accuracy(["chart_qa"], ["chart_qa", "document_qa"])
