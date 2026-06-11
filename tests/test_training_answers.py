"""Tests for fine-tuning answer target selection."""

from src.data.answers import choose_training_answer


def test_choose_training_answer_preserves_original_text() -> None:
    answer = choose_training_answer(["New York", "new york", "New York City"], "textvqa")

    assert answer == "New York"


def test_choose_training_answer_uses_shortest_original_from_winning_group() -> None:
    answer = choose_training_answer(["May 5, 2021", "May 5 2021", "May 5, 2021"], "docvqa")

    assert answer == "May 5 2021"


def test_choose_training_answer_maps_no_answer_only_when_majority() -> None:
    assert choose_training_answer(["unanswerable", "No Answer", "2021"], "docvqa") == "unanswerable"
    assert choose_training_answer(["unanswerable", "2021"], "docvqa") == "2021"


def test_choose_training_answer_does_not_concatenate_answers() -> None:
    answer = choose_training_answer(["red", "blue", "red"], "chartqa")

    assert answer == "red"
    assert "," not in answer
