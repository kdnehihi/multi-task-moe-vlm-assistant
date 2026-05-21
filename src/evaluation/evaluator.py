"""Evaluation helpers for vision-language QA predictions."""

from src.evaluation.metrics import exact_match, mean_score, routing_accuracy


def evaluate_predictions(
    predictions: list[str],
    references: list[dict],
) -> dict:
    """Evaluate answer predictions against reference examples."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length.")

    exact_match_scores = []

    for prediction, reference in zip(predictions, references):
        score = exact_match(prediction, reference["answers"])
        exact_match_scores.append(score)

    return {
        "num_examples": len(predictions),
        "exact_match": mean_score(exact_match_scores),
    }


def evaluate_routing(
    predicted_tasks: list[str],
    references: list[dict],
) -> dict:
    """Evaluate task routing predictions against reference task labels."""
    target_tasks = [reference["task_type"] for reference in references]

    return {
        "num_examples": len(predicted_tasks),
        "routing_accuracy": routing_accuracy(predicted_tasks, target_tasks),
    }
