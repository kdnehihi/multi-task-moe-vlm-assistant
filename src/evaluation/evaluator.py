"""Evaluation helpers for vision-language QA predictions."""

from src.evaluation.metrics import exact_match, mean_score, routing_accuracy


def evaluate_predictions(
    predictions: list[str],
    references: list[dict],
) -> dict:
    """Evaluate answer predictions against reference examples."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length.")

    exact_match_scores = compute_exact_match_scores(predictions, references)

    return {
        "num_examples": len(predictions),
        "exact_match": mean_score(exact_match_scores),
    }


def evaluate_predictions_by_task(
    predictions: list[str],
    references: list[dict],
) -> dict:
    """Evaluate answer predictions overall and grouped by task type."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length.")

    task_scores = {}

    for prediction, reference in zip(predictions, references):
        task_type = reference["task_type"]
        score = exact_match(prediction, reference["answers"])
        task_scores.setdefault(task_type, []).append(score)

    by_task = {
        task_type: {
            "num_examples": len(scores),
            "exact_match": mean_score(scores),
        }
        for task_type, scores in sorted(task_scores.items())
    }

    return {
        "overall": evaluate_predictions(predictions, references),
        "by_task": by_task,
    }


def compute_exact_match_scores(
    predictions: list[str],
    references: list[dict],
) -> list[float]:
    """Compute per-example exact match scores."""
    return [
        exact_match(prediction, reference["answers"])
        for prediction, reference in zip(predictions, references)
    ]


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
