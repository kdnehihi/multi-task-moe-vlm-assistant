"""Evaluation helpers for vision-language QA predictions."""

from src.evaluation.metrics import (
    containment,
    mean_score,
    normalized_exact_match,
    output_token_length,
    raw_exact_match,
    routing_accuracy,
    token_f1,
)


def evaluate_predictions(
    predictions: list[str],
    references: list[dict],
) -> dict:
    """Evaluate answer predictions against reference examples."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length.")

    records = build_prediction_records(predictions, references)

    return {
        "num_examples": len(predictions),
        **summarize_quality_records(records),
    }


def evaluate_predictions_by_task(
    predictions: list[str],
    references: list[dict],
) -> dict:
    """Evaluate answer predictions overall and grouped by task type."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length.")

    records = build_prediction_records(predictions, references)
    task_records = {}

    for record, reference in zip(records, references):
        task_type = reference["task_type"]
        task_records.setdefault(task_type, []).append(record)

    by_task = {
        task_type: summarize_quality_records(records)
        for task_type, records in sorted(task_records.items())
    }

    return {
        "overall": {
            "num_examples": len(records),
            **summarize_quality_records(records),
        },
        "by_task": by_task,
    }


def compute_exact_match_scores(
    predictions: list[str],
    references: list[dict],
) -> list[float]:
    """Compute per-example exact match scores."""
    return [
        normalized_exact_match(prediction, reference["answers"])
        for prediction, reference in zip(predictions, references)
    ]


def build_prediction_records(
    predictions: list[str],
    references: list[dict],
    cleaned_predictions: list[str] | None = None,
) -> list[dict]:
    """Build per-sample records with answer quality metrics."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length.")
    if cleaned_predictions is not None and len(cleaned_predictions) != len(predictions):
        raise ValueError("cleaned_predictions and predictions must have the same length.")

    records = []
    for index, (prediction, reference) in enumerate(zip(predictions, references)):
        cleaned_prediction = (
            cleaned_predictions[index]
            if cleaned_predictions is not None
            else str(prediction).strip()
        )
        answers = reference["answers"]
        records.append(
            {
                "index": index,
                "question": reference.get("question", ""),
                "task_type": reference.get("task_type", ""),
                "answers": answers,
                "chosen_training_answer": reference.get("chosen_training_answer"),
                "raw_prediction": prediction,
                "cleaned_prediction": cleaned_prediction,
                "output_length": output_token_length(cleaned_prediction),
                "raw_exact_match": raw_exact_match(prediction, answers),
                "normalized_em": normalized_exact_match(cleaned_prediction, answers),
                "token_f1": token_f1(cleaned_prediction, answers),
                "containment": containment(cleaned_prediction, answers),
            }
        )
    return records


def summarize_quality_records(records: list[dict]) -> dict:
    """Summarize per-sample answer quality records."""
    lengths = [record["output_length"] for record in records]
    return {
        "num_examples": len(records),
        "raw_exact_match": mean_score([record["raw_exact_match"] for record in records]),
        "normalized_exact_match": mean_score(
            [record["normalized_em"] for record in records]
        ),
        "exact_match": mean_score([record["normalized_em"] for record in records]),
        "token_f1": mean_score([record["token_f1"] for record in records]),
        "containment": mean_score([record["containment"] for record in records]),
        "avg_output_token_length": mean_score(lengths),
        "pct_outputs_gt_10_tokens": mean_score(
            [1.0 if length > 10 else 0.0 for length in lengths]
        ),
        "pct_outputs_gt_20_tokens": mean_score(
            [1.0 if length > 20 else 0.0 for length in lengths]
        ),
    }


def summarize_quality_records_by_task(records: list[dict]) -> dict:
    """Summarize quality records overall and grouped by task type."""
    task_records = {}
    for record in records:
        task_records.setdefault(record["task_type"], []).append(record)

    return {
        "overall": summarize_quality_records(records),
        "by_task": {
            task_type: summarize_quality_records(task_records_for_type)
            for task_type, task_records_for_type in sorted(task_records.items())
        },
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
