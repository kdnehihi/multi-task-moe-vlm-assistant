"""Evaluation metrics for vision-language QA."""

import re
import string


def normalize_answer(text: str) -> str:
    """Normalize answer text for exact-match style evaluation."""
    text = text.lower()
    text = remove_punctuation(text)
    text = remove_articles(text)
    text = fix_whitespace(text)
    return text


def remove_punctuation(text: str) -> str:
    """Remove punctuation characters from text."""
    return "".join(char for char in text if char not in string.punctuation)


def remove_articles(text: str) -> str:
    """Remove English articles from text."""
    return re.sub(r"\b(a|an|the)\b", " ", text)


def fix_whitespace(text: str) -> str:
    """Collapse repeated whitespace."""
    return " ".join(text.split())


def exact_match(prediction: str, answers: list[str]) -> float:
    """Return 1.0 if prediction matches any normalized answer."""
    normalized_prediction = normalize_answer(prediction)

    for answer in answers:
        if normalized_prediction == normalize_answer(answer):
            return 1.0

    return 0.0


def mean_score(scores: list[float]) -> float:
    """Return the mean score, or 0.0 for an empty list."""
    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def routing_accuracy(predicted_tasks: list[str], target_tasks: list[str]) -> float:
    """Compute task routing accuracy."""
    if len(predicted_tasks) != len(target_tasks):
        raise ValueError("predicted_tasks and target_tasks must have the same length.")

    if not predicted_tasks:
        return 0.0

    correct = 0
    for predicted, target in zip(predicted_tasks, target_tasks):
        if predicted == target:
            correct += 1

    return correct / len(predicted_tasks)