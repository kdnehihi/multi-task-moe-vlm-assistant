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
    return normalized_exact_match(prediction, answers)


def raw_exact_match(prediction: str, answers: list[str]) -> float:
    """Return 1.0 if prediction exactly matches any raw answer after trimming."""
    clean_prediction = str(prediction).strip()
    for answer in answers:
        if clean_prediction == str(answer).strip():
            return 1.0
    return 0.0


def normalized_exact_match(prediction: str, answers: list[str]) -> float:
    """Return 1.0 if prediction matches any normalized answer."""
    normalized_prediction = normalize_answer(prediction)

    for answer in answers:
        if normalized_prediction == normalize_answer(answer):
            return 1.0

    return 0.0


def token_f1(prediction: str, answers: list[str]) -> float:
    """Return the best token F1 against normalized reference answers."""
    prediction_tokens = normalize_answer(prediction).split()
    if not prediction_tokens:
        return 0.0

    best_score = 0.0
    for answer in answers:
        answer_tokens = normalize_answer(answer).split()
        if not answer_tokens:
            continue
        overlap = 0
        remaining = answer_tokens.copy()
        for token in prediction_tokens:
            if token in remaining:
                overlap += 1
                remaining.remove(token)
        if overlap == 0:
            continue
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(answer_tokens)
        best_score = max(best_score, 2 * precision * recall / (precision + recall))
    return best_score


def containment(prediction: str, answers: list[str]) -> float:
    """Return 1.0 if a normalized answer is contained in the prediction."""
    normalized_prediction = normalize_answer(prediction)
    if not normalized_prediction:
        return 0.0

    for answer in answers:
        normalized_answer = normalize_answer(answer)
        if normalized_answer and normalized_answer in normalized_prediction:
            return 1.0
    return 0.0


def output_token_length(prediction: str) -> int:
    """Count whitespace tokens in a generated answer."""
    return len(str(prediction).split())


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
