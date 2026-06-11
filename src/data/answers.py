"""Answer and task helpers for multi-task VQA data."""

from collections import Counter, defaultdict
import re
import string


TASK_ALIASES = {
    "docvqa": "docvqa",
    "document_qa": "docvqa",
    "chartqa": "chartqa",
    "chart_qa": "chartqa",
    "textvqa": "textvqa",
    "image_vqa": "textvqa",
}

NO_ANSWER_LABELS = {
    "unanswerable",
    "no answer",
    "not a question",
    "cannot be determined",
    "answering does not require reading text in the image",
}


def canonicalize_task_type(task_type: str | None, dataset: str | None = None) -> str:
    """Return the canonical task name used by training and routing."""
    raw_task = normalize_task_key(task_type)
    raw_dataset = normalize_task_key(dataset)
    return TASK_ALIASES.get(raw_task) or TASK_ALIASES.get(raw_dataset) or raw_task


def normalize_task_key(value: str | None) -> str:
    """Normalize task keys while preserving the answer-normalizer boundary."""
    return str(value or "").strip().lower()


def normalize_answer_group(text: str) -> str:
    """Group equivalent references for training-target voting."""
    text = str(text).lower().strip()
    text = "".join(char for char in text if char not in string.punctuation)
    return re.sub(r"\s+", " ", text).strip()


def is_no_answer_variant(text: str) -> bool:
    """Return whether a reference is a strong no-answer label."""
    return normalize_answer_group(text) in NO_ANSWER_LABELS


def choose_training_answer(answers: list[str], task_type: str) -> str:
    """Choose one safe supervised answer without lowercasing original text."""
    clean_answers = [str(answer).strip() for answer in answers if str(answer).strip()]
    if not clean_answers:
        return "unanswerable"

    no_answer_votes = sum(is_no_answer_variant(answer) for answer in clean_answers)
    if no_answer_votes > len(clean_answers) / 2:
        return "unanswerable"

    grouped_answers: dict[str, list[str]] = defaultdict(list)
    for answer in clean_answers:
        if is_no_answer_variant(answer):
            continue
        grouped_answers[normalize_answer_group(answer)].append(answer)

    if not grouped_answers:
        return "unanswerable"

    group_counts = Counter(
        group_key
        for group_key, originals in grouped_answers.items()
        for _ in originals
    )
    best_group, _ = group_counts.most_common(1)[0]
    candidates = grouped_answers[best_group]
    return min(candidates, key=lambda answer: (len(answer.split()), len(answer), answer))
