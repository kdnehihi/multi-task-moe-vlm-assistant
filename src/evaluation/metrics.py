"""Evaluation metrics for vision-language QA."""

import re
import string

NO_ANSWER_NORMALIZED_LABELS = {
    "unanswerable",
    "no answer",
    "not answerable",
    "answer not available",
    "not question",
    "cannot be determined",
    "cannot determine",
    "n/a",
    "na",
    "answering does not require reading text in image",
}


def normalize_answer(text: str) -> str:
    """Normalize answer text for exact-match style evaluation."""
    text = text.lower()
    text = remove_punctuation(text)
    text = remove_articles(text)
    text = fix_whitespace(text)
    return text


def normalize_text(text: str) -> str:
    """Return the normalized string used for diagnostic outputs."""
    return normalize_answer(text)


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
    """Return token/span-aware containment without substring matches."""
    return strict_containment(prediction, answers)


def loose_containment(prediction: str, answers: list[str]) -> float:
    """Return legacy substring containment for regression diagnostics."""
    normalized_prediction = normalize_answer(prediction)
    if not normalized_prediction:
        return 0.0

    for answer in answers:
        normalized_answer = normalize_answer(answer)
        if normalized_answer and normalized_answer in normalized_prediction:
            return 1.0
    return 0.0


def strict_containment(prediction: str, answers: list[str]) -> float:
    """Return 1.0 if an answer appears as a full token or phrase."""
    prediction_tokens = containment_tokens(prediction)
    if not prediction_tokens:
        return 0.0

    for answer in answers:
        answer_tokens = containment_tokens(answer)
        if not answer_tokens:
            continue
        if contains_token_span(prediction_tokens, answer_tokens):
            return 1.0
    return 0.0


def containment_tokens(text: str) -> list[str]:
    """Tokenize for containment while preserving percent-attached numbers."""
    return re.findall(r"[a-z]+|\d+(?:[,.]\d+)*%?", str(text).lower())


def contains_token_span(tokens: list[str], span: list[str]) -> bool:
    """Return whether span appears as a contiguous token sequence."""
    if len(span) > len(tokens):
        return False
    for start in range(len(tokens) - len(span) + 1):
        if tokens[start:start + len(span)] == span:
            return True
    return False


def extract_numbers(text: str) -> list[float]:
    """Extract numeric values, handling commas and percent signs."""
    values = []
    for match in re.finditer(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?", str(text)):
        value = match.group(0).replace(",", "").rstrip("%")
        try:
            values.append(float(value))
        except ValueError:
            continue
    return values


def relaxed_numeric_accuracy(
    prediction: str,
    answers: list[str],
    tolerance: float = 0.05,
) -> float:
    """Return 1.0 if prediction/reference numeric values match within tolerance."""
    prediction_values = extract_numbers(prediction)
    if not prediction_values:
        return 0.0

    for answer in answers:
        reference_values = extract_numbers(answer)
        for predicted_value in prediction_values:
            for reference_value in reference_values:
                if reference_value == 0:
                    if abs(predicted_value - reference_value) <= tolerance:
                        return 1.0
                elif abs(predicted_value - reference_value) / abs(reference_value) <= tolerance:
                    return 1.0
    return 0.0


def chart_hybrid_accuracy(prediction: str, answers: list[str]) -> float:
    """Return ChartQA accuracy for numeric, yes/no, and label answers."""
    prediction_has_number = bool(extract_numbers(prediction))
    numeric_answers = [answer for answer in answers if extract_numbers(answer)]

    if prediction_has_number and numeric_answers:
        return relaxed_numeric_accuracy(prediction, numeric_answers)

    if normalized_exact_match(prediction, answers):
        return 1.0

    return strict_containment(prediction, answers)


def levenshtein_distance(left: str, right: str) -> int:
    """Compute Levenshtein edit distance."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def anls(prediction: str, answers: list[str]) -> float:
    """Return best ANLS-like normalized edit similarity."""
    normalized_prediction = normalize_answer(prediction)
    if not normalized_prediction:
        return 0.0

    best_score = 0.0
    for answer in answers:
        normalized_answer = normalize_answer(answer)
        if not normalized_answer:
            continue
        distance = levenshtein_distance(normalized_prediction, normalized_answer)
        similarity = 1.0 - distance / max(len(normalized_prediction), len(normalized_answer))
        if similarity >= 0.5:
            best_score = max(best_score, similarity)
    return best_score


def vqa_soft_score(prediction: str, answers: list[str]) -> float:
    """Return TextVQA/VQA-style soft accuracy from repeated references."""
    normalized_prediction = normalize_vqa_label(prediction)
    if not normalized_prediction:
        return 0.0
    match_count = sum(
        1
        for answer in answers
        if normalize_vqa_label(answer) == normalized_prediction
    )
    return min(1.0, match_count / 3.0)


def normalize_vqa_label(text: str) -> str:
    """Normalize labels for VQA-style soft scoring."""
    normalized = normalize_answer(text)
    if normalized in NO_ANSWER_NORMALIZED_LABELS:
        return "unanswerable"
    return normalized


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
