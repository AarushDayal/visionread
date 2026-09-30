from rapidfuzz.distance import Levenshtein

# Canonical map: confusable pairs collapse to one form (letter -> digit).
_CANON = {"O": "0", "I": "1", "S": "5", "B": "8", "Z": "2"}


def _canon(s):
    return "".join(_CANON.get(c, c) for c in s.upper())


def _is_subsequence(pred, gt):
    it = iter(gt)
    return all(c in it for c in pred)


def compute_cer(pred, gt):
    if not gt:
        return 0.0 if not pred else 1.0
    return Levenshtein.distance(pred, gt) / len(gt)


def classify_error(pred, gt):
    """Pure function: classify prediction vs ground truth.

    none | confusable | extra_text | missing_chars | other
    """
    if pred == gt:
        return "none"
    if _canon(pred) == _canon(gt):
        return "confusable"
    if gt and gt in pred:
        return "extra_text"
    if _is_subsequence(pred, gt):
        return "missing_chars"
    return "other"
