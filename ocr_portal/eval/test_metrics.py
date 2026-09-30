"""Run: python test_metrics.py (assert-based, no framework)."""
from metrics import classify_error, compute_cer

cases = [
    # (pred, gt, expected_error_type, expected_cer)
    ("7ABC123", "7ABC123", "none", 0.0),
    ("7A8C123", "7ABC123", "confusable", 1 / 7),
    ("7ABC123XX", "7ABC123", "extra_text", 2 / 7),
    ("7AC123", "7ABC123", "missing_chars", 1 / 7),
    ("XYZ9999", "7ABC123", "other", 1.0),
    ("", "7ABC123", "missing_chars", 1.0),
]

for pred, gt, want_type, want_cer in cases:
    got_type = classify_error(pred, gt)
    got_cer = compute_cer(pred, gt)
    assert got_type == want_type, f"{pred!r} vs {gt!r}: type {got_type} != {want_type}"
    assert abs(got_cer - want_cer) < 1e-9, f"{pred!r} vs {gt!r}: cer {got_cer} != {want_cer}"
    print(f"ok  {pred!r:12s} -> {got_type:15s} cer={got_cer:.3f}")

print("all 6 cases passed")
