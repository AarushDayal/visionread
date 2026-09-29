import csv, os, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.csv")


def load_rows():
    if not os.path.exists(RESULTS_PATH):
        print(f"No results yet at {RESULTS_PATH} — run evaluate.py first.")
        sys.exit(1)
    with open(RESULTS_PATH) as f:
        return list(csv.DictReader(f))


def chart_accuracy_vs_severity(rows):
    # accuracy per (degradation_type, severity_level), averaged across all images tested at that level
    buckets = defaultdict(lambda: [0, 0])  # (dtype, level) -> [correct_count, total_count]
    for r in rows:
        key = (r["degradation_type"], int(r["severity_level"]))
        buckets[key][1] += 1
        if r["correct"] == "True":
            buckets[key][0] += 1

    dtypes = sorted(set(k[0] for k in buckets))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"blur": "#4a90d9", "noise": "#d97a4a", "rotation": "#4ad98f"}

    for dtype in dtypes:
        levels = sorted(set(k[1] for k in buckets if k[0] == dtype))
        acc = [buckets[(dtype, lvl)][0] / buckets[(dtype, lvl)][1] for lvl in levels]
        ax.plot(levels, acc, marker="o", label=dtype, color=colors.get(dtype, "#888"))

    ax.set_xlabel("Severity level (0 = clean)")
    ax.set_ylabel("Accuracy (exact match after normalization)")
    ax.set_title("Accuracy vs. Degradation Severity")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("chart_accuracy_vs_severity.png", dpi=180)
    plt.close()
    print("Saved chart_accuracy_vs_severity.png")


def chart_confidence_vs_correctness(rows):
    # of the runs the system marked "high" confidence, how many were actually correct?
    # of the runs marked "low" confidence, how many were actually correct?
    buckets = {"high": [0, 0], "low": [0, 0]}  # [correct, total]
    for r in rows:
        c = r["confidence"]
        if c not in buckets:
            continue
        buckets[c][1] += 1
        if r["correct"] == "True":
            buckets[c][0] += 1

    labels = ["high", "low"]
    accuracy = [buckets[l][0] / buckets[l][1] if buckets[l][1] else 0 for l in labels]
    counts = [buckets[l][1] for l in labels]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    bars = ax.bar(labels, accuracy, color=["#4caf7d", "#e0a83c"])
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"n={n}", ha="center", fontsize=9)
    ax.set_ylabel("Actual accuracy")
    ax.set_title("Does the Self-Consistency\nConfidence Signal Predict Correctness?")
    ax.set_ylim(0, 1.15)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("chart_confidence_vs_correctness.png", dpi=180)
    plt.close()
    print("Saved chart_confidence_vs_correctness.png")


if __name__ == "__main__":
    rows = load_rows()
    print(f"Loaded {len(rows)} logged runs from {RESULTS_PATH}\n")
    chart_accuracy_vs_severity(rows)
    chart_confidence_vs_correctness(rows)
