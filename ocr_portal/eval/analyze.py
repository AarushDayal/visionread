import csv, os, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.csv")
COLORS = {"blur": "#4a90d9", "noise": "#d97a4a", "rotation": "#4ad98f"}
ERROR_TYPES = ["none", "confusable", "extra_text", "missing_chars", "other"]
ERROR_COLORS = {"none": "#4caf7d", "confusable": "#e0a83c", "extra_text": "#d95f4a",
                "missing_chars": "#4a90d9", "other": "#8b98a5"}


def load_rows():
    if not os.path.exists(RESULTS_PATH):
        print(f"No results yet at {RESULTS_PATH} — run evaluate.py first.")
        sys.exit(1)
    with open(RESULTS_PATH) as f:
        return list(csv.DictReader(f))


def _mode(r):  # back-compat with CSVs written before the --mode flag
    return r.get("mode") or "baseline"


def _method(r):  # back-compat with CSVs written before --confidence-method
    return r.get("confidence_method") or "crosscheck"


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

    for dtype in dtypes:
        levels = sorted(set(k[1] for k in buckets if k[0] == dtype))
        acc = [buckets[(dtype, lvl)][0] / buckets[(dtype, lvl)][1] for lvl in levels]
        ax.plot(levels, acc, marker="o", label=dtype, color=COLORS.get(dtype, "#888"))

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
    # of the runs each method marked "high" confidence, how many were actually
    # correct? — crosscheck and voting side by side.
    methods = sorted({_method(r) for r in rows})
    palette = {"high": "#4caf7d", "low": "#e0a83c"}
    labels, accuracy, counts, colors = [], [], [], []
    for m in methods:
        for lvl in ("high", "low"):
            sel = [r for r in rows if _method(r) == m and r["confidence"] == lvl]
            acc = sum(r["correct"] == "True" for r in sel) / len(sel) if sel else 0
            labels.append(f"{m}\n{lvl}")
            accuracy.append(acc)
            counts.append(len(sel))
            colors.append(palette.get(lvl, "#888"))

    fig, ax = plt.subplots(figsize=(max(5, 1.7 * len(labels)), 4.5))
    bars = ax.bar(labels, accuracy, color=colors)
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"n={n}", ha="center", fontsize=9)
    ax.set_ylabel("Actual accuracy")
    ax.set_title("Does the Confidence Signal Predict Correctness?\nby confidence method")
    ax.set_ylim(0, 1.15)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("chart_confidence_vs_correctness.png", dpi=180)
    plt.close()
    print("Saved chart_confidence_vs_correctness.png")


def chart_cer_vs_severity(rows):
    # mean CER per (degradation_type, severity_level)
    buckets = defaultdict(lambda: [0.0, 0])  # [cer_sum, count]
    for r in rows:
        key = (r["degradation_type"], int(r["severity_level"]))
        buckets[key][0] += float(r["cer"])
        buckets[key][1] += 1

    dtypes = sorted(set(k[0] for k in buckets))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for dtype in dtypes:
        levels = sorted(set(k[1] for k in buckets if k[0] == dtype))
        cer = [buckets[(dtype, lvl)][0] / buckets[(dtype, lvl)][1] for lvl in levels]
        ax.plot(levels, cer, marker="o", label=dtype, color=COLORS.get(dtype, "#888"))

    ax.set_xlabel("Severity level (0 = clean)")
    ax.set_ylabel("Mean CER (0 = perfect)")
    ax.set_title("Character Error Rate vs. Degradation Severity")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("chart_cer_vs_severity.png", dpi=180)
    plt.close()
    print("Saved chart_cer_vs_severity.png")


def chart_error_taxonomy(rows):
    # stacked bars: error_type counts per degradation type
    buckets = defaultdict(lambda: defaultdict(int))  # dtype -> error_type -> count
    for r in rows:
        buckets[r["degradation_type"]][r["error_type"]] += 1

    dtypes = sorted(buckets)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bottoms = [0] * len(dtypes)
    for etype in ERROR_TYPES:
        counts = [buckets[d].get(etype, 0) for d in dtypes]
        ax.bar(dtypes, counts, bottom=bottoms, label=etype,
               color=ERROR_COLORS.get(etype, "#888"))
        bottoms = [b + c for b, c in zip(bottoms, counts)]

    ax.set_xlabel("Degradation type")
    ax.set_ylabel("Run count")
    ax.set_title("Error Taxonomy by Degradation Type")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("chart_error_taxonomy.png", dpi=180)
    plt.close()
    print("Saved chart_error_taxonomy.png")


def chart_accuracy_vs_rotation(rows):
    # rotation degradation only: one line per pipeline mode (baseline vs deskew)
    buckets = defaultdict(lambda: [0, 0])  # (mode, level) -> [correct, total]
    for r in rows:
        if r["degradation_type"] != "rotation":
            continue
        key = (_mode(r), int(r["severity_level"]))
        buckets[key][1] += 1
        if r["correct"] == "True":
            buckets[key][0] += 1
    if not buckets:
        print("No rotation rows found — skipping chart_accuracy_vs_rotation")
        return

    mode_colors = {"baseline": "#d97a4a", "deskew": "#4a90d9"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode in sorted({k[0] for k in buckets}):
        levels = sorted(set(k[1] for k in buckets if k[0] == mode))
        acc = [buckets[(mode, lvl)][0] / buckets[(mode, lvl)][1] for lvl in levels]
        ax.plot(levels, acc, marker="o", label=mode,
                color=mode_colors.get(mode, "#888"))

    ax.set_xlabel("Rotation severity level (0 = clean)")
    ax.set_ylabel("Accuracy (exact match after normalization)")
    ax.set_title("Accuracy vs. Rotation Severity: Baseline vs Deskew")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("chart_accuracy_vs_rotation.png", dpi=180)
    plt.close()
    print("Saved chart_accuracy_vs_rotation.png")


if __name__ == "__main__":
    rows = load_rows()
    print(f"Loaded {len(rows)} logged runs from {RESULTS_PATH}\n")
    chart_accuracy_vs_severity(rows)
    chart_confidence_vs_correctness(rows)
    chart_cer_vs_severity(rows)
    chart_error_taxonomy(rows)
    chart_accuracy_vs_rotation(rows)
