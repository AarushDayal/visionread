import argparse, csv, os, sys, time

CALL_DELAY_S = 3  # spacing between API calls — stay comfortably under free-tier RPM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ocr_core import extract_with_confidence, extract_with_voting, normalize, preprocess
from degrade import generate_variants
from metrics import classify_error, compute_cer

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.csv")
FIELDS = ["image_label", "category", "degradation_type", "severity_level",
          "severity_value", "ground_truth_norm", "predicted_norm", "correct",
          "confidence", "agreement", "time_s", "cer", "error_type",
          "mode", "confidence_method"]


def ensure_csv_header():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            header = f.readline().strip()
        if header != ",".join(FIELDS):
            sys.exit(f"{RESULTS_PATH} has an old header — delete it and rerun.")
        return
    with open(RESULTS_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()


def run(image_path, ground_truth, image_label, category, mode, confidence_method):
    ensure_csv_header()
    gt_norm = normalize(ground_truth)
    rows = []
    # baseline: pass the degraded variant through untouched.
    # deskew: full pipeline (resize + deskew + contrast/sharp).
    preprocess_flags = dict(use_preprocess=(mode == "deskew"), deskew=(mode == "deskew"))

    for dtype, level, value, variant_img in generate_variants(image_path):
        start = time.time()
        try:
            if confidence_method == "voting":
                work = preprocess(variant_img, deskew=True) if mode == "deskew" else variant_img
                result = extract_with_voting(work)
            else:
                result = extract_with_confidence(variant_img, **preprocess_flags)
        except Exception as e:
            print(f"  [{dtype} L{level}] ERROR: {e}")
            continue
        elapsed = round(time.time() - start, 2)
        pred_norm = result["normalized"]
        correct = pred_norm == gt_norm
        cer = round(compute_cer(pred_norm, gt_norm), 4)
        error_type = classify_error(pred_norm, gt_norm)

        row = {
            "image_label": image_label, "category": category,
            "degradation_type": dtype, "severity_level": level, "severity_value": value,
            "ground_truth_norm": gt_norm, "predicted_norm": pred_norm, "correct": correct,
            "confidence": result["confidence"], "agreement": result.get("agreement"),
            "time_s": elapsed,
            "cer": cer, "error_type": error_type,
            "mode": mode, "confidence_method": confidence_method,
        }
        rows.append(row)
        print(f"  [{dtype:8s} L{level}] pred={pred_norm!r:20s} correct={correct} conf={result['confidence']}")
        time.sleep(CALL_DELAY_S)

    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        for row in rows:
            writer.writerow(row)

    print(f"\nAppended {len(rows)} rows to {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="path to a CLEAN reference image")
    parser.add_argument("--ground-truth", required=True, help="correct text, e.g. 7ABC123")
    parser.add_argument("--label", required=True, help="short name for this image, e.g. plate1")
    parser.add_argument("--category", required=True,
                        help="plate | stop_sign | speed_limit | advisory | warning_sign")
    parser.add_argument("--mode", choices=["baseline", "deskew"], default="baseline",
                        help="baseline = no preprocess/no deskew; deskew = full pipeline")
    parser.add_argument("--confidence-method", choices=["crosscheck", "voting"],
                        default="crosscheck",
                        help="crosscheck = two-prompt agreement; voting = 3-view majority")
    args = parser.parse_args()
    run(args.image, args.ground_truth, args.label, args.category,
        args.mode, args.confidence_method)
