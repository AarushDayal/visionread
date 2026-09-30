import csv
import io
import os
import time
from datetime import datetime

import streamlit as st
from PIL import Image

st.set_page_config(page_title="VisionRead", layout="wide")

try:
    from ocr_core import extract_with_confidence, extract_with_voting, preprocess, PROVIDER, MODEL_IDS
except Exception as e:
    st.error(f"{e}\n\nSet one before running: export GEMINI_API_KEY=... or export OPENAI_API_KEY=...")
    st.stop()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(BASE_DIR, "eval")
CHARTS = ["chart_accuracy_vs_severity.png", "chart_confidence_vs_correctness.png",
          "chart_cer_vs_severity.png", "chart_error_taxonomy.png",
          "chart_accuracy_vs_rotation.png"]
HISTORY_FIELDS = ["time", "file", "text", "normalized", "confidence",
                  "method", "deskew", "time_s", "detail"]

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Settings")
    st.caption(f"Provider: {PROVIDER} · {', '.join(MODEL_IDS)}")
    auto_deskew = st.toggle("Auto-deskew", value=True)
    confidence_method = st.radio("Confidence method", ["cross-check", "voting"])

tab_extract, tab_history, tab_eval = st.tabs(["Extract", "History", "Evaluation"])

with tab_extract:
    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.info("Upload a plate, sign, or plaque image to begin.")
    else:
        img = Image.open(uploaded).convert("RGB")
        processed = preprocess(img, deskew=auto_deskew)

        if st.button("Extract", type="primary"):
            t0 = time.time()
            with st.spinner("Running inference..."):
                try:
                    if confidence_method == "voting":
                        res = extract_with_voting(processed)
                        detail = " | ".join(res["votes"])
                    else:
                        res = extract_with_confidence(processed, use_preprocess=False)
                        detail = res["cross_check"]
                    elapsed = round(time.time() - t0, 2)
                except Exception as e:
                    st.error(str(e))
                else:
                    rec = {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "file": uploaded.name,
                        "text": res["text"],
                        "normalized": res["normalized"],
                        "confidence": res["confidence"],
                        "method": confidence_method,
                        "deskew": auto_deskew,
                        "time_s": elapsed,
                        "detail": detail,
                    }
                    st.session_state.history.insert(0, rec)
                    st.session_state.last = {"id": time.time_ns(),
                                             "score": res.get("confidence_score"),
                                             **rec}

        left, right = st.columns([1, 1], gap="large")
        with left:
            c1, c2 = st.columns(2)
            c1.markdown("**Original**")
            c1.image(img, width="stretch")
            c2.markdown("**Preprocessed**" + (" (deskewed)" if auto_deskew else ""))
            c2.image(processed, width="stretch")
        with right:
            if "last" not in st.session_state:
                st.caption("Press Extract to run inference.")
            else:
                last = st.session_state.last
                st.code(last["text"], language=None)
                m1, m2 = st.columns(2)
                m1.metric("Confidence", last["confidence"])
                m2.metric("Time", f"{last['time_s']}s")
                with st.expander("Details"):
                    st.write("Normalized:", last["normalized"])
                    if last["method"] == "cross-check":
                        st.write("Cross-check:", last["detail"] or "(failed)")
                    else:
                        st.write("Votes:", last["detail"])
                        if last.get("score") is not None:
                            st.write("Confidence score:", last["score"])
                if last["confidence"] == "low":
                    st.warning("Needs review")
                    st.text_input("Reviewed text", value=last["text"],
                                  key=f"review_{last['id']}")

with tab_history:
    history = st.session_state.history
    if not history:
        st.info("No runs yet — extract something first.")
    else:
        st.dataframe(history, width="stretch", hide_index=True)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)
        st.download_button("Download CSV", buf.getvalue(),
                           file_name="visionread_history.csv", mime="text/csv")

with tab_eval:
    results_path = os.path.join(EVAL_DIR, "results.csv")
    if not os.path.exists(results_path):
        st.info("No evaluation results yet — run eval/evaluate.py to generate "
                "results.csv, then eval/analyze.py for charts.")
    else:
        with open(results_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            st.info("results.csv is empty — run eval/evaluate.py.")
        else:
            acc = sum(r["correct"] == "True" for r in rows) / len(rows)
            m1, m2, m3 = st.columns(3)
            m1.metric("Overall accuracy", f"{acc:.1%}")
            if "cer" in rows[0]:
                m2.metric("Mean CER",
                          f"{sum(float(r['cer']) for r in rows) / len(rows):.3f}")
            else:
                m2.metric("Mean CER", "—")
            m3.metric("Runs", len(rows))

            shown_chart = False
            for name in CHARTS:
                for folder in (EVAL_DIR, BASE_DIR):
                    path = os.path.join(folder, name)
                    if os.path.exists(path):
                        st.image(path, caption=name)
                        shown_chart = True
                        break
            if not shown_chart:
                st.caption("No charts found — run eval/analyze.py.")

            with st.expander("Raw results"):
                st.dataframe(rows, width="stretch", hide_index=True)
