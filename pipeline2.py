import google.generativeai as genai
import json, argparse, os, re, time
from PIL import Image
from google.api_core.exceptions import (
    ResourceExhausted,
    ServiceUnavailable,
    InternalServerError,
    DeadlineExceeded,
    NotFound,
)

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")
genai.configure(api_key=API_KEY)

_model_env = os.environ.get("GEMINI_MODELS") or os.environ.get("GEMINI_MODEL")
MODEL_IDS = (
    [m.strip() for m in _model_env.split(",") if m.strip()]
    if _model_env
    else ["gemini-3-flash-preview", "gemini-3.8-flash", "gemini-flash-latest"]
)

PROMPT = """Read the text in this image. Return ONLY the characters present, in reading order — no labels, no explanation.
Rules: drop state/region banners around a license plate; keep script-specific prefixes that are part of the plate itself;
join multi-line signs top-to-bottom with a single space; numeric plaques get digits only."""


def extract_text(image_path, max_retries=4):
    img = Image.open(image_path)
    last_err = RuntimeError("no models configured")
    for model_id in MODEL_IDS:
        model = genai.GenerativeModel(model_id, generation_config={"temperature": 0})
        delay = 4
        for attempt in range(max_retries):
            try:
                response = model.generate_content([PROMPT, img])
                text = (response.text or "").strip()
                if not text:
                    last_err = RuntimeError(f"empty response from {model_id}")
                    break
                return text
            except NotFound as e:
                last_err = e
                break
            except (
                ResourceExhausted,
                ServiceUnavailable,
                InternalServerError,
                DeadlineExceeded,
            ) as e:
                last_err = e
                if attempt == max_retries - 1:
                    break
                try:
                    m = re.search(r"retry in ([\d.]+)", str(e), re.IGNORECASE)
                    wait = max(delay, float(m.group(1))) if m else delay
                except (ValueError, TypeError):
                    wait = delay
                time.sleep(wait)
                delay = min(delay * 2, 45)
    raise RuntimeError(
        f"All models failed ({', '.join(MODEL_IDS)}). Last error: {last_err}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-image", required=True)
    args = parser.parse_args()
    print(json.dumps({"text": extract_text(args.input_image), "confidence": None}))
