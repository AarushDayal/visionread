import os, re, time
from PIL import Image, ImageEnhance, ImageOps
import google.generativeai as genai
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

# Primary first: "gemini-3-flash-preview" is verified working with its own
# quota bucket, while "gemini-3.8-flash" (the current default alias target)
# was observed returning 429 quota-exhausted (limit 20). Override with:
#   GEMINI_MODEL="my-model"  or  GEMINI_MODELS="model-a,model-b"
_model_env = os.environ.get("GEMINI_MODELS") or os.environ.get("GEMINI_MODEL")
MODEL_IDS = (
    [m.strip() for m in _model_env.split(",") if m.strip()]
    if _model_env
    else ["gemini-3-flash-preview", "gemini-3.8-flash", "gemini-flash-latest"]
)

# temperature=0 -> deterministic transcription, less hallucination.
_models = {}


def _get_model(model_id):
    if model_id not in _models:
        _models[model_id] = genai.GenerativeModel(
            model_id, generation_config={"temperature": 0}
        )
    return _models[model_id]


PROMPT_A = """Read the text in this image. Return ONLY the characters present, in reading order — no labels, no explanation, no quotes.

Rules:
- License plate: return the plate number only. Drop any state/region banner or slogan printed around it (e.g. "CALIFORNIA"). EXCEPTION: if a leading character/letter is printed as part of the plate itself (e.g. a province character), keep it.
- Multi-line sign: read top to bottom, join lines with a single space.
- Numeric-only plaque: digits only, no unit.
- Stop sign: return exactly what's printed."""

PROMPT_B = """Transcribe exactly what is printed in this image, character by character, in the order a person would read it. Output nothing but the transcription itself.

Constraints:
- For a vehicle plate, omit any jurisdiction name or slogan printed around the plate; keep any character that is part of the registration string itself (e.g. a leading province letter).
- Join multiple lines of text with one space.
- If the image shows only digits, output only those digits."""


def preprocess(img_or_path):
    img = Image.open(img_or_path) if isinstance(img_or_path, str) else img_or_path
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    max_dim = 1536
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.15)
    img = ImageEnhance.Sharpness(img).enhance(1.4)
    return img


def normalize(text):
    if text is None:
        return ""
    t = text.strip().upper()
    t = re.sub(r"[-.\u00b7_]", "", t)
    # Collapse whitespace to a single space (but keep it): "ROAD  WORK"
    # -> "ROAD WORK". Removing spaces entirely made "SPEED LIMIT 65"
    # score/display as "SPEEDLIMIT65" and hid missing-space errors.
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _server_retry_delay(exc, default):
    """Honor the server's 'retry in Xs' hint when present (it is often much
    longer than a fixed 2s backoff, e.g. ~38s on free-tier 429s)."""
    try:
        m = re.search(r"retry in ([\d.]+)", str(exc), re.IGNORECASE)
        if m:
            return max(default, float(m.group(1)))
    except (ValueError, TypeError):
        pass
    return default


def run_inference(img, prompt, max_retries=4):
    """Try each model in MODEL_IDS in order, with exponential backoff on
    rate-limit (429) or transient server errors.

    Falls through to the next model when the current one is exhausted
    (quota 429 after all retries), missing (404), or returns empty text —
    so one bad/quota-drained model can't take the whole app down.
    """
    last_err = RuntimeError("no models configured")
    for model_id in MODEL_IDS:
        model = _get_model(model_id)
        delay = 4
        for attempt in range(max_retries):
            try:
                response = model.generate_content([prompt, img])
                text = (response.text or "").strip()
                if not text:
                    last_err = RuntimeError(f"empty response from {model_id}")
                    break  # retrying won't help; try next model
                return text
            except NotFound as e:
                last_err = e
                break  # model doesn't exist here; try next model now
            except (
                ResourceExhausted,
                ServiceUnavailable,
                InternalServerError,
                DeadlineExceeded,
            ) as e:
                last_err = e
                if attempt == max_retries - 1:
                    break  # out of retries for this model; try next model
                wait = _server_retry_delay(e, delay)
                print(f"    ({model_id}: {type(e).__name__}, retrying in {wait:g}s...)")
                time.sleep(wait)
                delay = min(delay * 2, 45)
    raise RuntimeError(
        f"All models failed ({', '.join(MODEL_IDS)}). Last error: {last_err}"
    )


def extract_with_confidence(img_or_path):
    """img_or_path: file path OR an already-loaded PIL Image (skips preprocessing
    if you've already degraded/prepared it yourself, e.g. in evaluate.py).

    The cross-check prompt (B) is only a confidence signal: if it fails
    (e.g. quota) after A succeeded, we still return A's result with
    confidence "low" instead of failing the whole request.
    """
    img = preprocess(img_or_path) if isinstance(img_or_path, str) else img_or_path

    raw_a = run_inference(img, PROMPT_A)
    try:
        raw_b = run_inference(img, PROMPT_B)
    except Exception as e:
        print(f"    (cross-check skipped: {e})")
        raw_b = ""
    norm_a, norm_b = normalize(raw_a), normalize(raw_b)
    agree = bool(raw_b) and (norm_a == norm_b) and norm_a != ""

    return {
        "text": raw_a.strip(),
        "normalized": norm_a,
        "confidence": "high" if agree else "low",
        "cross_check": raw_b.strip(),
        "agreement": agree,
    }
