import base64, io, os, re, time
from collections import Counter
from PIL import Image, ImageEnhance, ImageOps
from deskew import deskew as apply_deskew

GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if not GEMINI_KEY and not OPENAI_KEY:
    raise RuntimeError(
        "Set GEMINI_API_KEY or OPENAI_API_KEY environment variable"
    )

# Provider is picked from whichever key is set; GEMINI_API_KEY wins if both are.
PROVIDER = "gemini" if GEMINI_KEY else "openai"

if PROVIDER == "gemini":
    import google.generativeai as genai
    from google.api_core.exceptions import (
        ResourceExhausted,
        ServiceUnavailable,
        InternalServerError,
        DeadlineExceeded,
        NotFound,
    )

    genai.configure(api_key=GEMINI_KEY)
    # Primary first: "gemini-3-flash-preview" was verified working while
    # "gemini-3.8-flash" hit 429 quota. Override with:
    #   GEMINI_MODEL="my-model"  or  GEMINI_MODELS="model-a,model-b"
    _model_env = os.environ.get("GEMINI_MODELS") or os.environ.get("GEMINI_MODEL")
    MODEL_IDS = (
        [m.strip() for m in _model_env.split(",") if m.strip()]
        if _model_env
        else ["gemini-3-flash-preview", "gemini-3.8-flash", "gemini-flash-latest"]
    )
    _SKIP_EXC = (NotFound,)
    _RETRY_EXC = (
        ResourceExhausted,
        ServiceUnavailable,
        InternalServerError,
        DeadlineExceeded,
    )
else:
    import openai
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_KEY)
    # Primary first: "gpt-4o" strong reader, "gpt-4o-mini" cheap fallback.
    # Override with:
    #   OPENAI_MODEL="my-model"  or  OPENAI_MODELS="model-a,model-b"
    _model_env = os.environ.get("OPENAI_MODELS") or os.environ.get("OPENAI_MODEL")
    MODEL_IDS = (
        [m.strip() for m in _model_env.split(",") if m.strip()]
        if _model_env
        else ["gpt-4o", "gpt-4o-mini"]
    )
    _SKIP_EXC = (openai.NotFoundError, openai.BadRequestError)
    _RETRY_EXC = (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )

# temperature=0 -> deterministic transcription, less hallucination.
print(f"ocr_core: provider={PROVIDER}, models={MODEL_IDS}")


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


def preprocess(img_or_path, deskew=True):
    img = Image.open(img_or_path) if isinstance(img_or_path, str) else img_or_path
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    max_dim = 1536
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    if deskew:
        img, _angle = apply_deskew(img)
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
    """Honor the server's 'retry in Xs' / 'try again in Xs' hint when present
    (it can be much longer than a fixed backoff)."""
    try:
        m = re.search(r"(?:retry|try again) in ([\d.]+)", str(exc), re.IGNORECASE)
        if m:
            return max(default, float(m.group(1)))
    except (ValueError, TypeError):
        pass
    return default


def _to_data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _generate(model_id, img, prompt):
    """One attempt on the active provider; returns stripped text (may be '')."""
    if PROVIDER == "gemini":
        model = genai.GenerativeModel(
            model_id, generation_config={"temperature": 0}
        )
        return (model.generate_content([prompt, img]).text or "").strip()
    response = client.chat.completions.create(
        model=model_id,
        temperature=0,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _to_data_url(img)}},
            ],
        }],
    )
    return (response.choices[0].message.content or "").strip()


def run_inference(img, prompt, max_retries=4):
    """Try each model in MODEL_IDS in order, with exponential backoff on
    rate-limit (429) or transient server errors.

    Falls through to the next model when the current one is exhausted
    (429 after all retries), missing/bad-request, or returns empty text —
    so one bad model can't take the whole app down.
    """
    last_err = RuntimeError("no models configured")
    for model_id in MODEL_IDS:
        delay = 4
        for attempt in range(max_retries):
            try:
                text = _generate(model_id, img, prompt)
                if not text:
                    last_err = RuntimeError(f"empty response from {model_id}")
                    break  # retrying won't help; try next model
                return text
            except _SKIP_EXC as e:
                last_err = e
                break  # model missing or rejects the request; try next model now
            except _RETRY_EXC as e:
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


def extract_with_confidence(img_or_path, use_preprocess=True, deskew=True):
    """Returns {text, normalized, confidence, cross_check, agreement}.

    use_preprocess=True runs preprocess() on paths AND PIL images (the current
    default). use_preprocess=False reproduces the old PIL-passthrough behaviour
    (evaluate.py's degraded variants); paths are then only opened. deskew is
    forwarded to preprocess and ignored when use_preprocess=False.

    The cross-check prompt (B) is only a confidence signal: if it fails
    (e.g. quota) after A succeeded, we still return A's result with
    confidence "low" instead of failing the whole request.
    """
    if use_preprocess:
        img = preprocess(img_or_path, deskew=deskew)
    elif isinstance(img_or_path, str):
        img = Image.open(img_or_path).convert("RGB")
    else:
        img = img_or_path

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


def _build_views(img, n_views):
    """The n_views variants: original, contrast x1.3, sharpened + rescaled 0.8x."""
    views = [img]
    if n_views > 1:
        views.append(ImageEnhance.Contrast(img).enhance(1.3))
    if n_views > 2:
        w, h = img.size
        sharpened = ImageEnhance.Sharpness(img).enhance(1.5)
        views.append(sharpened.resize((max(int(w * 0.8), 1), max(int(h * 0.8), 1)),
                                      Image.LANCZOS))
    return views[:n_views]


def extract_with_voting(img, n_views=3):
    """Self-consistency voting: PROMPT_A on each of n_views image variants,
    majority of normalized outputs wins.

    Returns {text, normalized, confidence_score, votes, confidence}. A view
    whose API call fails is dropped (casts no vote) but the denominator stays
    fixed at len(views), so confidence_score remains in {0.33, 0.67, 1.0}.
    """
    views = _build_views(img, n_views)
    votes = []
    for i, view in enumerate(views):
        try:
            raw = run_inference(view, PROMPT_A)
        except Exception as e:
            print(f"    (view {i} dropped: {e})")
            continue
        nv = normalize(raw)
        if nv:
            votes.append(nv)
    if not votes:
        raise RuntimeError("all views failed")
    winner, count = Counter(votes).most_common(1)[0]
    score = round(count / len(views), 2)
    return {
        "text": winner,
        "normalized": winner,
        "confidence_score": score,
        "votes": votes,
        "confidence": "high" if score == 1.0 else "low",
    }
