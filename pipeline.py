import torch, json, argparse
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = (
    "Qwen/Qwen2.5-VL-3B-Instruct"  # 3B for local GPU; use 7B on cloud if VRAM allows
)

# Load ONCE — this is the part the doc's methodology section emphasizes (Section 2.3, step 3)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

PROMPT = """Read the text in this image. Return ONLY the characters present, in reading order — no labels, no quotes, no explanation.

Rules:
- License plate: return the plate number only. Drop any state/region banner or slogan printed around it (e.g. "CALIFORNIA"). EXCEPTION: if a leading character or letter is printed as part of the plate itself (e.g. a province character on a non-US plate), keep it.
- Multi-line sign: read top to bottom, join lines with a single space.
- Numeric-only plaque: digits only, no unit.
- Stop sign: return exactly what's printed."""


def extract_text(image_path):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=64)
    trimmed = output_ids[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-image", required=True)
    args = parser.parse_args()
    result = {"text": extract_text(args.input_image), "confidence": None}
    print(json.dumps(result))
