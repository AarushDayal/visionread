"""
Generates one clean, synthetic reference image per required category.
Ground truth is known exactly because we drew the text ourselves — this is
what evaluate.py will degrade and test against.

Run: python make_references.py
Then for each line it prints, run the matching evaluate.py command.
"""
import os
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = os.path.join(os.path.dirname(__file__), "references")
os.makedirs(OUT_DIR, exist_ok=True)


def font(size):
    # Falls back to PIL's default bitmap font if no TTF is found — still legible.
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def plate(text="7ABC123"):
    img = Image.new("RGB", (400, 200), (235, 240, 245))
    d = ImageDraw.Draw(img)
    d.rectangle([12, 12, 388, 188], outline=(20, 20, 20), width=6)
    d.rectangle([12, 12, 388, 188], outline=(200, 200, 200), width=2)
    d.text((110, 40), "CALIFORNIA", fill=(180, 30, 30), font=font(22))
    d.text((60, 95), text, fill=(20, 30, 90), font=font(48))
    img.save(f"{OUT_DIR}/plate.png")
    return text


def stop_sign(text="STOP"):
    img = Image.new("RGB", (300, 300), (235, 238, 235))
    d = ImageDraw.Draw(img)
    cx, cy, r = 150, 150, 110
    pts = []
    import math
    for i in range(8):
        ang = math.pi/8 + i * math.pi/4
        pts.append((cx + r*math.cos(ang), cy + r*math.sin(ang)))
    d.polygon(pts, fill=(178, 24, 24), outline=(245, 245, 245), width=4)
    d.text((cx - 62, cy - 24), text, fill=(255, 255, 255), font=font(50))
    img.save(f"{OUT_DIR}/stop_sign.png")
    return text


def speed_limit(number="65"):
    img = Image.new("RGB", (260, 340), (240, 240, 240))
    d = ImageDraw.Draw(img)
    d.rectangle([10, 10, 250, 330], outline=(20, 20, 20), width=5)
    d.rectangle([18, 18, 242, 322], outline=(20, 20, 20), width=2)
    d.text((45, 35), "SPEED", fill=(15, 15, 15), font=font(34))
    d.text((45, 80), "LIMIT", fill=(15, 15, 15), font=font(34))
    d.text((70, 160), number, fill=(15, 15, 15), font=font(90))
    img.save(f"{OUT_DIR}/speed_limit.png")
    return f"SPEED LIMIT {number}"


def advisory_plaque(number="35"):
    img = Image.new("RGB", (220, 220), (240, 240, 240))
    d = ImageDraw.Draw(img)
    d.rectangle([15, 15, 205, 205], outline=(20, 20, 20), width=5)
    d.rectangle([25, 25, 195, 195], fill=(245, 190, 40), outline=(20, 20, 20), width=3)
    d.text((65, 65), number, fill=(15, 15, 15), font=font(80))
    img.save(f"{OUT_DIR}/advisory_plaque.png")
    return number


def warning_sign(lines=("ROAD", "WORK", "AHEAD")):
    img = Image.new("RGB", (320, 320), (240, 240, 240))
    d = ImageDraw.Draw(img)
    cx, cy, half = 160, 160, 140
    d.polygon([(cx, cy-half), (cx+half, cy), (cx, cy+half), (cx-half, cy)],
               fill=(235, 140, 30), outline=(15, 15, 15), width=5)
    y = 105
    for line in lines:
        d.text((cx - 45, y), line, fill=(15, 15, 15), font=font(34))
        y += 42
    img.save(f"{OUT_DIR}/warning_sign.png")
    return " ".join(lines)


if __name__ == "__main__":
    refs = [
        ("plate.png", plate(), "plate"),
        ("stop_sign.png", stop_sign(), "stop_sign"),
        ("speed_limit.png", speed_limit(), "speed_limit"),
        ("advisory_plaque.png", advisory_plaque(), "advisory"),
        ("warning_sign.png", warning_sign(), "warning_sign"),
    ]
    print(f"\nGenerated reference images in {OUT_DIR}/\n")
    print("Run these to build up results.csv:\n")
    for fname, gt, cat in refs:
        label = fname.replace(".png", "")
        print(f'python evaluate.py --image references/{fname} --ground-truth "{gt}" '
              f'--label {label} --category {cat}')
