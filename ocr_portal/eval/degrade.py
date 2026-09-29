import numpy as np
from PIL import Image, ImageFilter

# Severity ladders — index 0 is always "clean" (no degradation applied)
BLUR_RADII = [0, 1, 2, 3, 4]          # gaussian blur radius, px
NOISE_SIGMAS = [0, 10, 20, 30, 40]    # gaussian noise std-dev, 0-255 scale
ROTATION_ANGLES = [0, 5, 10, 20, 30]  # degrees, off-axis simulation


def blur(img, radius):
    if radius == 0:
        return img.copy()
    return img.filter(ImageFilter.GaussianBlur(radius))


def add_noise(img, sigma):
    if sigma == 0:
        return img.copy()
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def rotate(img, angle):
    if angle == 0:
        return img.copy()
    # fill with mid-grey so rotation doesn't introduce pure-black corners
    return img.convert("RGB").rotate(angle, expand=True, fillcolor=(128, 128, 128))


def generate_variants(image_path):
    """Yields (degradation_type, severity_level, severity_value, PIL.Image) for
    every rung of every ladder, based on one clean source image."""
    base = Image.open(image_path).convert("RGB")

    for level, radius in enumerate(BLUR_RADII):
        yield ("blur", level, radius, blur(base, radius))

    for level, sigma in enumerate(NOISE_SIGMAS):
        yield ("noise", level, sigma, add_noise(base, sigma))

    for level, angle in enumerate(ROTATION_ANGLES):
        yield ("rotation", level, angle, rotate(base, angle))


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(description="Preview degraded variants of one image to disk.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="degraded_previews")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for dtype, level, value, img in generate_variants(args.image):
        img.save(f"{args.out}/{dtype}_L{level}_{value}.png")
    print(f"Saved variants to {args.out}/")
