import cv2
import numpy as np
from PIL import Image


def deskew(img: Image.Image) -> tuple[Image.Image, float]:
    """Return (image rotated upright, applied angle in degrees, CCW positive).

    Ink = inverted Otsu threshold capped at 100 so large mid-tone sign bodies
    (the orange warning diamond, gray ~156) can't fuse with the frame they
    enclose. Per-component min-area rect angles are wrapped into +/-45; the
    largest component (the frame — an exact measurement) is used when it agrees
    with the glyph median, else the median is used so one 45-deg-off structure
    can't dominate. Grey (128,128,128) fill created by degrade.rotate is masked
    out first. Correction clamped to +/-45 deg.
    """
    gray = np.array(img.convert("L"))
    g = gray.copy()
    g[(gray >= 118) & (gray <= 138)] = 255  # grey rotation fill -> background
    otsu, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, th = cv2.threshold(g, int(min(otsu, 100)), 255, cv2.THRESH_BINARY_INV)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(th)
    cands = []  # (area, correction angle) — biggest frame first
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 8:  # ponytail: sub-glyph specks are pure angle noise
            continue
        sub = labels[y:y + h, x:x + w] == i
        pts = np.column_stack(np.where(sub))[:, ::-1].astype(np.float32).reshape(-1, 1, 2)
        a = cv2.minAreaRect(pts)[-1]  # [-90, 0): tilt shows up as -tilt, or
        cands.append((area, a + 90 if a < -45 else a))  # -(90+tilt) when tilt < 0
    if not cands:
        return img, 0.0

    # Frames (borders, octagon) measure the tilt exactly; tiny glyph rects
    # jitter. Trust the largest component when it agrees with the glyph median,
    # otherwise the glyphs are outvoting a 45-deg-off structure (warning sign's
    # diamond border) and the median wins.
    cands.sort(reverse=True)
    big = cands[0][1]
    median = float(np.median([c for _, c in cands]))
    angle = big if abs(big - median) <= 15 else median
    angle = max(-45.0, min(45.0, angle))
    if abs(angle) < 0.1:
        return img, 0.0
    out = img.rotate(angle, resample=Image.BICUBIC, fillcolor=(128, 128, 128))
    return out, angle
