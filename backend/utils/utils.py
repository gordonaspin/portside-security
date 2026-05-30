import time
from collections import defaultdict
from datetime import datetime
from math import sqrt

import cv2
import numpy as np

def make_ts_string(epoch=time.time()):
    return datetime.fromtimestamp(epoch).strftime("%Y%m%d_%H%M%S")

def make_ts_string_precise(epoch=time.time()):
    return datetime.fromtimestamp(epoch).strftime("%Y%m%d_%H%M%S_%f")[:-3]

def make_readable_ts(epoch=time.time()):
    return datetime.fromtimestamp(epoch).strftime("%Y/%m/%d %H:%M:%S")

def make_readable_hms(epoch=time.time()):
    return datetime.fromtimestamp(epoch).strftime("%H:%M:%S")

def tags_to_str(tags: defaultdict[set]):
    if not tags:
        return ""

    parts = []
    for obj, colors in tags.items():
        object_str = obj
        color_str = "-".join(colors)
        parts.append(f"{object_str}-{color_str}")
    return "_".join(parts)

def boxes_overlap(a, b) -> bool:
    """Return True if two boxes (x1,y1,x2,y2) overlap."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

def yolo_box_to_roi(frame_bgr, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

    # Clamp to image bounds
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))

    roi = frame_bgr[y1:y2, x1:x2].copy()
    return roi

def detect_object_color(roi_bgr, is_night: bool):
    """
    Adaptive color classifier.
    Uses day classifier (LAB + kmeans) or night classifier (brightness-based)
    depending on lighting conditions.
    """

    if roi_bgr is None or roi_bgr.size == 0:
        return "unknown"

    if is_night:
        return detect_object_color_night(roi_bgr)
    else:
        return detect_object_color_day(roi_bgr)


def detect_object_color_day(roi_bgr, k=2):
    if roi_bgr is None or roi_bgr.size == 0:
        return "unknown"

    roi = cv2.GaussianBlur(roi_bgr, (5, 5), 0)

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 1] -= 128.0
    lab[:, :, 2] -= 128.0

    pixels = lab.reshape((-1, 3))

    _, labels, centers = cv2.kmeans(
        pixels.astype(np.float32), k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
        3,
        cv2.KMEANS_PP_CENTERS
    )

    counts = np.bincount(labels.flatten())
    total = counts.sum()

    # Sort clusters by size
    idxs = np.argsort(-counts)

    for idx in idxs:
        if counts[idx] < 0.05 * total:
            continue

        lab_color = centers[idx]
        name = classify_color_lab(lab_color)
        if name != "unknown":
            return name

    return "unknown"


def detect_object_color_night(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))

    # IR reflection → white
    b, g, r = cv2.split(roi_bgr.astype(np.float32))
    chroma = np.mean(np.abs(r - g)) + np.mean(np.abs(g - b))
    if mean > 170 and chroma < 10:
        return "white"

    if mean < 40:
        return "black"

    if mean < 120:
        return "gray"

    return "white"

def classify_color_lab(lab_color):
    # -----------------------------------------
    # Reference LAB colors (approximate swatches)
    # -----------------------------------------
    REF_COLORS = {
        # --- Primary automotive colors ---
        "red":      np.array([53,   80,   67]),
        "blue":     np.array([32,   79, -108]),
        "green":    np.array([87,  -86,   83]),
        "yellow":   np.array([97,  -21,   94]),
        "orange":   np.array([65,   45,   70]),
        "purple":   np.array([60,   98,  -60]),
        "pink":     np.array([75,   25,   -5]),
        "cyan":     np.array([91,  -48,  -14]),

        # --- Earth tones (common in cars & clothing) ---
        "brown":    np.array([40,   15,   20]),
        "beige":    np.array([78,    0,   18]),
        "tan":      np.array([70,    5,   30]),

        # --- Metallics (approximate LAB reflectance centers) ---
        "silver":   np.array([82,    0,    0]),
        "gold":     np.array([75,    5,   55]),

        # --- Additional useful colors ---
        "lime":     np.array([90,  -70,   80]),
        "teal":     np.array([60,  -40,  -10]),
        "navy":     np.array([20,   10,  -40]),
    }

    L, a, b = lab_color
    chroma = sqrt(a*a + b*b)

    # --- Neutral colors ---
    if L < 35:
        return "black"

    if chroma < 12:
        if L > 180:
            return "white"
        return "gray"

    # --- Metallics ---
    # Silver: mid‑L, low chroma
    if 55 < L < 110 and chroma < 22:
        return "silver"

    # Gold: warm b channel + moderate chroma
    if 55 < L < 110 and 22 <= chroma < 45 and b > 15:
        return "gold"

    # --- Earth tones ---
    if 35 < L < 80 and 12 < chroma < 40:
        if b > 25:
            return "tan"
        if 10 < b <= 25:
            return "beige"
        if b <= 10:
            return "brown"

    # --- Standard colors (LAB distance) ---
    best = None
    best_dist = 1e9

    for name, ref in REF_COLORS.items():
        dist = np.linalg.norm(lab_color - ref)
        if dist < best_dist:
            best_dist = dist
            best = name

    return best
