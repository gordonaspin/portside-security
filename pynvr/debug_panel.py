import os
import time
from logging import getLogger

import cv2
import numpy as np
import PIL
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from ultralytics.engine.results import Results

from .utils import tags_to_str

logger = getLogger("pynvr")

FONT_PATH = os.path.join(os.path.dirname(PIL.__file__), "fonts", "DejaVuSansMono.ttf")

def begin_pillow_draw(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    return pil_img, draw

def end_pillow_draw(frame, pil_img):
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    frame[:,:,:] = bgr

def draw_text(draw, text, x, y, font, color="yellow", bg=None):
    """Draw text with optional background box."""
    bbox = draw.textbbox((x, y), "|"+text[:-1], font=font) # trick to get full height box
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    padding = 2

    if bg is not None:
        draw.rectangle([x, y, x + tw + padding*2, y + th + padding*2], fill=bg)

    draw.text((x + padding, y), text, font=font, fill=color)
    return th + padding*2

class TextLayout:
    def __init__(self, panel_h, panel_w):
        self.panel_h = panel_h
        self.panel_w = panel_w

        self.font_path = FONT_PATH
        self.ref_font_size = 100
        self.ref_h = self._measure_ref_height()

        self.title_fs, self.title_spacing = self._compute(20, 0.90)
        self.dbg_fs,   self.dbg_spacing   = self._compute(20, 0.80)
        self.label_fs, self.label_spacing = self._compute(20, 0.80)

    def _measure_ref_height(self):
        font = ImageFont.truetype(self.font_path, self.ref_font_size)
        bbox = font.getbbox("Xg")
        return bbox[3] - bbox[1]

    def _compute(self, lines, text_ratio):
        target_line_h = (self.panel_h / lines) * text_ratio
        font_size = int((target_line_h / self.ref_h) * self.ref_font_size)
        spacing = int(target_line_h)
        return font_size, spacing

    def font(self, size):
        return ImageFont.truetype(self.font_path, size)

    def px(self, frac):
        return int(self.panel_w * frac)

    def py(self, frac):
        return int(self.panel_h * frac)

def draw_debug_panels(
    camera,
    model_names,
    frame_count,
    frame_bgr,
    yolo_result,
    active_tracks
):
    """
    ByteTrack-only debug mosaic:
    - Panel 1: Original frame with YOLO + Track IDs
    - Panel 2: Motion panel (tracks + speeds)
    - Panel 3: Empty placeholder (future use)
    - Panel 4: Debug text (recording state, track count, etc.)
    """

    h, w = frame_bgr.shape[:2]
    half_w = w // 2
    half_h = h // 2

    # ---------------- PANEL 1: ORIGINAL + YOLO + TRACKS ----------------
    p1 = frame_bgr.copy()

    # Draw YOLO boxes (only if moving)
    if camera.motion.has_moving_object and yolo_result is not None:
        for box in yolo_result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls)
            conf = float(box.conf)
            label = f"{model_names[cls_id]} {conf:.2f}"
            cv2.rectangle(p1, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(p1, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # Draw ByteTrack tracks
    for t in active_tracks:
        # Case 1: ByteTrack STrack object
        if hasattr(t, "tlbr"):
            x1, y1, x2, y2 = t.tlbr

        # Case 2: raw tuple (x1, y1, x2, y2)
        elif isinstance(t, tuple) or isinstance(t, list):
            if len(t) >= 4:
                x1, y1, x2, y2 = t[:4]
            else:
                continue  # malformed tuple, skip
        else:
            continue  # unknown type, skip

        cv2.rectangle(p1, (x1, y1), (x2, y2), (0,128,255), 2)
        if hasattr(t, "track_id") and hasattr(t, "relative_speed"):
            cv2.putText(p1, f"i:{t.track_id} v:{t.relative_speed:.1f}",
                        (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0,128,255), 1)

    p1 = cv2.resize(p1, (half_w, half_h), cv2.INTER_AREA)

    # ---------------- PANEL 2: MOTION PANEL ----------------
    # Shows only tracks + speeds on a black background
    p2 = np.zeros_like(p1)

    for t in active_tracks:
        # Case 1: ByteTrack STrack object
        if hasattr(t, "tlbr"):
            x1, y1, x2, y2 = t.tlbr

        # Case 2: raw tuple (x1, y1, x2, y2)
        elif isinstance(t, tuple) or isinstance(t, list):
            if len(t) >= 4:
                x1, y1, x2, y2 = t[:4]
            else:
                continue  # malformed tuple, skip
        else:
            continue  # unknown type, skip

        # scale to half-res
        sx1 = int(x1 * half_w / w)
        sy1 = int(y1 * half_h / h)
        sx2 = int(x2 * half_w / w)
        sy2 = int(y2 * half_h / h)

        cv2.rectangle(p2, (sx1, sy1), (sx2, sy2), (0,255,255), 2)
        if hasattr(t, "track_id") and hasattr(t, "relative_speed"):
            cv2.putText(p2, f"i:{t.track_id} v:{t.relative_speed:.1f}",
                        (sx1, sy1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0,255,255), 1)

    # ---------------- PANEL 3: EMPTY / FUTURE USE ----------------
    p3 = np.zeros_like(p1)
    cv2.putText(p3, "reserved", (10, half_h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128,128,128), 2)

    # ---------------- PANEL 4: DEBUG TEXT ----------------
    p4 = np.zeros_like(p1)

    pil_img, draw = begin_pillow_draw(p4)
    layout = TextLayout(half_h, half_w)
    font_dbg = layout.font(layout.dbg_fs)

    x = layout.px(0.02)
    y = layout.py(0.02)

    def line(text, color="white"):
        nonlocal y
        y += draw_text(draw, text, x, y, font_dbg, color)

    line(f"frame={frame_count}")
    line(f"recording={camera.recording_state.recording}")
    line(f"should_record={camera.recording_state.should_record}")
    line(f"should_continue={camera.recording_state.should_continue}")
    line(f"has_moving_object={camera.motion.has_moving_object}")
    line(f"active_tracks={len(active_tracks)}")
    line(f"since_last_motion={time.time() - camera.motion.last_motion_time:.2f}s")
    line(f"objects={tags_to_str(camera.motion.active_objects_dict)}")

    end_pillow_draw(p4, pil_img)

    # ---------------- COMBINE PANELS ----------------
    return np.vstack((np.hstack((p1, p2)),
                      np.hstack((p3, p4))))
