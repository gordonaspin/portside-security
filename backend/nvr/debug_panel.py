import os
import time
from logging import getLogger

import cv2
import numpy as np
import PIL
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from ultralytics.engine.results import Results

from nvr.camera.camera import Camera
from utils.utils import tags_to_str

logger = getLogger("pynvr")

FONT_PATH = os.path.join(os.path.dirname(PIL.__file__), "fonts", "DejaVuSansMono.ttf")

def begin_pillow_draw(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    return pil_img, draw

def draw_text(draw, text, x, y, font, color, bg=None):
    """Draw text with optional background box."""
    bbox = draw.textbbox((x, y), "|"+text[:-1], font=font) # trick to get full height box
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    padding = 2

    if bg is not None:
        draw.rectangle([x, y, x + tw + padding*2, y + th + padding*2], fill=bg)

    draw.text((x + padding, y), text, font=font, fill=color)
    return th + padding*2

def end_pillow_draw(frame, pil_img):
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    frame[:,:,:] = bgr

def draw_status_text(frame, status_text, objects_text, is_recording):
    layout = TextLayout(panel_h=frame.shape[0], panel_w=frame.shape[1])
    font = layout.font(layout.title_fs)

    # Convert once
    pil_img, draw = begin_pillow_draw(frame)

    x = 0
    y = 0

    # First line
    h1 = draw_text(
        draw,
        status_text,
        x=x,
        y=y,
        font=font,
        color="red" if is_recording else "lime",
        bg=(32,32,32)
    )

    # Second line
    if objects_text:
        y += h1
        h2 = draw_text(
            draw,
            objects_text,
            x=x,
            y=y,
            font=font,
            color="white",
            bg=(32,32,32)
        )

    # Convert back once
    end_pillow_draw(frame, pil_img)


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
    result,
    status_text,
    objects_text,
    is_recording,
    krs, kcs, dsrs, dscs, dars, dacs
):
    h, w = frame_bgr.shape[:2]

    # ---------------- BUILD PANELS (NO TEXT YET) ----------------
    if result is not None:
        yolo_img = result.plot(pil=False).copy()
        orig_panel = cv2.resize(yolo_img, (w, h))
    else:
        orig_panel = frame_bgr.copy()

    bg_panel = cv2.cvtColor(
        cv2.convertScaleAbs(camera.buffers.background_buf),
        cv2.COLOR_GRAY2BGR
    )

    diff_panel = cv2.cvtColor(camera.buffers.diff_filtered_buf, cv2.COLOR_GRAY2BGR)
    thresh_panel = cv2.cvtColor(camera.buffers.thresh_buf, cv2.COLOR_GRAY2BGR)

    CV2_GREEN = (0,255,0)
    CV2_WHITE = (255,255,255)
    CV2_YELLOW = (255,255,0)
    CV2_RED = (0,0,255)
    # Draw rectangles BEFORE resizing (shapes scale fine)
    for (x1, y1, x2, y2) in krs:
        cv2.rectangle(thresh_panel, (x1, y1), (x2, y2), CV2_GREEN, 2)

    for cnt in kcs:
        x, y, w0, h0 = cv2.boundingRect(cnt)
        cv2.rectangle(thresh_panel, (x, y), (x+w0, y+h0), CV2_GREEN, 2)

    if result is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # YOLO's own color for this class
            cls_id = int(box.cls[0])
            color = result.names.get(cls_id, (255, 255, 255))

            # Ultralytics stores colors as RGB floats 0–255
            if hasattr(result, "plot"):
                # YOLO v8/v9: colors stored in result.plot() palette
                try:
                    color = result.plot().names[cls_id]
                except:
                    pass

            # Convert to BGR for OpenCV
            if isinstance(color, (list, tuple)) and len(color) == 3:
                bgr = (int(color[2]), int(color[1]), int(color[0]))
            else:
                bgr = (255, 255, 255)

            cv2.rectangle(thresh_panel, (x1, y1), (x2, y2), bgr, 2)


    # ---------------- RESIZE FIRST (CRITICAL FOR SHARP TEXT) ----------------
    half_w = w // 2
    half_h = h // 2

    p1 = cv2.resize(orig_panel, (half_w, half_h), cv2.INTER_AREA)
    p2 = cv2.resize(bg_panel, (half_w, half_h), cv2.INTER_AREA)
    p3 = cv2.resize(diff_panel, (half_w, half_h), cv2.INTER_AREA)
    p4 = cv2.resize(thresh_panel, (half_w, half_h), cv2.INTER_AREA)

    # ---------------- DRAW TEXT AFTER RESIZE ----------------
    layout = TextLayout(half_h, half_w)
    font_title = layout.font(layout.title_fs)
    font_label = layout.font(layout.label_fs)

    # --- Panel 1: Original ---
    pil_img, draw = begin_pillow_draw(p1)

    draw_text(draw, "original frame",
              layout.px(0.01), layout.py(0.80),
              font_title, "yellow")
    end_pillow_draw(p1, pil_img)

    # --- Panel 2: Background ---
    pil_img, draw = begin_pillow_draw(p2)
    draw_text(draw, "background Model",
              layout.px(0.01), layout.py(0.90),
              font_title, "yellow")
    end_pillow_draw(p2, pil_img)

    # --- Panel 3: Diff ---
    pil_img, draw = begin_pillow_draw(p3)
    draw_text(draw, "diff (filtered)",
              layout.px(0.01), layout.py(0.90),
              font_title, "yellow")
    end_pillow_draw(p3, pil_img)

    # --- Panel 4: Threshold ---
    pil_img, draw = begin_pillow_draw(p4)
    draw_text(draw, "threshold",
              layout.px(0.01), layout.py(0.90),
              font_title, "yellow")
    
    # Per-contour metrics (scaled coords)
    scale_x = half_w / w
    scale_y = half_h / h

    for cnt in kcs:
        x, y, w0, h0 = cv2.boundingRect(cnt)
        xs = int(x * scale_x)
        ys = int(y * scale_y)
        text = f"S:{cv2.contourArea(cnt)/max(1,cv2.contourArea(cv2.convexHull(cnt))):.2f}"
        draw_text(draw, text, xs, ys - layout.py(0.005),
                  font_label, "green")

    # YOLO labels
    if result is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            xs = int(x1 * scale_x)
            ys = int(y1 * scale_y)
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = f"{model_names[cls_id]} {conf:.2f}"
            draw_text(draw, label, xs, ys - layout.py(0.005),
                      font_label, "white")

    end_pillow_draw(p4, pil_img)

    # --- Debug text (drawn on p4) ---
    draw_combined_debug_layout_scaled(camera, p4, frame_count, layout)

    # ---------------- COMBINE PANELS ----------------
    return np.vstack((np.hstack((p1, p2)),
                      np.hstack((p3, p4))))


def draw_combined_debug_layout_scaled(camera, panel, frame_count, layout):
    pil_img, draw = begin_pillow_draw(panel)
    font_dbg = layout.font(layout.dbg_fs)

    x = layout.px(0.01)
    y = layout.py(0.01)

    def dbg(text, x, y, color="yellow"):
        draw_text(draw, text, x, y, font_dbg, color)
        return y + layout.dbg_spacing

    # LEFT COLUMN
    y = dbg(f"frames={frame_count}", x, y)
    y = dbg(f"recording={camera.recording_state.recording}", x, y)
    y = dbg(f"should_record={camera.recording_state.should_record}", x, y)
    y = dbg(f"should_continue={camera.recording_state.should_continue}", x, y)
    y = dbg(f"motion_confidence={camera.motion.motion_confidence:.2f}/"
            f"{camera.motion.profile.min_motion_confidence.value:.2f}", x, y)
    y = dbg(f"score={camera.motion.score}/"
            f"{camera.motion.profile.motion_threshold_pixels:.2f}", x, y)
    y = dbg(f"pixel_score={camera.motion.pixel_score:.2f}", x, y)
    y = dbg(f"box_score={camera.motion.box_score:.2f}", x, y)
    y = dbg(f"persist_score={camera.motion.persist_score:.2f}", x, y)
    y = dbg(f"has_moving_object={camera.motion.has_moving_object}", x, y)
    y = dbg(f"motion_boxes={len(camera.motion.motion_boxes_list)}", x, y)
    y = dbg(f"motion_persistence={camera.motion.motion_persistence:.2f}/"
            f"{camera.motion.profile.min_motion_frames.value}", x, y)
    y = dbg(f"since_last_motion={time.time() - camera.motion.last_motion_time:.2f}s", x, y)
    y = dbg(f"stop_conf={max(0.10, camera.motion.profile.min_motion_confidence.value * 0.30):.2f}", x, y)
    y = dbg(f"objects={tags_to_str(camera.motion.active_objects_dict)}", x, y)

    # RIGHT COLUMN
    x = layout.px(0.50)
    y = layout.py(0.01)

    y = dbg("Auto-Tuner Dashboard", x, y)
    tuner = camera.tuner.tuner
    stats = tuner.summarize()
    recs = tuner.recommend_adjustments()

    y = dbg("Rule Hits:", x, y)
    for rule, count in sorted(stats.items(), key=lambda x: -x[1])[:6]:
        color = "green" if "accepted" in rule else "tomato"
        y = dbg(f"{rule}: {count}", x + layout.px(0.04), y, color)

    y = dbg("Recommendations:", x, y)
    if not recs:
        y = dbg("No adjustments needed.", x + layout.px(0.04), y, color="green")
    else:
        for k, v in recs.items():
            y = dbg(f"{k} -> {v}", x + layout.px(0.04), y, color="yellow")

    end_pillow_draw(panel, pil_img)
    return panel
