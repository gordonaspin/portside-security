import time
from typing import Tuple
from logging import getLogger

import cv2
import numpy as np
from numpy.typing import NDArray
from ultralytics.engine.results import Results

from nvr.camera.camera import Camera
from utils.utils import tags_to_str

logger = getLogger("pynvr")

def draw_status_text(frame, status_text, objects_text, is_recording):
    h, w = frame.shape[:2]
    layout = TextLayout(panel_h=h, panel_w=w)
    FONT = TextLayout.FONT

    def draw_text(layout, frame, text, x, y, font, color, bg):
        text_size, _ = cv2.getTextSize(text, font, layout.title_fs, layout.title_th)
        tw, th = text_size
        cv2.rectangle(frame, (x, y),
                      (x + tw + 2*layout.title_th, y + th + 4*layout.title_th),
                      bg, -1)
        cv2.putText(frame, text, (x+layout.title_th, y + th + layout.title_th),
                    font, layout.title_fs, color, layout.title_th)

    x = 0
    y = 0
    draw_text(layout, frame, status_text, x, y, FONT, (0,0,255) if is_recording else (0,255,0), (32,32,32))
    draw_text(layout, frame, objects_text, x, y + 4*layout.title_th + layout.title_spacing, FONT, (255,255,255), (32,32,32))
    #draw_text(layout, frame, f"{layout.title_th} {layout.title_fs:.2f} {layout.title_spacing} {layout.ref_w} {layout.ref_h} {layout.panel_w} {layout.panel_h}", x, y + 4*layout.title_th + layout.title_spacing, FONT, (255,255,255), (32,32,32))


class TextLayout:
    """
    Resolution‑independent text layout.
    Produces identical‑looking text on 480p, 1080p, 4K, 8K.
    """

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, panel_h: int, panel_w: int):
        self.panel_h = panel_h
        self.panel_w = panel_w

        # Precompute reference glyph metrics at scale=1
        self.ref_w, self.ref_h = self._measure_ref_glyph()

        # dbg() text: ~15–20 lines
        self.dbg_fs, self.dbg_th, self.dbg_spacing = \
            self._compute(lines=20, text_ratio=0.60)

        # Titles
        self.title_fs, self.title_th, self.title_spacing = \
            self._compute(lines=20, text_ratio=0.80)

        # Dashboard
        self.dash_fs, self.dash_th, self.dash_spacing = \
            self._compute(lines=20, text_ratio=0.75)

        # YOLO labels
        self.label_fs, self.label_th, self.label_spacing = \
            self._compute(lines=20, text_ratio=0.70)

    # ------------------------------------------------------------
    # Reference glyph measurement
    # ------------------------------------------------------------
    def _measure_ref_glyph(self):
        """
        Measure the true vertical extent of the font at scale=1.
        'Xg' gives cap height + descender.
        """
        (w, h), _ = cv2.getTextSize("Xg", self.FONT, 1.0, 1)
        return w / 2, h

    # ------------------------------------------------------------
    # Resolution‑independent scaling
    # ------------------------------------------------------------
    def _compute(self, lines: int, text_ratio: float):
        """
        Compute font scale so that text height is identical across resolutions.
        """
        # Desired pixel height per line
        target_line_h = int((self.panel_h / lines) * text_ratio)

        # Scale so that rendered height matches target height
        font_scale = target_line_h / self.ref_h

        # Stroke thickness proportional to font scale, not resolution
        thickness = self.panel_w * 2 // 704 #int(font_scale*2)

        # Vertical spacing
        spacing = int(target_line_h)

        return font_scale, thickness, spacing

    # ------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------
    def px(self, frac_w: float) -> int:
        return int(self.panel_w * frac_w)

    def py(self, frac_h: float) -> int:
        return int(self.panel_h * frac_h)


def draw_debug_panels(
    camera: Camera,
    model_names: dict[int, str],
    frame_count: int,
    frame_bgr: NDArray[np.uint8],
    result: Results,
    status_text: str,
    objects_text: str,
    is_recording: bool,
    krs, kcs, dsrs, dscs, dars, dacs
):
    h, w = frame_bgr.shape[:2]
    layout = TextLayout(panel_h=h, panel_w=w)
    FONT = TextLayout.FONT

    # --- ORIGINAL PANEL ---
    if result is not None:
        yolo_img = result.plot(pil=False).copy()
        orig_panel = cv2.resize(yolo_img, (w, h))
    else:
        orig_panel = frame_bgr.copy()

    draw_status_text(
        orig_panel,
        status_text,
        objects_text,
        is_recording,
    )

    title_pos = (layout.px(0.01), layout.py(0.90))
    cv2.putText(
        orig_panel,
        "Original Frame (YOLO)",
        title_pos,
        FONT,
        layout.title_fs,
        (0, 255, 255),
        layout.title_th,
    )

    # --- BACKGROUND PANEL ---
    bg_panel = cv2.cvtColor(
        cv2.convertScaleAbs(camera.buffers.background_buf),
        cv2.COLOR_GRAY2BGR
    )
    cv2.putText(
        bg_panel,
        "Background Model",
        title_pos,
        FONT,
        layout.title_fs,
        (0, 255, 255),
        layout.title_th,
    )

    # --- DIFF PANEL ---
    diff_panel = cv2.cvtColor(camera.buffers.diff_filtered_buf, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        diff_panel,
        "Diff (Filtered)",
        title_pos,
        FONT,
        layout.title_fs,
        (0, 255, 255),
        layout.title_th,
    )

    # --- THRESH PANEL ---
    thresh_panel = cv2.cvtColor(camera.buffers.thresh_buf, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        thresh_panel,
        "Threshold",
        title_pos,
        FONT,
        layout.title_fs,
        (0, 255, 255),
        layout.title_th,
    )

    # --- MOTION BOXES ---
    for (x1, y1, x2, y2) in krs:
        cv2.rectangle(thresh_panel, (x1, y1), (x2, y2),
                      (0, 255, 0), layout.label_th)

    # --- PER-CONTOUR METRICS ---
    for cnt in kcs:
        x, y, w0, h0 = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        roi_edges = camera.buffers.edges_buf[y:y+h0, x:x+w0]
        edge_density = cv2.countNonZero(roi_edges) / max(1, (w0*h0))
        aspect = max(w0, h0) / max(1, min(w0, h0))

        color = (0, 255, 0)
        cv2.rectangle(thresh_panel, (x, y), (x+w0, y+h0), color, layout.label_th)

        text = f"S:{solidity:.2f} E:{edge_density:.2f} A:{aspect:.1f}"
        cv2.putText(
            thresh_panel,
            text,
            (x, y - layout.py(0.005)),
            FONT,
            layout.label_fs,
            color,
            layout.label_th,
        )

    # --- YOLO BOXES ---
    if result is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = f"{model_names[cls_id]} {conf:.2f}"

            cv2.rectangle(thresh_panel, (x1, y1), (x2, y2),
                          (255, 255, 255), layout.label_th)
            cv2.putText(
                thresh_panel,
                label,
                (x1, y1 - layout.py(0.005)),
                FONT,
                layout.label_fs,
                (255, 255, 255),
                layout.label_th,
            )

    # --- DEBUG TEXT + DASHBOARD ---
    draw_combined_debug_layout_scaled(camera, thresh_panel, frame_count, layout)

    # --- RESIZE PANELS ---
    half_w = w // 2
    half_h = h // 2

    p1 = cv2.resize(orig_panel, (half_w, half_h))
    p2 = cv2.resize(bg_panel, (half_w, half_h))
    p3 = cv2.resize(diff_panel, (half_w, half_h))
    p4 = cv2.resize(thresh_panel, (half_w, half_h))

    return np.vstack((np.hstack((p1, p2)),
                      np.hstack((p3, p4))))


def draw_combined_debug_layout_scaled(
    camera: Camera,
    panel: NDArray[np.uint8],
    frame_count: int,
    layout: TextLayout
):
    FONT = TextLayout.FONT
    h, w = panel.shape[:2]

    # --- LEFT COLUMN ---
    x = layout.px(0.02)
    y = layout.py(0.12)

    def dbg(text, x, y, color=(0, 255, 255)):
        cv2.putText(
            panel,
            text,
            (x, y),
            FONT,
            layout.dbg_fs,
            color,
            layout.dbg_th,
        )

    dbg(f"frames={frame_count}", x, y)
    dbg(f"recording={camera.recording_state.recording}", x, y:=y + layout.dbg_spacing)
    dbg(f"should_record={camera.recording_state.should_record}", x, y:=y + layout.dbg_spacing)
    dbg(f"should_continue={camera.recording_state.should_continue}", x, y:=y + layout.dbg_spacing)
    dbg(f"motion_confidence={camera.motion.motion_confidence:.2f}/"
        f"{camera.motion.profile.min_motion_confidence.value:.2f}", x, y:=y + layout.dbg_spacing)
    dbg(f"score={camera.motion.score}/"
        f"{camera.motion.profile.motion_threshold_pixels:.2f}", x, y:=y + layout.dbg_spacing)
    dbg(f"  pixel_score={camera.motion.pixel_score:.2f}", x, y:=y + layout.dbg_spacing)
    dbg(f"  box_score={camera.motion.box_score:.2f}", x, y:=y + layout.dbg_spacing)
    dbg(f"  persist_score={camera.motion.persist_score:.2f}", x, y:=y + layout.dbg_spacing)
    dbg(f"has_moving_object={camera.motion.has_moving_object}", x, y:=y + layout.dbg_spacing)
    dbg(f"motion_boxes={len(camera.motion.motion_boxes_list)}", x, y:=y + layout.dbg_spacing)
    dbg(f"motion_persistence={camera.motion.motion_persistence:.2f}/"
        f"{camera.motion.profile.min_motion_frames.value}", x, y:=y + layout.dbg_spacing)
    dbg(f"since_last_motion={time.time() - camera.motion.last_motion_time:.2f}s", x, y:=y + layout.dbg_spacing)
    dbg(f"stop_conf={max(0.10, camera.motion.profile.min_motion_confidence.value * 0.30):.2f}", x, y:=y + layout.dbg_spacing)
    dbg(f"objects={tags_to_str(camera.motion.active_objects_dict)}", x, y:=y + layout.dbg_spacing)

    # --- RIGHT COLUMN (Dashboard) ---
    x = layout.px(0.50)
    y = layout.py(0.12)

    dbg("Auto-Tuner Dashboard", x, y)
    tuner = camera.tuner.tuner
    stats = tuner.summarize()
    recs = tuner.recommend_adjustments()
    dbg("Rule Hits:", x, y:=y + layout.dbg_spacing)
    for rule, count in sorted(stats.items(), key=lambda x: -x[1])[:6]:
        color = (0, 255, 0) if "accepted" in rule else (0, 165, 255)
        dbg(f"{rule}: {count}", x + layout.px(0.04), y:=y + layout.dbg_spacing, color)

    dbg("Recommendations:", x, y:=y + layout.dbg_spacing)
    if not recs:
        dbg("No adjustments needed.", x + layout.px(0.04), y:=y + layout.dbg_spacing, color=(0, 255, 0))
    else:
        for k, v in recs.items():
            dbg(f"{k} -> {v}", x + layout.px(0.04), y:=y + layout.dbg_spacing, color=(0, 255, 255))

    return panel
