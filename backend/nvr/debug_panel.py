import time
from datetime import datetime
from typing import Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from ultralytics.engine.results import Results

from camera.camera import Camera
from nvr.utils import tags_to_str

def draw_debug_panels(
                    camera: Camera,
                    frame_bgr: NDArray[np.uint8],
                    result: Results,
                    krs: list[Tuple[int, int, int, int]],
                    kcs: list[NDArray[np.int32]],
                    dsrs: list[Tuple[int, int, int, int]],
                    dscs: list[NDArray[np.int32]],
                    dars: list[Tuple[int, int, int, int]],
                    dacs: list[NDArray[np.int32]]
                    ):

    # --- BUILD 4-PANEL DEBUG COMPOSITE ---

    # 1. Original frame (with YOLO annotations via result.plot)
    if result is not None:
        orig_panel = result.plot(pil=False).copy()
    else:
        orig_panel = frame_bgr.copy()

    draw_status_text(orig_panel, camera.status_text, camera.objects_text, camera.recording)

    TITLE_Y = 40
    cv2.putText(orig_panel, "Original Frame (YOLO)", (10, TITLE_Y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 2. Background model
    bg_panel = cv2.convertScaleAbs(camera.background_buf)
    bg_panel = cv2.cvtColor(bg_panel, cv2.COLOR_GRAY2BGR)
    cv2.putText(bg_panel, "Background Model", (10, TITLE_Y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 3. Diff (filtered)
    diff_panel = cv2.cvtColor(camera.diff_filtered_buf, cv2.COLOR_GRAY2BGR)
    cv2.putText(diff_panel, "Diff (Filtered)", (10, TITLE_Y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 4. Threshold panel
    thresh_panel = cv2.cvtColor(camera.thresh_buf, cv2.COLOR_GRAY2BGR)
    cv2.putText(thresh_panel, "Threshold", (10, TITLE_Y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # --- DRAW MOTION BOXES ON THRESH PANEL ---
    for (x1, y1, x2, y2) in krs:
        cv2.rectangle(thresh_panel, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #for (x1, y1, x2, y2) in dsrs:
    #    cv2.rectangle(thresh_panel, (x1, y1), (x2, y2), (0, 165, 255), 2)

    #for (x1, y1, x2, y2) in dars:
    #    cv2.rectangle(thresh_panel, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # --- PER-CONTOUR METRICS ---
    for cnt in kcs:# + dscs + dacs:
        x, y, w0, h0 = cv2.boundingRect(cnt)

        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        roi_edges = camera.edges_buf[y:y+h0, x:x+w0]
        edge_density = cv2.countNonZero(roi_edges) / max(1, (w0 * h0))

        aspect = max(w0, h0) / max(1, min(w0, h0))

        if any(cnt is kc for kc in kcs):
            color = (0, 255, 0)
        elif any(cnt is dsc for dsc in dscs):
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)

        cv2.rectangle(thresh_panel, (x, y), (x + w0, y + h0), color, 2)

        text = f"S:{solidity:.2f} E:{edge_density:.2f} A:{aspect:.1f}"
        cv2.putText(thresh_panel, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)

    # --- YOLO ANNOTATIONS ON THRESH PANEL ---
    if result is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = f"{camera.model.names[cls_id]} {conf:.2f}"

            cv2.rectangle(thresh_panel, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(thresh_panel, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)

    # --- DEBUG TEXT ---
    draw_combined_debug_layout(camera, thresh_panel)

    # --- RESIZE PANELS ---
    h, w = frame_bgr.shape[:2]
    half_w = w // 2
    half_h = h // 2

    p1 = cv2.resize(orig_panel, (half_w, half_h))
    p2 = cv2.resize(bg_panel, (half_w, half_h))
    p3 = cv2.resize(diff_panel, (half_w, half_h))
    p4 = cv2.resize(thresh_panel, (half_w, half_h))

    # --- STACK INTO 4-PANEL COMPOSITE ---
    top = np.hstack((p1, p2))
    bottom = np.hstack((p3, p4))
    composite = np.vstack((top, bottom))

    return composite

def draw_status_text(
    frame_bgr: NDArray[np.uint8],
    camera_text: str,
    objects_text: str,
    is_recording: bool,
) -> None:
    """
    Draw status text and object text on the frame.
    Preserves original colors and shadow styling.
    """
    def draw_text(frame, text, position, font, font_scale, color, thickness, bg_color):

        x, y = position
        text_size, _ = cv2.getTextSize(text, font, 0.7, thickness)
        text_w, text_h = text_size
        # rectangle: (x, y) top-left, (x + text_w, y + text_h) bottom-right
        # text: (x, y + text_h) bottom-left corner
        cv2.rectangle(frame, (x, y-2), (x + text_w + 2, y + text_h + 6), bg_color, -1)
        cv2.putText(frame, text, (x+1, y + text_h), font, font_scale, color, thickness)

    draw_text(
        frame_bgr, camera_text, (0, 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (0, 0, 255) if is_recording else (0, 255, 0),
        2, (32, 32, 32)
    )

    draw_text(
        frame_bgr, objects_text, (0, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (255, 255, 255), 2, (32, 32, 32)
    )


def draw_tuner_dashboard(camera: Camera, panel):
    """
    Draws a live tuner dashboard overlay on the given panel.
    Shows rule hit counts, last decisions, and recommendations.
    """

    tuner = camera.auto_tuner
    stats = tuner.summarize()
    recs  = tuner.recommend_adjustments()

    # --- Dashboard box ---
    x0, y0 = 10, 10
    w, h = 420, 260
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (32, 32, 32), -1)
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (255, 255, 255), 2)

    # --- Stats Section ---
    y = y0 + 30
    cv2.putText(panel, "Rule Hits:", (x0 + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    y += 25

    # Show top 6 most frequent rules
    for rule, count in sorted(stats.items(), key=lambda x: -x[1])[:6]:
        color = (0, 255, 0) if "accepted" in rule else (0, 165, 255)
        cv2.putText(panel, f"{rule}: {count}",
                    (x0 + 20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 2)
        y += 22

    # --- Recommendations Section ---
    y += 10
    cv2.putText(panel, "Recommendations:",
                (x0 + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 2)
    y += 25

    if not recs:
        cv2.putText(panel, "All thresholds stable",
                    (x0 + 20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
    else:
        for k, v in recs.items():
            cv2.putText(panel, f"{k}: {v}",
                        (x0 + 20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 2)
            y += 22

    return panel

def draw_combined_debug_layout(camera: Camera, thresh_panel):
    """
    Left column  = dbg() motion stats
    Right column = auto-tuner dashboard
    """

    h, w = thresh_panel.shape[:2]

    # -----------------------------
    # LEFT COLUMN: dbg() output
    # -----------------------------
    xL = 10
    yL = 60
    spacing = 20

    def dbg(text, color=(0,255,255)):
        nonlocal yL
        cv2.putText(
            thresh_panel, text, (xL, yL),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            color, 2
        )
        yL += spacing

    # --- Your existing dbg() content ---
    dbg(f"frames={camera.frame_count}")
    dbg(f"recording={camera.recording}")
    dbg(f"should_record={camera.should_record}")
    dbg(f"should_start={camera.should_start}")
    dbg(f"should_continue={camera.should_continue}")

    dbg(f"has_moving_object={camera.has_moving_object}")
    dbg(f"motion_boxes={len(camera.motion_boxes_list)}")

    dbg(f"score={camera.score} / {camera.profile.motion_threshold_pixels}")
    dbg(f"motion_confidence={camera.motion_confidence:.2f} / {camera.profile.min_motion_confidence.value}")
    dbg(f"STOP_CONF={max(0.10, camera.profile.min_motion_confidence.value * 0.30):.2f}")

    dbg(f"motion_persistence={camera.motion_persistence} / {camera.profile.min_motion_frames.value}")
    dbg(f"persist_score={camera.persist_score:.2f}")

    dbg(f"since_last_motion={time.time() - camera.last_motion_time:.2f}s")
    dbg(f"pixel_score={camera.pixel_score:.2f}")
    dbg(f"box_score={camera.box_score:.2f}")

    dbg(f"objects={tags_to_str(camera.active_objects_dict)}")


    # -----------------------------
    # RIGHT COLUMN: tuner dashboard
    # -----------------------------
    tuner = camera.auto_tuner
    stats = tuner.summarize()
    recs  = tuner.recommend_adjustments()

    # Dashboard box
    dash_w = 420
    dash_h = 420
    xR = w - dash_w - 10
    yR = 10

    #cv2.rectangle(thresh_panel, (xR, yR), (xR + dash_w, yR + dash_h), (32, 32, 32), -1)
    cv2.rectangle(thresh_panel, (xR, yR), (xR + dash_w, yR + dash_h), (255, 255, 255), 2)

    # Title
    cv2.putText(thresh_panel, "AUTO-TUNER DASHBOARD",
                (xR + 10, yR + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2)

    # Rule hits
    y = yR + 60
    cv2.putText(thresh_panel, "Rule Hits:",
                (xR + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 2)
    y += 25

    for rule, count in sorted(stats.items(), key=lambda x: -x[1])[:6]:
        color = (0, 255, 0) if "accepted" in rule else (0, 165, 255)
        cv2.putText(thresh_panel, f"{rule}: {count}",
                    (xR + 20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 2)
        y += 22

    # Recommendations
    y += 10
    cv2.putText(thresh_panel, "Recommendations:",
                (xR + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 2)
    y += 25

    if not recs:
        cv2.putText(thresh_panel, "All thresholds stable",
                    (xR + 20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 2)
    else:
        for k, v in recs.items():
            cv2.putText(thresh_panel, f"{k}: {v}",
                        (xR + 20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 255), 2)
            y += 22

    return thresh_panel


