from datetime import datetime
import os
from pathlib import Path
import time
import threading

import cv2
import torch
import numpy as np
import gradio as gr

import constants
from logger import log_event, event_log
from context import Context
from model import Model
from nvr import NVR
from camera import Camera


LOG_STREAM_DIV = """
    <div class="inner-log" style="
        height: 300px; 
        overflow-y: auto; 
        border: 1px solid #ccc; 
        padding: 5px; 
        font-family: monospace;
        font-size: small;
        background-color: #1e1e1e; 
        color: #ffffff;
        box-sizing: border-box;
    ">
    <div style="font-weight: bold; margin-bottom: 8px; font-size: small;">
        📜 Event Log
    </div>
    """

def _is_night_time(frame, brightness_threshold=50):
    # Convert to HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Calculate average brightness (V channel)
    mean_brightness = np.mean(hsv[:,:,2])
    
    # If brightness is low, it's likely night time
    return mean_brightness < brightness_threshold

def _keep_overlapping_any(boxes, ref_boxes):
    """
    boxes: YOLO r.boxes.xyxy  -> (N, 4)
    ref_boxes: cv2.boundingRect -> (M, 4) in (x, y, x1, y1)
    """

    # 1. Ensure correct shapes
    boxes = boxes.view(-1, 4)
    ref_boxes = ref_boxes.view(-1, 4)

    # 3. Compute pairwise intersection
    x1 = torch.maximum(boxes[:, None, 0], ref_boxes[None, :, 0])
    y1 = torch.maximum(boxes[:, None, 1], ref_boxes[None, :, 1])
    x2 = torch.minimum(boxes[:, None, 2], ref_boxes[None, :, 2])
    y2 = torch.minimum(boxes[:, None, 3], ref_boxes[None, :, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)

    overlap = (inter_w * inter_h) > 0  # (N, M)

    # 4. Keep if overlaps ANY ROI
    return overlap.any(dim=1)

# =========================
# STREAM
# =========================



class GUI:
    def __init__(self, ctx: Context, model: Model, nvr: NVR):
        self.ctx = ctx
        self.classes = ctx.classes
        self.width = ctx.resolution[0]
        self.height = ctx.resolution[1]
        self.selected_classes = model.class_to_index(self.classes)

        self.motion_threshold = self.ctx.motion_threshold
        self.confidence_threshold = self.ctx.confidence_threshold
        self.model = model
        self.nvr = nvr
        nvr.start()

    def make_stream_fn(self, camera: Camera):
        #active_events = {}
        #active_objects = {}
        #last_event_time = {}

        def stream():
            threading.current_thread().name = f"{camera.name} camera GUI"

            frame_gen = self.nvr.frame_generator(camera, self.width, self.height)

            prev_gray = None
            motion_frames = 0
            no_motion_frames = 0
            recording = False
            prev_time = time.time()

            log_event(message=f"reading from stream", level="info", camera=camera)
            while True:
                try:
                    ret, frame = next(frame_gen)
                except StopIteration:
                    break
                if not ret:
                    log_event(message="stream read failed, attempting to reconnect...", level="warn", camera=camera)
                    self.nvr.restart()
                    continue

                motion_threshold_index = 1 if _is_night_time(frame, 100) else 0

                # motion
                gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),(21,21),0)
                motion_boxes = []
                overlay = None

                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    _, thresh = cv2.threshold(diff,25,255,cv2.THRESH_BINARY)
                    score = cv2.countNonZero(thresh)

                    if score > self.motion_threshold[motion_threshold_index]:
                        #log_event(message=f"motion detected score: {score}, threshold: {self.motion_threshold[motion_threshold_index]}", levell="info", camera=camera)                    
                        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        overlay = frame.copy()
                        for contour in contours:
                            area = cv2.contourArea(contour)
                            x1, y1, w, h = cv2.boundingRect(contour)
                            x2 = x1 + w
                            y2 = y1 + h
                            if area < self.motion_threshold[motion_threshold_index] / 10:  # filter small noise
                                cv2.drawContours(overlay, [contour], -1, (0, 0, 255), 1)
                                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 1)
                                #log_event(message=f"ignoring motion contour rect ({x1}, {y1}), ({x2}, {y2}) with area {area}")
                            else:
                                cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 1)
                                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
                                #log_event(message=f"motion contour rect ({x1}, {y1}), ({x2}, {y2}) with area {area}")
                                motion_boxes.append([x1, y1, x2, y2])
                            if self.ctx.debug:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                tag = "_".join(classes_in_frame) if classes_in_frame else "motion"
                                image_filename = os.path.join(camera.images_dir, f"{timestamp}_{tag}.jpg")
                                cv2.imwrite(image_filename, overlay)

                prev_gray = gray

                # YOLO
                result = None
                classes_in_frame = set()

                if motion_boxes:
                    result = self.model.model.predict(frame, conf=self.confidence_threshold, classes=self.selected_classes if self.selected_classes else None, verbose=False)[0]
                    boxes = result.boxes.xyxy.reshape(-1, 4)
                    ref_motion_boxes = torch.as_tensor(motion_boxes, dtype=boxes.dtype, device=boxes.device)
                    keep = _keep_overlapping_any(boxes, ref_motion_boxes)
                    #log_event(message=f"motion_boxes {ref_motion_boxes.shape} {ref_motion_boxes}")
                    #log_event(message=f"boxes {boxes.shape} {boxes}")
                    #log_event(message=f"keep {keep.shape} {keep}")
                    boxes = result.boxes[keep]

                    for box in boxes:
                        classes_in_frame.add(self.model.model.names[int(box.cls)])

                # counters
                if motion_boxes:
                    motion_frames += 1
                    no_motion_frames = 0
                else:
                    no_motion_frames += 1

                valid_objects = len(classes_in_frame) > 0

                # start
                now = time.time()
                if motion_frames >= constants.MOTION_DETECT_FRAME_COUNT and not recording:
                    if (not constants.REQUIRE_OBJECT_FOR_RECORDING or valid_objects):
                        if now - camera.last_event_time > constants.EVENT_COOLDOWN:
                            recording = True
                            camera.active_events = self.nvr.get_segments(camera, constants.PRE_RECORD_SEGMENTS)
                            camera.active_objects = set(classes_in_frame)
                            camera.last_event_time = now
                            log_event(message=f"recording start {",".join(classes_in_frame) if classes_in_frame else ""}", level="info", camera=camera)

                # update
                if recording:
                    camera.active_events += self.nvr.get_segments(camera,1)
                    camera.active_objects.update(classes_in_frame)

                # stop
                if recording and no_motion_frames >= constants.NO_MOTION_DETECT_FRAME_COUNT:
                    recording = False
                    segments = list(dict.fromkeys(camera.active_events))

                    if segments:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        tag = "_".join(camera.active_objects) if camera.active_objects else "motion"

                        recording_filename = os.path.join(camera.recordings_dir, f"{timestamp}_{tag}.mp4")
                        self.nvr.merge_segments(segments, recording_filename)

                        recorded_cap = cv2.VideoCapture(recording_filename)
                        frame_count = recorded_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        #if frame_count < NO_MOTION_DETECT_FRAME_COUNT + cam_fps and os.path.isfile(recording_filename):
                        if frame_count < constants.NO_MOTION_DETECT_FRAME_COUNT + 20 and os.path.isfile(recording_filename):
                            os.remove(recording_filename)
                            log_event(message=f"recording auto-deleted {os.path.basename(recording_filename)} with {frame_count} frames", info="info", camera=camera, file_path=recording_filename)
                        else:
                            log_event(message=f"recording available {os.path.basename(recording_filename)}", level="record", camera=camera, file_path=recording_filename)

                    camera.active_events = {}
                    classes_in_frame = {}
                    camera.active_objects = {}
                    motion_frames = 0
                    no_motion_frames = 0

                # render
                img = result.plot() if result else frame
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if overlay is not None:
                    img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

                if not self.nvr.cameras[camera.name].hd:
                    img = cv2.resize(img, constants.RENDER_SIZE)

                fps = 1/(time.time()-prev_time)
                prev_time = time.time()

                status = "🔴 REC" if recording else "🟢 LIVE"
                yield img, f"{status} {"| Night " if motion_threshold_index else ""}| FPS {int(fps)}" + (f" | {",".join(classes_in_frame)}" if len(classes_in_frame) > 0 else "")
                
        return stream

    # =========================
    # UI HANDLERS
    # =========================
    def update_confidence_threshold(self, val):
        self.confidence_threshold = val
        log_event(message=f"confidence updated → {val}")

    def update_day_motion_threshold(self, val):
        self.motion_threshold[0] = val
        log_event(message=f"day motion threshold → {val}")

    def update_night_motion_threshold(self, val):
        self.motion_threshold[1] = val
        log_event(message=f"night motion threshold → {val}")

    def update_detection_classes(self, names):
        self.selected_classes = self.model.class_to_index(names)
        log_event(message=f"classes → {names}")

    def update_hd_mode(self, name, val):
        self.nvr.cameras[name].hd = val    

    # =========================
    # UI STREAMS
    # =========================
    # --- Event log streamer ---
    def log_stream(self):
        while True:
            html_content = "".join(event_log)
            # JS snippet to scroll to bottom
            scroll_js = "<script>var el=document.currentScript.parentElement; el.scrollTop=el.scrollHeight;</script>"
            yield LOG_STREAM_DIV +html_content + "</div>" + scroll_js
            time.sleep(0.5)

    def recordings_stream(self):
        while True:
            files=[]
            for r,_,f in os.walk(self.nvr.recordings_dir):
                for x in f:
                    if x.endswith(".mp4"):
                        files.append(os.path.join(r,x))

            files.sort(key=os.path.getmtime, reverse=True)

            html="""
            <div style="
                height: 200px;
                overflow-y: auto;
                border: 1px solid #ccc;
                padding: 5px;
                font-family: monospace;
                font-size: small;
                background-color: #1e1e1e;
                color: #ffffff;
                box-sizing: border-box;
            ">
                <div style="font-weight: bold; margin-bottom: 8px; font-size: medium;">
                    🎥 Recordings
                </div>
            """
            for f in files:
                p = Path(f)
                html+=f'<a href="/gradio_api/file={f}" target="_blank" style="color: white;">{p.parent.name}/{p.name}</a><br>'
            html+="</div>"

            yield html
            time.sleep(2)

    def run(self):
        # =========================
        # BUILD UI
        # =========================
        with gr.Blocks() as demo:
            gr.Markdown("## Portside Condominiums Security Cam Viewer")

            with gr.Accordion("Controls", open=True):
                with gr.Row():
                    with gr.Column(scale=1):
                        confidence_threshold_slider = gr.Slider(label="Confidence",
                                                                minimum=constants.CONFIDENCE_THRESHOLD_MIN,
                                                                maximum=constants.CONFIDENCE_THRESHOLD_MAX,
                                                                value=self.ctx.confidence_threshold,
                                                                step=0.05,
                                                                )
                    with gr.Column(scale=1):
                        day_motion_threshold_slider = gr.Slider(label="Day Motion",
                                                                minimum=constants.MOTION_THRESHOLD_MIN[0],
                                                                maximum=constants.MOTION_THRESHOLD_MAX[0],
                                                                value=self.ctx.motion_threshold[0],
                                                                step=50,)
                    with gr.Column(scale=1):
                        night_motion_threshold_slider = gr.Slider(label="Night Motion",
                                                                minimum=constants.MOTION_THRESHOLD_MIN[1],
                                                                maximum=constants.MOTION_THRESHOLD_MAX[1],
                                                                value=self.ctx.motion_threshold[1],
                                                                step=50,)

                    with gr.Column(scale=4):
                        detection_classes = gr.CheckboxGroup(
                                choices=self.classes,
                                value=self.classes,
                                label="Objects"
                            )

                confidence_threshold_slider.change(self.update_confidence_threshold, confidence_threshold_slider)
                day_motion_threshold_slider.change(self.update_day_motion_threshold, day_motion_threshold_slider)
                night_motion_threshold_slider.change(self.update_night_motion_threshold, night_motion_threshold_slider)
                detection_classes.change(self.update_detection_classes, detection_classes)

            outputs = []
            for i in range(0, int(len(self.nvr.cameras)/5)):
                with gr.Row():
                    d = dict(list(self.nvr.cameras.items())[i*5:5+i*5])
                    for camera in d.values():
                        if camera.enabled:
                            with gr.Column():
                                annotated = gr.Image(label=f"{camera.name}", streaming=True)
                                stats_box = gr.Textbox(
                                    label=f"{camera.name} Stats",
                                    show_label=False,
                                    interactive=False,
                                    elem_classes="mono-textbox"
                                )
                                # Add HD Mode toggle button
                                hd_toggle = gr.Checkbox(label="HD Mode", value=False)
                                hd_toggle.change(fn=self.update_hd_mode, inputs=[gr.State(value=camera.name), hd_toggle],  outputs=[])
                                outputs.append((annotated, stats_box, camera))

            # recordings HTML
            recordings_box = gr.HTML(label="All Recordings")

            # Event log HTML
            log_box = gr.HTML(label="Event Log", value=LOG_STREAM_DIV+"</div>", elem_classes="scrollable-log")

            # --- LAUNCH STREAMS ---
            # Image streams
            for annotated, stats, camera in outputs:
                demo.load(fn=self.make_stream_fn(camera), inputs=None, outputs=[annotated, stats])
            # Recordings stream
            demo.load(fn=self.recordings_stream, inputs=None, outputs=recordings_box)
            # Event log stream
            demo.load(fn=self.log_stream, inputs=None, outputs=log_box)

        demo.launch(
            #server_name="0.0.0.0",
            theme=gr.themes.Soft(),
            allowed_paths=[self.ctx.directory],
            css="""
    .mono-textbox textarea {
        font-family: "Courier New", monospace !important;
        font-size: x-small !important;
    }
    """,
            )
        