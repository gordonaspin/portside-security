import colorsys
from datetime import datetime
import json
from logging import getLogger
import math
import time
from pathlib import Path

from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import FastAPI, Request
import gradio as gr
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
import uvicorn


from camera import Camera
import constants
from context import Context
from logger import log_event, event_log
from nvr import NVR
from webrtc import CameraTrack, MosaicTrack

logger = getLogger("nvr")

class GUI:
    def __init__(self, ctx: Context, nvr: NVR):
        self._ctx = ctx
        self._classes = ctx.classes
        self._nvr = nvr
        self._color_map = {}
        file = font_manager.findfont('Verdana', fontext='ttf')
        self._courier_font = ImageFont.truetype(file, 14)
        self._image_width = 920
        self._row_height = 40
        self._padding = 10
        self._scale_height = 30
        self._legend_height = 50
        self._label_width = 90

        for i, s in enumerate(self._classes):
            hue = i / len(self._classes)  # evenly spaced
            r, g, b = colorsys.hsv_to_rgb(h=hue, s=0.65, v=0.95)

            self._color_map[s] = "#{:02x}{:02x}{:02x}".format(
                int(r * 255), int(g * 255), int(b * 255)
            )

        self._js = f"""
async function start() {{
    const cameraMap = {json.dumps(self._get_cameras_for_js())};
    let timelineZoom = {constants.GUI_TIMELINE_ZOOM};
    let timelineOffset = {constants.GUI_TIMELINE_OFFSET};
    const minZoom = {constants.GUI_TIMELINE_MIN_ZOOM};
    const maxZoom = {constants.GUI_TIMELINE_MAX_ZOOM};

    // ----------------------------
    // --- Helper functions
    // ----------------------------
    function deepQuery(selector) {{
        const results = [];
        const stack = [document];

        while (stack.length) {{
            const node = stack.pop();

            // Query this node
            try {{
                const found = node.querySelectorAll(selector);
                if (found.length) results.push(...found);
            }} catch {{}}

            // Add shadow roots to search stack
            if (node.querySelectorAll) {{
                node.querySelectorAll("*").forEach(el => {{
                    if (el.shadowRoot) stack.push(el.shadowRoot);
                }});
            }}
        }}

        return results;
    }}

    function sendToPython(data) {{
        const textbox = deepQuery('#timeline_scroll_json textarea, #timeline_scroll_json input')[0];

        if (!textbox) {{
            console.warn("timeline_scroll_text not found yet");
            return;
        }}

        textbox.value = JSON.stringify(data);
        textbox.dispatchEvent(new Event("input", {{ bubbles: true }}));

        console.log("Dispatched to Gradio (Textbox):", data);
    }}

    function sendUpdateThrottled(data) {{
        const now = performance.now();

        if (now - lastSend >= throttleDelay) {{
            // Send immediately
            lastSend = now;
            sendToPython(data);
        }} else {{
            // Save the latest event to send after delay
            pendingData = data;

            setTimeout(() => {{
                if (pendingData) {{
                    sendToPython(pendingData);
                    pendingData = null;
                    lastSend = performance.now();
                }}
            }}, throttleDelay - (now - lastSend));
        }}
    }}

    function waitForElement(name) {{
        return new Promise(resolve => {{
            const check = () => {{
                const els = deepQuery(name);
                if (els.length > 0) {{
                    console.log("Found ", name);
                    resolve(els[0]);
                }} else {{
                    console.log("Waiting for ", name, " ...")
                    requestAnimationFrame(check);
                }}
            }};
            check();
        }});
    }}
    // --------------------------------------------
    // Video / WebRTC management functions
    // --------------------------------------------
    function waitForMosaic() {{
        return new Promise(resolve => {{
            const check = () => {{
                const mosaic = document.querySelector("video[id^='mosaic']");
                if (mosaic) {{
                    resolve(mosaic);
                }} else {{
                    requestAnimationFrame(check);
                }}
            }};
            check();
        }});
    }}

    async function startMosaic() {{
        mosaicPC = new RTCPeerConnection();

        mosaicPC.addTransceiver("video", {{ direction: "recvonly" }});

        mosaicPC.ontrack = (event) => {{
            const el = document.querySelector("#mosaic");
            if (el) el.srcObject = event.streams[0];
        }};

        const offer = await mosaicPC.createOffer();
        await mosaicPC.setLocalDescription(offer);

        const res = await fetch("/signal", {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify({{
                mode: "mosaic",
                sdp: offer.sdp,
                type: offer.type,
            }}),
        }});

        const answer = await res.json();
        await mosaicPC.setRemoteDescription(answer);
    }}

    async function startCamera(id) {{
        focusPC = new RTCPeerConnection();
        focusPC.addTransceiver("video", {{ direction: "recvonly" }});

        focusPC.ontrack = (event) => {{
            const el = document.querySelector("#mosaic");
            el.srcObject = event.streams[0];
            el.muted = true;
            el.play();
        }};

        const offer = await focusPC.createOffer();
        await focusPC.setLocalDescription(offer);

        const res = await fetch("/signal", {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify({{
            mode: "single",
            cameraId: id,
            sdp: offer.sdp,
            type: offer.type
            }})
        }});

        const answer = await res.json();
        await focusPC.setRemoteDescription(answer);
    }}

    async function enterMosaicMode() {{
        if (focusPC) {{
            focusPC.close();
            focusPC = null;
        }}
        await startMosaic();
    }}

    async function enterFocusMode(id) {{
        console.log("Entering focus mode for camera:", id);
        if (mosaicPC) {{
            mosaicPC.close();
            mosaicPC = null;
        }}
        await startCamera(id);
    }}

    // ------------------------------------------------------------
    // Bridge timeline scroll/zoom events → Gradio Textbox
    // ------------------------------------------------------------
    window.addEventListener("timeline-update", (e) => {{
        const data = e.detail;

        // Throttled dispatch
        sendUpdateThrottled(data);
    }});

    // Video state
    let focusPC = null;
    let mosaicPC = null;

    // Timeline Pan + Zoom State
    let isDragging = false;
    let dragStartX = 0;
    // snapshot of timeline state at drag start
    let dragStartZoom = 0;
    let dragStartOffset = 0;
        
    let redrawTimeout = null;
    const redrawDelay = 150;   // ms after user stops interacting

    let lastSend = 0;
    let pendingData = null;
    const throttleDelay = 500;  // ms

    const app = await waitForElement("gradio-app");
    const root = app.shadowRoot || document;
    console.log("Shadow root found:", !!app.shadowRoot);

    const mosaic = await waitForElement("video[id^='mosaic']");

    // clicking the video switches to the camera clicked if we are
    // in mosaic, or returns to mosaic if we are single camera mode
    mosaic.addEventListener("click", (event) => {{
        if (focusPC === null) {{
            const rect = mosaic.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;

            const tileWidth = rect.width / Math.min(5, cameraMap.length);
            const tileHeight = rect.height / Math.max(2, Math.trunc(cameraMap.length / 5));

            const col = Math.floor(x / tileWidth);
            const row = Math.floor(y / tileHeight);

            const index = row * 5 + col;
            if (index < cameraMap.length) {{
                const cameraId = cameraMap[index];
                console.log("Clicked camera: ", cameraId, "row:", row, "col: ", col, "index: ", index);
                enterFocusMode(cameraId);
            }}
        }} else {{
            enterMosaicMode()
        }}
    }});

    // ------------------------------------------------------------
    // TIMELINE SCROLL + ZOOM CONTROL
    // ------------------------------------------------------------

    // immediate redraw
    function redrawNow() {{
        window.dispatchEvent(
            new CustomEvent("timeline-update", {{
                detail: {{
                    zoom: timelineZoom,
                    offset: timelineOffset
                }}
            }})
        );
    }}

    // throttled redraw fallback (optional)
    function scheduleRedraw() {{
        clearTimeout(redrawTimeout);

        redrawTimeout = setTimeout(() => {{
            redrawNow();
        }}, redrawDelay);
    }}

    const timelineContainer = await waitForElement("div#timeline_bars_img");
    const timelineScrollJson = await waitForElement("#timeline_scroll_json textarea, #timeline_scroll_json input");
    const timeline_viewport = await waitForElement("#timeline_row");
    const timeline_bars_img = await waitForElement("#timeline_bars_img");
    const img = document.querySelector("#timeline_bars_img img");
    console.log("img:", img)
    timeline_bars_img.draggable = false;

    timeline_bars_img.addEventListener("dragstart", (e) => {{
        e.preventDefault();
    }});

    timelineContainer.addEventListener("mousedown", (e) => {{
        if (!e.shiftKey) return;
    
        timelineContainer.classList.add("dragging");

        isDragging = true;
        dragStartX = e.clientX;

        // freeze timeline state at start
        dragStartZoom = timelineZoom;
        dragStartOffset = timelineOffset;
    }});

    timelineContainer.addEventListener("mousemove", (e) => {{
        if (!isDragging) return;

        const dx = e.clientX - dragStartX;
        const hoursPerPixel = dragStartZoom / timeline_viewport.clientWidth;

        // base everything off snapshot, not current state
        // drag right = move to past
        // drag left  = move to future
        timelineOffset = dragStartOffset + (dx * hoursPerPixel);

        redrawNow();
    }});

    document.addEventListener("keydown", (e) => {{
        if (!e.shiftKey) return;

        timelineContainer.classList.add("zoom");
        }});

    document.addEventListener("keyup", (e) => {{
        timelineContainer.classList.remove("zoom");
    }});

    document.addEventListener("mouseup", async () => {{
        isDragging = false;
        timelineContainer.classList.remove("dragging");
    }});

    timeline_viewport.addEventListener("mouseenter", () => {{
        document.body.style.overflow = "hidden";
    }});

    timeline_viewport.addEventListener("mouseleave", () => {{
        document.body.style.overflow = "";
    }});

    // Attach scroll handler
    timelineContainer.addEventListener("wheel", async (e) => {{
        if (!timeline_viewport.contains(e.target)) return;
        if (!e.shiftKey) return;

        // only vertical wheel controls zoom
        if (Math.abs(e.deltaY) <= Math.abs(e.deltaX)) return;

        e.preventDefault();

        const zoomFactor = 1.05;

        // zoom around cursor position
        const rect = timeline_viewport.getBoundingClientRect();
        const cursorX = e.clientX - rect.left;
        const cursorRatio = cursorX / rect.width;

        // timeline time under cursor BEFORE zoom
        const cursorTimeBefore =
            timelineOffset + (cursorRatio * timelineZoom);

        // apply zoom
        if (e.deltaY < 0) {{
            timelineZoom *= zoomFactor;
        }} else {{
            timelineZoom /= zoomFactor;
        }}

        timelineZoom =
            Math.max(minZoom, Math.min(maxZoom, timelineZoom));

        // preserve cursor anchor point
        timelineOffset =
            cursorTimeBefore - (cursorRatio * timelineZoom);

        // prevent future scrolling
        timelineOffset = Math.max(0, timelineOffset);

        // immediate redraw
        redrawNow();

    }}, {{ passive: false }});

    const fixCursor = () => {{
        const el = document.querySelector("#timeline_bars_img");
        if (!el) return;

        el.style.cursor = "default";
        el.querySelectorAll("*").forEach(n => n.style.cursor = "default");
    }};
    setInterval(fixCursor, 500);    

    if (mosaic) startMosaic();


}}
console.log("loaded javascript")
start();
"""
    def get_status(self, camera: Camera):
        return camera.status_text
    
    def _get_cameras_for_js(self):
        return [key for key in self._nvr.cameras.keys() if self._nvr.cameras[key].enabled]

    # UI HANDLERS
    def update_confidence_threshold(self, val):
        """ modifies the object detection confidence threshold of the NVR YOLO model """
        self._nvr.update_yolo_confidence_threshold(val)
        log_event(message=f"confidence updated → {val}")

    def update_motion_threshold(self, val):
        """ modifies the motion detection threshold of the NVR """
        self._nvr.update_motion_threshold(val)
        log_event(message=f"motion threshold → {val}")

    def update_detection_classes(self, names):
        """ updates the set of object class indexes to detect in the NVR """
        self._nvr.selected_classes = self._nvr.model.class_to_index(names)
        log_event(message=f"classes → {names}")

    #def update_hd(self, name, val):
    #    """ updates the HD option for viewing the camera image """
    #    self._nvr.cameras[name].hd = val
    #    log_event(message=f"HD mode {'on' if val else 'off'}", camera=self._nvr.cameras[name])

    def update_camera_debug(self, name, val):
        """ updates the Debug option for the camera """
        self._nvr.cameras[name].debug = val
        log_event(message=f"Debug mode {'on' if val else 'off'}", camera=self._nvr.cameras[name])

    def update_debug(self, val):
        """" updates the debug option of the NVR """
        self._nvr.debug = val
        log_event(message=f"Verbose logging {'on' if val else 'off'}")

    def update_debug_files(self, val):
        """ updates the debug_files option of the NVR """
        self._nvr.debug_files = val
        log_event(message=f"Debug files {'on' if val else 'off'}")

    # UI STREAMS
    def get_log_html(self):
        """ writes the HTML for the log view """
        content = "".join(x for x in event_log[-constants.MAX_LOG_LINES:])

        return f"""
        <div style="
            height:300px;
            overflow-y:auto;
            border:1px solid #ccc;
            padding:5px;
            font-family:monospace;
            font-size:xsmall;
            background-color:#1e1e1e;
            color:#ffffff;
        ">
            <div style="font-weight:bold; margin-bottom:8px;">
                📜 Event Log
            </div>
            {content}
        </div>
        """

    def get_height(self):
        """ computes the height of the timeline image based on the number of cameras """
        height = self._scale_height + len(self._nvr.cameras) * self._row_height + self._legend_height + self._padding * 2
        logger.debug(f"get_height: {height}")
        return height

    def _draw_full_timeline(self, window):
        """
        Draw the timeline image for all cameras.

        window = {
            "zoom": <float hours>,
            "offset": <float hours>
        }
        """
        now = time.time()
        zoom_hours = float(window.get("zoom", constants.GUI_TIMELINE_ZOOM))
        offset_hours = float(window.get("offset", constants.GUI_TIMELINE_OFFSET))
        offset_hours = max(0.0, offset_hours)

        # ------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------
        def tag_colors(tags):
            colors = []
            if isinstance(tags, dict):
                for obj, _ in tags.items():
                    colors.append(self._color_map.get(obj, "#9E9E9E"))
            else:
                for tag in tags:
                    colors.append(self._color_map.get(tag[0], "#9E9E9E"))
            return colors

        def tag_label(tags):
            objects = [f"{obj}({color})" for obj, color in tags]
            return ", ".join(objects) if objects else "motion"

        now = datetime.now().timestamp()
        # ------------------------------------------------------------
        # Load events
        # ------------------------------------------------------------
        grouped_events = self._nvr.recordings
        logger.debug(f"loaded events {time.time() - now}")

        # ------------------------------------------------------------
        # ⭐ NEW WINDOW LOGIC (scroll + zoom)
        # ------------------------------------------------------------
        # end = right edge of timeline
        # start = left edge of timeline
        end = now - offset_hours * 3600
        start = end - zoom_hours * 3600
        span = end - start

        # ------------------------------------------------------------
        # Filter events to visible window
        # ------------------------------------------------------------
        filtered = {}

        for cam, events in grouped_events.items():
            visible = [
                e for e in events
                if e["end_time"] >= start and e["start_time"] <= end
            ]

            if visible:
                filtered[str(cam)] = visible

                if not filtered:
                    img = Image.new("RGB", (self._image_width, self.get_height()), (31, 41, 55))
                    return img, []

        grouped_events = filtered

        # ------------------------------------------------------------
        # Layout
        # ------------------------------------------------------------
        width = self._image_width
        label_width = self._label_width
        height = self.get_height()

        img = Image.new("RGB", (width, height), (31, 41, 55))
        draw = ImageDraw.Draw(img)

        clickable_regions = []

        scale_top = self._padding
        scale_bottom = scale_top + self._scale_height - 5

        # Background for time scale
        draw.rectangle([label_width, scale_top, width - 10, scale_bottom], fill="#2d3748")

        # ------------------------------------------------------------
        # Adaptive tick logic (density based on zoom level)
        # ------------------------------------------------------------

        timeline_pixels = width - label_width - 20

        seconds_per_pixel = span / timeline_pixels

        # aim for ~80–120 px between ticks
        target_px = 100
        tick_seconds = seconds_per_pixel * target_px

        # snap to "nice" intervals
        if tick_seconds <= 60:
            tick_seconds = 30          # 30 sec
        elif tick_seconds <= 120:
            tick_seconds = 60          # 1 min
        elif tick_seconds <= 300:
            tick_seconds = 300         # 5 min
        elif tick_seconds <= 900:
            tick_seconds = 900         # 15 min
        elif tick_seconds <= 1800:
            tick_seconds = 1800        # 30 min
        elif tick_seconds <= 3600:
            tick_seconds = 3600        # 1 hour
        elif tick_seconds <= 7200:
            tick_seconds = 7200        # 2 hour
        else:
            tick_seconds = 14400       # 4 hour

        # align first tick to grid
        first_tick = math.ceil(start / tick_seconds) * tick_seconds

        t = first_tick
        while t <= end:
            x = label_width + int((t - start) / span * timeline_pixels)

            draw.line([(x, scale_top), (x, scale_bottom)], fill="#ffffff", width=1)

            label = datetime.fromtimestamp(t).strftime("%H:%M")
            draw.text((x + 2, scale_top + 2), label, fill="white")

            t += tick_seconds

        # ------------------------------------------------------------
        # Draw camera rows
        # ------------------------------------------------------------
        for idx, (camera_name, camera) in enumerate(
            sorted(self._nvr.cameras.items(), key=lambda c: c[1].name)
        ):
            if camera.enabled:
                y_top = scale_bottom + self._padding + idx * self._row_height
                y_bottom = y_top + self._row_height - 5

                draw.text((10, y_top + 10), camera.name, font=self._courier_font, fill="white")

                draw.rectangle(
                    [label_width, y_top + 5, width - 10, y_bottom],
                    fill="#374151"
                )

        # ------------------------------------------------------------
        # Draw events
        # ------------------------------------------------------------
        timeline_width = width - label_width - 20

        for idx, (camera_name, camera) in enumerate(
            sorted(self._nvr.cameras.items(), key=lambda c: c[1].name)
        ):
            if camera.enabled:
                y_top = scale_bottom + self._padding + idx * self._row_height
                y_bottom = y_top + self._row_height - 5

                for e in grouped_events.get(camera.name, []):

                    # ----------------------------------------------------
                    # Convert event times -> bars_img coordinate space
                    # ----------------------------------------------------
                    left = int(
                        (e["start_time"] - start) / span * timeline_width
                    )

                    right = int(
                        (e["end_time"] - start) / span * timeline_width
                    )

                    # Clamp to visible viewport
                    left = max(0, min(timeline_width, left))
                    right = max(0, min(timeline_width, right))

                    # Skip degenerate regions
                    if right <= left:
                        continue

                    # ----------------------------------------------------
                    # Convert to full-image coordinates for drawing
                    # ----------------------------------------------------
                    draw_left = label_width + left
                    draw_right = label_width + right

                    # ----------------------------------------------------
                    # Draw event bars
                    # ----------------------------------------------------
                    colors = tag_colors(e["tags"])

                    for i, color in enumerate(colors):
                        draw.rectangle(
                            [
                                draw_left,
                                y_top + 5 + i * (y_bottom - y_top - 5) // len(colors),
                                draw_right,
                                y_bottom
                            ],
                            fill=color
                        )

                    # ----------------------------------------------------
                    # Tooltip / metadata HTML
                    # ----------------------------------------------------
                    metadata_str = (
                        f'<a href="/gradio_api/file={e["metadata"]}" '
                        f'target="_blank">View</a>'
                        if e.get("metadata")
                        else "N/A"
                    )

                    info_html = f"""
                    <b>Camera:</b> {camera.name} |
                    <b>Tags:</b> {
                        self._nvr._tags_to_str(e["tags"])
                        if isinstance(e["tags"], dict)
                        else tag_label(e["tags"])
                    } |
                    <b>Start:</b>
                    {datetime.fromtimestamp(e["start_time"]).strftime("%Y-%m-%d %H:%M:%S")}
                    -
                    <b>End:</b>
                    {datetime.fromtimestamp(e["end_time"]).strftime("%Y-%m-%d %H:%M:%S")}
                    <br>
                    <b>Metadata:</b> {metadata_str}
                    """

                    # ----------------------------------------------------
                    # IMPORTANT:
                    # Store clickable regions in bars_img coordinates,
                    # NOT full-image coordinates
                    # ----------------------------------------------------
                    clickable_regions.append(
                        (
                            left,
                            y_top + 5,
                            right,
                            y_bottom,
                            e["output"],
                            info_html
                        )
                    )
        # ------------------------------------------------------------
        # Legend
        # ------------------------------------------------------------
        for index, (cls, color) in enumerate(self._color_map.items()):
            draw.text((label_width + index * 80, y_bottom + 20),
                    cls, font=self._courier_font, fill=color)

        logger.debug(f"updated timeline {time.time() - now}")
        return img, clickable_regions

    def draw_timeline(self, window):
        full_img, regions = self._draw_full_timeline(window)

        W, H = full_img.size
        logger.debug(f"full image size: {W} x {H}")

        labels_width = self._label_width         # must match CSS
        legend_height = self._legend_height      # adjust to your design

        # 1. Camera labels (left side)
        labels_img = full_img.crop((0, 0, labels_width, H - legend_height))

        # 2. Timeline bars + ticks (right side)
        bars_img = full_img.crop((labels_width, 0, W, H - legend_height))

        # 3. Legend (bottom)
        legend_img = full_img.crop((0, H - legend_height, W, H))

        return labels_img, bars_img, legend_img, regions

    def handle_click(self, evt: gr.SelectData, regions):
        x, y = evt.index

        for (x1, y1, x2, y2, video, info_html) in regions:
            if x1 <= x <= x2 and y1 <= y <= y2:
                log_event(message=f"user selected {video}", level="info", file_path=video)
                return video, info_html

        return None, "No video selected"

    def on_load(self):
        """ called when the GUI loads for a client """
        log_event(f"A browser has connected")

    def build_blocks(self) -> gr.Blocks:
        # BUILD UI
        with gr.Blocks() as demo:
            gr.Markdown("## Portside Condominiums Security Cam Viewer")
            # Controls
            with gr.Accordion("Controls", open=False):
                with gr.Row():
                    with gr.Column(scale=1):
                        confidence_threshold_slider = gr.Slider(
                            label="Detection Confidence",
                            minimum=constants.CONFIDENCE_THRESHOLD_MIN,
                            maximum=constants.CONFIDENCE_THRESHOLD_MAX,
                            value=self._ctx.confidence_threshold,
                            step=constants.CONFIDENCE_THRESHOLD_STEP,
                        )
                    with gr.Column(scale=1):
                        motion_threshold_slider = gr.Slider(
                            label="% Pixel Change in Motion",
                            minimum=constants.MOTION_THRESHOLD_MIN,
                            maximum=constants.MOTION_THRESHOLD_MAX,
                            value=self._ctx.motion_threshold,
                            step=constants.MOTION_THRESHOLD_STEP,
                        )
                    with gr.Column(scale=4):
                        detection_classes = gr.CheckboxGroup(
                            label="Objects",
                            choices=self._classes,
                            value=self._classes,
                        )
                with gr.Row():
                    for camera in self._nvr.cameras.values():
                        if camera.enabled:
                            camera_debug_checkbox = gr.Checkbox(
                                label=f"Debug {camera.name}",
                                value=camera.debug,
                                elem_classes="custom-checkbox"
                            )
                            camera_debug_checkbox.change(
                                fn=self.update_camera_debug,
                                inputs=[gr.State(value=camera.name),
                                        camera_debug_checkbox],
                                        outputs=[]
                            )
                with gr.Row():
                    files_checkbox = gr.Checkbox(
                        label="Produce Debug Images",
                        value=self._ctx.debug_files,
                        elem_classes="custom-checkbox"
                    )
                debug_checkbox = gr.Checkbox(label="Verbose Logging", value=self._ctx.debug, elem_classes="custom-checkbox")
                confidence_threshold_slider.change(self.update_confidence_threshold, confidence_threshold_slider)
                motion_threshold_slider.change(self.update_motion_threshold, motion_threshold_slider)
                detection_classes.change(self.update_detection_classes, detection_classes)
                debug_checkbox.change(fn=self.update_debug, inputs=debug_checkbox,  outputs=[])
                files_checkbox.change(fn=self.update_debug_files, inputs=files_checkbox,  outputs=[])

            # Mosaic
            gr.HTML(
                """
                <div style="text-align:center; width:100%;">
                <video id="mosaic" autoplay playsinline muted
                style="width:100%; border:2px solid #888; background:black;"></video>
                </div>
                """,
                container=False,
                sanitize_html=False
                )

            # recording timeline and playback
            with gr.Row(elem_id="main_row"):
                selected_video = gr.Textbox(visible=False)
                with gr.Column(variant="compact"):
                    with gr.Group(elem_classes="timeline_labels_img"):
                        w = 95
                        h = 20
                        with gr.Row(height=self.get_height()-self._legend_height, elem_id="timeline_row"):
                            labels_img = gr.Image(
                                elem_id="timeline_labels_img",
                                elem_classes="timeline_labels_img",
                                min_width=self._label_width,
                                width=self._label_width,
                                height=self.get_height()-self._legend_height,
                                show_label=False,
                                interactive=False,
                                buttons=[],
                                container=False
                            )
                            bars_img = gr.Image(
                                elem_id="timeline_bars_img",
                                elem_classes="timeline_bars_img",
                                min_width=self._image_width-w,
                                width=self._image_width-w  ,
                                height=self.get_height()-self._legend_height,
                                show_label=False,
                                interactive=False,
                                container=False,
                                buttons=[],
                            )
                        legend_img = gr.Image(
                            elem_id="timeline_legend_img",
                            elem_classes="timeline_legend_img",
                            width=self._image_width,
                            min_width=self._image_width,
                            height=self._legend_height,
                            show_label=False,
                            interactive=False,
                            buttons=[],
                            container=False
                        )
                with gr.Column():
                    video_player = gr.Video(label="Selected Recording", height=self.get_height(), autoplay=True, interactive=False)
                    event_info = gr.HTML(label="Event Info")

            # Event log HTML
            with gr.Row():
                log_box = gr.HTML()
            with gr.Row():
            #    timeline_scroll_json = gr.JSON(elem_id="timeline_scroll_json", height=0, elem_classes="hidden-json")
                timeline_scroll_json = gr.Textbox(elem_id="timeline_scroll_json", lines=1, container=False, scale=0, min_width=0)

            # UI State

            # Store clickable regions in a State object
            regions_state = gr.State([])

            # timeline window state
            timeline_window_state = gr.State({"zoom": constants.GUI_TIMELINE_ZOOM, "offset": constants.GUI_TIMELINE_OFFSET})

            # on change handlers
            # When selected_video changes, update video player
            selected_video.change(lambda x: x, selected_video, video_player)

            def on_scroll(window: str):
                window = json.loads(window)
                labels, bars, legend, regions = self.draw_timeline(window)
                return window, bars, regions

            # Timer updates the timeline
            def initial_render(window):
                labels, bars, legend, regions = self.draw_timeline(window)
                return labels, bars, legend, regions

            #update state when JS sends scroll/zoom
            timeline_scroll_json.change(
                fn=on_scroll,
                inputs=[timeline_scroll_json],
                outputs=[timeline_window_state, bars_img, regions_state]
            )
            # Clicking the image selects a video
            bars_img.select(
                fn=self.handle_click,
                inputs=[regions_state],
                outputs=[selected_video, event_info]
            )

            log_timer = gr.Timer(1.0)
            log_timer.tick(fn=self.get_log_html, outputs=log_box)

            demo.load(
                fn=initial_render,
                inputs=[timeline_window_state],
                outputs=[labels_img, bars_img, legend_img, regions_state]
            )
        return demo

    def run(self):
        app = FastAPI()
        demo = self.build_blocks()

        @app.post("/signal")
        async def signal(request: Request):
            """
            WebRTC signaling endpoint.

            Expected JSON:
            {
              "mode": "single" | "mosaic",
              "cameraId": "camera.name",   # for single
              "sdp": "...",
              "type": "offer"
            }
            """
            data = await request.json()
            logger.info(f"Received WebRTC signal: mode={data.get('mode')} cameraId={data.get('cameraId')}")

            mode = data.get("mode", "single")
            camera_id = data.get("cameraId")
            sdp = data["sdp"]
            sdp_type = data["type"]

            offer = RTCSessionDescription(sdp=sdp, type=sdp_type)
            pc = RTCPeerConnection()

            if mode == "mosaic":
                cams = [c for c in self._nvr.cameras.values() if getattr(c, "enabled", True)]
                track = MosaicTrack(cams)
            else:
                # here we assume cameraId == camera.name; adjust if you use IDs
                cam = next(c for c in self._nvr.cameras.values() if c.name == camera_id)
                track = CameraTrack(cam)

            pc.addTrack(track)

            await pc.setRemoteDescription(offer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            return {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
            }

        # mount Gradio at root
        app = gr.mount_gradio_app(
            app,
            demo,
            js=self._js,
            path="/",
            auth=[self._ctx.gui_username, self._ctx.gui_password] if all([self._ctx.gui_username, self._ctx.gui_password]) else None,
            #server_name=self._ctx.bind_address,
            theme=gr.themes.Soft(),
            allowed_paths=[self._ctx.directory],
            css="""
                .mono-textbox textarea {
                    font-family: "Courier New", monospace !important;
                    font-size: x-small !important;
                }
                .custom-checkbox span {
                font-family: 'Courier New', monospace !important;
                    font-size: small !important;
                }
                .gradio-container > footer,
                .gradio-container footer,
                footer,
                div:has(> .footer) {
                    display: none !important;
                }
                .no-scale {object-fit: none !important; height: auto !important;}
                """,
        )

        uvicorn.run(app, host=self._ctx.bind_address, port=7860, log_level="warning")
    
        try:
            demo.launch(
                #share=True,
                auth=[self._ctx.gui_username, self._ctx.gui_password] if all([self._ctx.gui_username, self._ctx.gui_password]) else None,
                server_name=self._ctx.bind_address,
                theme=gr.themes.Soft(),
                allowed_paths=[self._ctx.directory],
                css="""
                    .mono-textbox textarea {
                        font-family: "Courier New", monospace !important;
                        font-size: x-small !important;
                    }
                    .custom-checkbox span {
                    font-family: 'Courier New', monospace !important;
                        font-size: small !important;
                    }
                    .gradio-container > footer,
                    .gradio-container footer,
                    footer,
                    div:has(> .footer) {
                        display: none !important;
                    }
                    #timeline_scroll_json {
                        display: none !important;
                    }
                    /* =====================================================
                    HARD RESET CURSOR FOR GRADIO IMAGE COMPONENT
                    ===================================================== */

                    #timeline_bars_img,
                    #timeline_bars_img *,
                    #timeline_bars_img *::before,
                    #timeline_bars_img *::after {
                        cursor: default !important;
                    }

                    /* zoom state */
                    #timeline_bars_img.zoom,
                    #timeline_bars_img.zoom *,
                    #timeline_bars_img.zoom *::before,
                    #timeline_bars_img.zoom *::after {
                        cursor: zoom-in !important;
                    }

                    /* dragging state */
                    #timeline_bars_img.dragging,
                    #timeline_bars_img.dragging *,
                    #timeline_bars_img.dragging *::before,
                    #timeline_bars_img.dragging *::after {
                        cursor: grabbing !important;
                    }
                    """,
                )
        except KeyboardInterrupt as e:
            logger.info("Shutting down on CTRL-C")
            demo.close()
