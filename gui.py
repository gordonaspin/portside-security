import colorsys
from datetime import datetime
import json
from logging import getLogger
import math

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
        self._row_height = 40
        self._padding = 10
        self._scale_height = 30
        self._legend_height = 50

        for i, s in enumerate(self._classes):
            hue = i / len(self._classes)  # evenly spaced
            r, g, b = colorsys.hsv_to_rgb(h=hue, s=0.65, v=0.95)

            self._color_map[s] = "#{:02x}{:02x}{:02x}".format(
                int(r * 255), int(g * 255), int(b * 255)
            )

        self._js = f"""
console.log("loaded javascript")
let mosaicPC = null;
let focusPC = null;
const cameraMap = {json.dumps(self._get_cameras_for_js())};

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

// ------------------------------------------------------------
// Bridge timeline scroll/zoom events → Gradio JSON component
// ------------------------------------------------------------
window.addEventListener("timeline-update", (e) => {{
    const data = e.detail;

    const textbox = deepQuery('#timeline_scroll_json textarea, #timeline_scroll_json input')[0];

    if (!textbox) {{
        console.warn("timeline_scroll_text not found yet");
        return;
    }}

    textbox.value = JSON.stringify(data);

    textbox.dispatchEvent(new Event("input", {{ bubbles: true }}));

    console.log("Dispatched to Gradio (Textbox):", data);
}});

async function startMosaic() {{
    const pc = new RTCPeerConnection();

    // Same here: recvonly video
    pc.addTransceiver("video", {{ direction: "recvonly" }});

    pc.ontrack = (event) => {{
        const el = document.querySelector("#mosaic");
        if (el) el.srcObject = event.streams[0];
    }};

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

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
    await pc.setRemoteDescription(answer);
}}

// Helper functions
function stopMosaic() {{
    const m = document.querySelector("#mosaic");
    if (m && m.srcObject) {{
        m.srcObject.getTracks().forEach(t => t.stop());
        m.srcObject = null;
    }}
}}

function showMosaic() {{
    const m = document.querySelector("#mosaic");
    m.style.display = "block";
    startMosaic();
}}

function hideMosaic() {{
    const m = document.querySelector("#mosaic");
    m.style.display = "none";
    stopMosaic();
}}

function showFocus() {{
    const f = document.querySelector("#focus");
    f.style.display = "block";
}}

function hideFocus() {{
    const f = document.querySelector("#focus");
    f.style.display = "none";
    if (f.srcObject) {{
        f.srcObject.getTracks().forEach(t => t.stop());
        f.srcObject = null;
    }}
}}

async function startFocusedCamera(id) {{
    focusPC = new RTCPeerConnection();
    focusPC.addTransceiver("video", {{ direction: "recvonly" }});

    focusPC.ontrack = (event) => {{
        const el = document.querySelector("#focus");
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

function exitFocusMode() {{
    hideFocus();
    if (focusPC) {{
        focusPC.close();
        focusPC = null;
    }}
    showMosaic();
}}

async function enterFocusMode(id) {{
    console.log("Entering focus mode for camera:", id);
    hideMosaic();
    showFocus();
    await startFocusedCamera(id);
}}

function waitForVideos() {{
    return new Promise(resolve => {{
        const check = () => {{
            const vids = document.querySelectorAll("video[id^='cam_']");
            if (vids.length > 0) {{
                resolve(vids);
            }} else {{
                requestAnimationFrame(check);
            }}
        }};
        check();
    }});
}}

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

async function start() {{
    const app = document.querySelector("gradio-app");
    const root = app.shadowRoot || document;

    console.log("Shadow root found:", !!app.shadowRoot);

    //const vids = await waitForVideos();
    //console.log("FOUND VIDEOS:", vids.length);

    const mosaic = await waitForMosaic();
    console.log("FOUND MOSAIC");

    async function startSingleCamera(id) {{
        const pc = new RTCPeerConnection();

        // Tell the browser we want to RECEIVE video
        pc.addTransceiver("video", {{ direction: "recvonly" }});

        pc.ontrack = (event) => {{
            const el = document.querySelector("#cam_" + id);
            if (!el) return;

            el.srcObject = event.streams[0];
            el.muted = true;

            console.log("SET SRC:", el);

            el.play()
                .then(() => {{
                    console.log("PLAY OK, READY STATE:", el.readyState);
                }})
                .catch((err) => {{
                    console.error("VIDEO PLAY ERROR:", err);
                }});
            }};

        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        const res = await fetch("/signal", {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify({{
                mode: "single",
                cameraId: id,
                sdp: offer.sdp,
                type: offer.type,
            }}),
        }});

        const answer = await res.json();
        await pc.setRemoteDescription(answer);
    }}

    // Clicking the focused video returns to mosaic
    const focusEl = root.querySelector("#focus");
    focusEl.addEventListener("click", exitFocusMode);

    mosaic.addEventListener("click", (event) => {{
        const rect = mosaic.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        const tileWidth = rect.width / 5;
        const tileHeight = rect.height / 2;

        const col = Math.floor(x / tileWidth);
        const row = Math.floor(y / tileHeight);

        const index = row * 5 + col;

        const cameraId = cameraMap[index];

        console.log("Clicked camera:", cameraId);
        enterFocusMode(cameraId);
    }});

    // ------------------------------------------------------------
    // TIMELINE SCROLL + ZOOM CONTROL
    // ------------------------------------------------------------
    console.log("Initializing timeline scroll control...");
    console.log("ROOT:", root);
    console.log("ALL IMAGES:", root.querySelectorAll("img"));

    let timelineZoom = 4.0;      // default zoom window (hours)
    let timelineOffset = 0.0;    // hours from 'now' (0 = rightmost)
    const minZoom = 0.25;
    const maxZoom = 24;

    function waitForTimelineContainer() {{
        return new Promise(resolve => {{
            const check = () => {{
                const imgs = deepQuery("div#timeline");
                if (imgs.length > 0) {{
                    resolve(imgs[0]);
                }} else {{
                    console.log("Checking ...")
                    requestAnimationFrame(check);
                }}
            }};
            check();
        }});
    }}

    function waitForTimelineScrollJson() {{
        return new Promise(resolve => {{
            const check = () => {{
                const els = deepQuery('#timeline_scroll_json textarea, #timeline_scroll_json input');
                if (els.length > 0) {{
                    resolve(els[0]);
                }} else {{
                    console.log("Checking ...")
                    requestAnimationFrame(check);
                }}
            }};
            check();
        }});
    }}

    const timelineContainer = await waitForTimelineContainer();
    console.log("FOUND TIMELINE CONTAINER");

    const timelineScrollJson = await waitForTimelineScrollJson();
    console.log("FOUND TIMELINE SCROLL JSON")

    // Attach scroll handler
    timelineContainer.addEventListener("wheel", (e) => {{
        e.preventDefault();

        // Horizontal scroll → pan timeline
        if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) {{
            timelineOffset += (e.deltaX / 150);   // sensitivity
            timelineOffset = Math.max(0, Math.min(timelineOffset, 24 - timelineZoom));
        }}
        // Vertical scroll → zoom timeline
        else {{
            timelineZoom += (e.deltaY < 0 ? -0.25 : 0.25);
            timelineZoom = Math.max(minZoom, Math.min(timelineZoom, maxZoom));

            // Clamp offset so window stays inside 24h
            timelineOffset = Math.max(0, Math.min(timelineOffset, 24 - timelineZoom));
        }}

        // ⭐ Add this log
        console.log("SCROLL UPDATE:", {{
            zoom: timelineZoom,
            offset: timelineOffset
        }});

        // Send update to Gradio
        const event = new CustomEvent("timeline-update", {{
            detail: {{ zoom: timelineZoom, offset: timelineOffset }}
        }});
        window.dispatchEvent(event);
    }});

    if (mosaic) startMosaic();
}}

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
        return self._scale_height + len(self._nvr.cameras) * self._row_height + self._legend_height + self._padding * 2

    def draw_timeline(self, window):
        """
        Draw the timeline image for all cameras.

        window = {
            "zoom": <float hours>,
            "offset": <float hours>
        }
        """

        if isinstance(window, str):
            window = json.loads(window)

        zoom_hours = float(window.get("zoom", 4.0))
        offset_hours = float(window.get("offset", 0.0))

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

        # ------------------------------------------------------------
        # Load events
        # ------------------------------------------------------------
        grouped_events = self._nvr.load_events()

        now = datetime.now().timestamp()

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
            visible = [e for e in events if e["end_time"] >= start]
            if visible:
                filtered[cam] = visible

        if not filtered:
            img = Image.new("RGB", (900, 200), (31, 41, 55))
            return img, []

        grouped_events = filtered

        # ------------------------------------------------------------
        # Layout
        # ------------------------------------------------------------
        width = 900
        label_width = 100
        height = self.get_height()

        img = Image.new("RGB", (width, height), (31, 41, 55))
        draw = ImageDraw.Draw(img)

        clickable_regions = []

        scale_top = self._padding
        scale_bottom = scale_top + self._scale_height - 5

        # Background for time scale
        draw.rectangle([label_width, scale_top, width - 10, scale_bottom], fill="#2d3748")

        # ------------------------------------------------------------
        # ⭐ NEW TICK LOGIC
        # ------------------------------------------------------------
        if zoom_hours >= 4:
            tick_seconds = 3600      # 1 hour
        else:
            tick_seconds = 900       # 15 minutes

        first_tick = math.ceil(start / tick_seconds) * tick_seconds

        t = first_tick
        while t <= end:
            x = label_width + int((t - start) / span * (width - label_width - 20))

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
        for idx, (camera_name, camera) in enumerate(
            sorted(self._nvr.cameras.items(), key=lambda c: c[1].name)
        ):
            if camera.enabled:
                y_top = scale_bottom + self._padding + idx * self._row_height
                y_bottom = y_top + self._row_height - 5

                for e in grouped_events.get(camera.name, []):
                    left = label_width + max(
                        0,
                        int((e["start_time"] - start) / span * (width - label_width - 20))
                    )
                    right = label_width + int(
                        (e["end_time"] - start) / span * (width - label_width - 20)
                    )

                    colors = tag_colors(e["tags"])
                    for i, color in enumerate(colors):
                        draw.rectangle(
                            [
                                left,
                                y_top + 5 + i * (y_bottom - y_top - 5) // len(colors),
                                right,
                                y_bottom
                            ],
                            fill=color
                        )

                    metadata_str = (
                        f"<a href=\"/gradio_api/file={e['metadata']}\" target=\"_blank\">View</a>"
                        if e.get("metadata") else "N/A"
                    )

                    info_html = f"""
                    <b>Camera:</b> {camera.name} | 
                    <b>Tags:</b> {self._nvr._tags_to_str(e["tags"]) if isinstance(e["tags"], dict) else tag_label(e["tags"])} |
                    <b>Start:</b> {datetime.fromtimestamp(e['start_time']).strftime('%Y-%m-%d %H:%M:%S')} - 
                    <b>End:</b> {datetime.fromtimestamp(e['end_time']).strftime('%Y-%m-%d %H:%M:%S')}<br>
                    <b>Metadata:</b> {metadata_str}
                    """

                    clickable_regions.append(
                        (left, y_top+5, right, y_bottom, e["output"], info_html)
                    )

        # ------------------------------------------------------------
        # Legend
        # ------------------------------------------------------------
        for index, (cls, color) in enumerate(self._color_map.items()):
            draw.text((label_width + index * 80, y_bottom + 20),
                    cls, font=self._courier_font, fill=color)

        return img, clickable_regions



    def handle_click(self, evt: gr.SelectData, regions):
        x, y = evt.index

        for (x1, y1, x2, y2, video, info_html) in regions:
            if x1 <= x <= x2 and y1 <= y <= y2:
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

            # Focus View
            gr.HTML(
                """
                <div style="text-align:center; width:100%; margin-bottom: 12px;">
                <video id="focus" autoplay playsinline muted
                style="width:100%; border:2px solid #888; background:black; display:none;">
                </video>
                </div>
                """,
                sanitize_html=False
                )
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
            with gr.Row():
                selected_video = gr.Textbox(visible=False)
                with gr.Column():
                    timeline_img = gr.Image(type="pil", label="Timeline", elem_id="timeline", interactive=False, buttons=[], container=True)

                with gr.Column():
                    video_player = gr.Video(label="Selected Recording", height=self.get_height(), autoplay=True, interactive=False)
                    event_info = gr.HTML(label="Event Info")

            # Event log HTML
            with gr.Row():
                log_box = gr.HTML()
            with gr.Row():
            #    timeline_scroll_json = gr.JSON(elem_id="timeline_scroll_json", height=0, elem_classes="hidden-json")
                timeline_scroll_json = gr.Textbox(elem_id="timeline_scroll_json", lines=1, container=False, scale=0, min_width=0)

            # When selected_video changes, update video player
            selected_video.change(lambda x: x, selected_video, video_player)

            # Store clickable regions in a State object
            regions_state = gr.State([])

            # timeline window state
            timeline_window_state = gr.State({"zoom": 4.0, "offset": 0.0})

            #update state when JS sends scroll/zoom
            timeline_scroll_json.change(
                fn=lambda data: data,
                inputs=[timeline_scroll_json],
                outputs=[timeline_window_state]
            )
            # Clicking the image selects a video
            timeline_img.select(
                fn=self.handle_click,
                inputs=[regions_state],
                outputs=[selected_video, event_info]
            )

            # Timer updates the timeline every 5 seconds
            def refresh(window):
                img, regions = self.draw_timeline(window)
                return img, regions

            timeline_timer = gr.Timer(0.5)
            timeline_timer.tick(fn=refresh, inputs=[timeline_window_state], outputs=[timeline_img, regions_state])

            # Initial render
            img, regions = refresh(timeline_window_state.value)
            timeline_img.value = img
            regions_state.value = regions

            log_timer = gr.Timer(1.0)  # update every 0.5s
            log_timer.tick(fn=self.get_log_html, outputs=log_box)

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
                    """,
                )
        except KeyboardInterrupt as e:
            logger.info("Shutting down on CTRL-C")
            demo.close()
