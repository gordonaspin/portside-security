import asyncio
import bisect
import json
import secrets
import time
from datetime import datetime, timedelta
from logging import getLogger

from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import FastAPI, Request, Depends, HTTPException, Query, Response, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, EventSourceResponse
from passlib.context import CryptContext
from pydantic import BaseModel

from ..utils import ConfigValue
from ..logger import log_event, event_log
from ..nvr import NVR
from ..webrtc import CameraTrack, MosaicTrack

logger = getLogger("pynvr")

# -------------------------
# AUTH SETUP
# -------------------------

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

SESSIONS = {}  # session_id -> username

    # -------------------------
    # PROTECT STATIC FILES
    # -------------------------

from starlette.staticfiles import StaticFiles
from starlette.responses import Response
from fastapi import HTTPException

from starlette.staticfiles import StaticFiles
from starlette.responses import Response
from fastapi import HTTPException

class AuthStaticFiles(StaticFiles):
    async def __call__(self, scope, receive, send):
        path = scope["path"]

        # Always allow login page and its assets
        if path.startswith("/login"):
            return await super().__call__(scope, receive, send)

        # Allow SvelteKit build assets
        if path.startswith("/_app"):
            return await super().__call__(scope, receive, send)

        # Allow mosaic.js and related scripts
        if path.startswith("/mosaic") or path.endswith(".js"):
            return await super().__call__(scope, receive, send)

        # Allow static assets
        if path.startswith("/static") or path.startswith("/assets"):
            return await super().__call__(scope, receive, send)

        # Check session cookie
        cookie_header = None
        for k, v in scope["headers"]:
            if k == b"cookie":
                cookie_header = v.decode()
                break

        session_id = None
        if cookie_header:
            for part in cookie_header.split(";"):
                if "session_id=" in part:
                    session_id = part.split("=")[1].strip()

        if session_id not in SESSIONS:
            response = Response(status_code=302, headers={"Location": "/login"})
            await response(scope, receive, send)
            return

        return await super().__call__(scope, receive, send)

def require_user(session_id: str | None = Cookie(None)):
    """Require a valid session cookie for all protected routes."""
    if not session_id or session_id not in SESSIONS:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return SESSIONS[session_id]


class LoginForm(BaseModel):
    username: str
    password: str

class SettingValue(BaseModel):
    value: float | bool

class ClassToggle(BaseModel):
    class_name: str
    value: bool
# -------------------------
# APP FACTORY
# -------------------------

def create_app(config: dict, nvr: NVR):
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -------------------------
    # AUTH ROUTES
    # -------------------------
    @app.get("/login")
    async def login_page():
        return FileResponse("pynvr/frontend_dist/index.html")
    
    @app.post("/login")
    async def login(payload: LoginForm, response: Response):
        if payload.username != config["gui_username"] or not pwd.verify(payload.password, config["gui_password"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        session_id = secrets.token_hex(32)
        SESSIONS[session_id] = payload.username

        response.set_cookie(
            "session_id",
            session_id,
            httponly=True,
            secure=False,
            samesite="lax",
            max_age=86400,
        )
        return {"ok": True}

    @app.post("/logout")
    async def logout(response: Response, session_id: str | None = Cookie(None)):
        if session_id in SESSIONS:
            del SESSIONS[session_id]
        response.delete_cookie("session_id")
        return {"ok": True}

    @app.get("/whoami")
    async def whoami(user=Depends(require_user)):
        return {"user": user}

    # -------------------------
    # PROTECTED ROUTES
    # -------------------------

    @app.post("/signal")
    async def signal(request: Request, user=Depends(require_user)):
        data = await request.json()
        mode = data.get("mode", "single")
        name = data.get("name")
        offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
        pc = RTCPeerConnection()

        if mode == "mosaic":
            cameras = [camera for camera in nvr.cameras.values() if camera.config.enabled]
            track = MosaicTrack(cameras, config["mosaic"]["rows"], config["mosaic"]["columns"])
        else:
            camera = nvr.cameras[name]
            track = CameraTrack(camera)

        pc.addTrack(track)
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    @app.get("/api/cameras")
    def get_cameras(user=Depends(require_user)):
        return [{"name": camera.config.name, "debug": camera.config.debug}
                for camera in nvr.cameras.values() if camera.config.enabled]

    @app.get("/api/classes")
    def get_classes(user=Depends(require_user)):
        return {"classes": list(config["model"]["classes"])}

    @app.get("/api/system_name")
    def get_system_name():
        return {"system_name": config["system_name"]}

    @app.get("/api/mosaic_dimensions")
    def get_mosaic_dimensions(user=Depends(require_user)):
        return config["mosaic"]


    @app.get("/api/events")
    def api_events(
        start: float | None = Query(None),
        end: float | None = Query(None),
        mobile: bool = False,
        user = Depends(require_user)
    ):
        """import os

        Returns events for the requested time window.

        Modes:
        • mobile=true → last 4 hours (unchanged)
        • start/end provided → return only events overlapping [start, end]
        • no params → return all events (fallback)
        """

        # nvr.recordings MUST be sorted by start_time ascending
        events = list(nvr.recordings)

        # ------------------------------------------------------------
        # MOBILE MODE: last 4 hours (your existing behavior)
        # ------------------------------------------------------------
        if mobile:
            cutoff = datetime.now().timestamp() - 4 * 3600
            end_times = [ev["end_time"] for ev in events]
            idx = bisect.bisect_left(end_times, cutoff)
            return {"events": events[idx:]}

        # ------------------------------------------------------------
        # DESKTOP MODE: window-based filtering
        # ------------------------------------------------------------
        if start is not None and end is not None:
            # 1. Find first event whose end_time >= start
            end_times = [ev["end_time"] for ev in events]
            left = bisect.bisect_left(end_times, start)

            # 2. Find last event whose start_time <= end
            start_times = [ev["start_time"] for ev in events]
            right = bisect.bisect_right(start_times, end)

            # 3. Slice and return only overlapping events
            window_events = events[left:right]

            # 4. Final overlap check (safety)
            window_events = [
                ev for ev in window_events
                if ev["end_time"] >= start and ev["start_time"] <= end
            ]

            return {"events": window_events}

        # ------------------------------------------------------------
        # FALLBACK: return everything
        # ------------------------------------------------------------
        return {"events": events}


    @app.get("/api/logs")
    def get_logs(
        since: float | None = None,
        user=Depends(require_user)
        ):

        logs = event_log

        if since is not None:
            timestamps = [obj["timestamp"] for obj in event_log]
            index = bisect.bisect_right(timestamps, since)
            logs = event_log[index:]

        return {"logs": logs}

    @app.get("/api/cameras/{camera_name}/settings")
    def get_camera_settings(camera_name: str):
        camera = nvr.cameras[camera_name]

        settings = {}
        for attr, value in vars(camera.config).items():
            if isinstance(value, ConfigValue):
                settings[attr] = vars(value)

        for attr, value in vars(camera.motion).items():
            if isinstance(value, ConfigValue):
                settings[attr] = vars(value)

        settings["classes"] = {}
        for processor in nvr.frame_processors.values():
            if processor.camera.config.name == camera_name:
                settings["classes"] = processor.classes
                break
        return settings

    @app.post("/api/cameras/{camera_name}/settings/{setting}")
    def update_camera_setting(camera_name: str, setting: str, payload: SettingValue):

        camera = nvr.cameras[camera_name]
        if setting == "yolo_confidence":
            attr = getattr(camera.config, setting)
            attr.value = payload.value
        else:
            attr = getattr(camera.motion, setting)
            attr.value = payload.value
            camera.motion.create_tracker()  # Ensure BYTETracker is recreated with new setting

        log_event(f"{setting} {attr.value:.2f} -> {payload.value:.2f}", camera=camera)

        return {
            "status": "ok",
            "camera": camera_name,
            "setting": setting,
            "value": payload.value
        }

    @app.post("/api/processor/{camera_name}/class_toggle")
    def update_class_toggle(camera_name: str, payload: ClassToggle):

        camera = nvr.cameras[camera_name]

        # Update the boolean toggle
        for processor in nvr.frame_processors.values():
            if processor.camera.config.name == camera_name:
                processor.classes[payload.class_name] = payload.value
                processor.set_selected_classes(processor.classes)
                break

        # Recreate tracker if class filters affect tracking
        camera.motion.create_tracker()

        log_event(
            f"class {payload.class_name}: {payload.value}",
            camera=camera
        )

        return {
            "status": "ok",
            "camera": camera_name,
            "class_name": payload.class_name,
            "value": payload.value
        }

    # camera debug
    @app.get("/api/settings/debug/{camera_name}")
    def get_camera_debug(camera_name: str):
        camera = nvr.cameras.get(camera_name)

        return {
            "camera": camera_name,
            "debug": camera.config.debug if camera else False
        }

    @app.post("/api/settings/debug/{camera_name}")
    def set_camera_debug(camera_name: str, payload: SettingValue, user=Depends(require_user)):
        camera = nvr.cameras.get(camera_name)
        log_event(f"debug {payload.value}", camera=camera)
        camera.config.debug = payload.value
        return {"status": "ok", "camera": camera_name, "value": payload.value}
    
    # Verbose debug
    @app.get("/api/settings/debug")
    def get_debug(user=Depends(require_user)):
        return {"value": nvr.debug}
    
    @app.post("/api/settings/debug")
    def set_debug(payload: SettingValue, user=Depends(require_user)):
        log_event(f"verbose logging {payload.value}")
        nvr.debug = payload.value
        return {"status": "ok", "value": payload.value}

    @app.get("/api/server_time")
    def server_time(user=Depends(require_user)):
        return {"epoch": time.time()}
    
    async def event_generator(request: Request):
        last_log_time = time.time()
        last_recording_time = time.time()

        try:
            while True:
            
                if nvr.stop_event.is_set():
                    break;

                if await request.is_disconnected():
                    logger.debug("SSE request is disconnected")
                    break;

                # CAMERA STATUS
                for processor in nvr.frame_processors.values():
                    data = {
                        "name": processor.camera.config.name,
                        "status": processor.status_text,
                        "objects": processor.objects_text,
                        "recording": processor.camera.recording_state.recording
                    }
                    payload = {"type": "cameraStatus", "data": data}

                    yield (
                        "event: cameraStatus\n"
                        f"data: {json.dumps(payload)}\n\n"
                    )

                # LOGS
                logs_list = list(event_log)

                if last_log_time is not None:
                    timestamps = [obj["timestamp"] for obj in logs_list]
                    index = bisect.bisect_right(timestamps, last_log_time)
                    logs_list = logs_list[index:]

                for logLine in logs_list:
                    payload = {"type": "logLine", "data": logLine}
                    yield (
                        "event: logLine\n"
                        f"data: {json.dumps(payload)}\n\n"
                    )

                last_log_time = time.time()

                # RECORDINGS
                recordings = list(nvr.recordings)
                start_times = None
                if last_recording_time is not None:
                    start_times = [obj["start_time"] for obj in recordings]
                    index = bisect.bisect_right(start_times, last_recording_time)
                    recordings = recordings[index:]

                for recording in recordings:
                    payload = {"type": "newEvent", "data": recording}
                    logger.debug(f"sending event: {payload}")
                    yield (
                        "event: newEvent\n"
                        f"data: {json.dumps(payload)}\n\n"
                    )

                if recordings:
                    last_recording_time = recordings[-1]["start_time"]

                await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            log_event("Client disconnected from SSE stream.")
            raise

        pass

    @app.get("/api/stream")
    async def stream(request: Request, user=Depends(require_user)):
        return EventSourceResponse(
            event_generator(request)
        )

    app.mount("/recordings", AuthStaticFiles(directory=config["recordings_directory"], check_dir=True), name="recordings")
    app.mount("/logs", AuthStaticFiles(directory=config["logs_directory"], check_dir=True), name="recordings")
    app.mount("/", AuthStaticFiles(directory="pynvr/frontend_dist", html=True), name="frontend")

    @app.get("/{path:path}")
    async def spa_fallback(path: str, user=Depends(require_user)):
        return FileResponse("pynvr/frontend_dist/index.html")

    return app
