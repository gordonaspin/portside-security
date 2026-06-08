import asyncio
import bisect
import json
import secrets
import time
from datetime import datetime, timedelta

from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import FastAPI, Request, Depends, HTTPException, Response, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from passlib.context import CryptContext
from pydantic import BaseModel

from nvr.nvr import NVR
from logger.logger import log_event, event_log
from webrtc.webrtc import CameraTrack, MosaicTrack

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
        return FileResponse("backend/frontend_dist/index.html")
    
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
        return {"classes": config["model"]["classes"]}

    @app.get("/api/system_name")
    def get_system_name(user=Depends(require_user)):
        return {"system_name": config["system_name"]}

    @app.get("/api/mosaic_dimensions")
    def get_mosaic_dimensions(user=Depends(require_user)):
        return config["mosaic"]

    @app.get("/api/events")
    def api_events(
        mobile: bool = False,
        since: float | None = None,
        user = Depends(require_user)
    ):
        events = list(nvr.recordings)

        # --- MOBILE FILTER (your existing logic) ---
        if mobile:
            cutoff_time = datetime.now() - timedelta(hours=4)
            end_times = [obj["end_time"] for obj in events]
            index = bisect.bisect_left(end_times, cutoff_time.timestamp())
            events = events[index:]

        # --- NEW: RETURN ONLY EVENTS NEWER THAN "since" ---
        if since is not None:
            start_times = [obj["start_time"] for obj in events]
            index = bisect.bisect_right(start_times, since)
            events = events[index:]

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

        settings = {
            "yolo_confidence_threshold": { 
                "value": camera.motion.profile.yolo_confidence_threshold.value,
                "min": camera.motion.profile.yolo_confidence_threshold.min,
                "max": camera.motion.profile.yolo_confidence_threshold.max,
                "step": camera.motion.profile.yolo_confidence_threshold.step
            },
            "motion_threshold": {
                "value": camera.motion.profile.motion_threshold.value,
                "min": camera.motion.profile.motion_threshold.min,
                "max": camera.motion.profile.motion_threshold.max,
                "step": camera.motion.profile.motion_threshold.step
            },
            "min_motion_confidence": {
                "value": camera.motion.profile.min_motion_confidence.value,
                "min": camera.motion.profile.min_motion_confidence.min,
                "max": camera.motion.profile.min_motion_confidence.max,
                "step": camera.motion.profile.min_motion_confidence.step
            },
            "min_motion_frames": {
                "value": camera.motion.profile.min_motion_frames.value,
                "min": camera.motion.profile.min_motion_frames.min,
                "max": camera.motion.profile.min_motion_frames.max,
                "step": camera.motion.profile.min_motion_frames.step
            },
            "min_sum_box_area": {
                "value": camera.motion.profile.min_sum_box_area.value,
                "min": camera.motion.profile.min_sum_box_area.min,
                "max": camera.motion.profile.min_sum_box_area.max,
                "step": camera.motion.profile.min_sum_box_area.step
            }
        }

        return settings

    @app.post("/api/cameras/{camera_name}/settings/{setting}")
    def update_camera_setting(camera_name: str, setting: str, payload: SettingValue):

        camera = nvr.cameras[camera_name]
        profile = camera.motion.profile

        # Update the camera object directly
        attr = getattr(profile, setting)
        log_event(f"{setting} {attr.value:.2f} -> {payload.value:.2f}", camera=camera)

        attr.value = payload.value

        return {
            "status": "ok",
            "camera": camera_name,
            "setting": setting,
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
    async def set_camera_debug(camera_name: str, payload: SettingValue, user=Depends(require_user)):
        camera = nvr.cameras.get(camera_name)
        log_event(f"debug {payload.value}", camera=camera)
        camera.config.debug = payload.value
        return {"status": "ok", "camera": camera_name, "value": payload.value}
    
    # Verbose debug
    @app.get("/api/settings/debug")
    async def get_debug(user=Depends(require_user)):
        return {"value": nvr.debug}
    
    @app.post("/api/settings/debug")
    async def set_debug(payload: SettingValue, user=Depends(require_user)):
        log_event(f"verbose logging {payload.value}")
        nvr.debug = payload.value
        return {"status": "ok", "value": payload.value}

    @app.get("/api/server_time")
    async def server_time(user=Depends(require_user)):
        return {"epoch": time.time()}
    

    async def status_generator():
        try:
            while True:
                payload = []

                for processor in nvr.frame_processors.values():
                    payload.append({
                        "name": processor.camera.config.name,
                        "status": processor.status_text,
                        "objects": processor.objects_text,
                        "recording": processor.camera.recording_state.recording
                    })

                # Convert to JSON and wrap in SSE format
                msg = (
                    "retry: 3000\n"
                   f"data: {json.dumps(payload)}\n\n"
                )
                yield msg
                await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            print("Client disconnected from SSE stream.")
            raise

    @app.get("/camera_status")
    async def stream_events(user=Depends(require_user)):
        return StreamingResponse(
            status_generator(),
            media_type="text/event-stream"
        )


    app.mount("/recordings", AuthStaticFiles(directory=config["recordings_directory"], check_dir=True), name="recordings")
    app.mount("/", AuthStaticFiles(directory="backend/frontend_dist", html=True), name="frontend")

    @app.get("/{path:path}")
    async def spa_fallback(path: str, user=Depends(require_user)):
        return FileResponse("backend/frontend_dist/index.html")

    return app
