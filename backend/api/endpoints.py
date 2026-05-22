from pathlib import Path
import os, secrets, time
from datetime import datetime, timedelta
import bisect

from fastapi import FastAPI, Request, Depends, HTTPException, Response, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from aiortc import RTCPeerConnection, RTCSessionDescription
from pydantic import BaseModel
from passlib.context import CryptContext

from webrtc.webrtc import CameraTrack, MosaicTrack
from logger.logger import log_event, event_log
from nvr.nvr import NVR
from context import Context
from camera.camera import Camera
from nvr.motion_profiles import DayMotionProfile

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

def create_app(ctx: Context, nvr: NVR):
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
        if payload.username != ctx.gui_username or not pwd.verify(payload.password, ctx.gui_password):
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
            cams = [c for c in nvr.cameras.values() if c.enabled]
            track = MosaicTrack(cams)
        else:
            cam = nvr.cameras[name]
            track = CameraTrack(cam)

        pc.addTrack(track)
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    @app.get("/api/cameras")
    def get_cameras(user=Depends(require_user)):
        return [{"name": cam.name, "debug": cam.debug}
                for cam in nvr.cameras.values() if cam.enabled]

    @app.get("/api/classes")
    def get_classes(user=Depends(require_user)):
        return {"classes": nvr.ctx.yolo_config["classes"]}

    @app.get("/api/events")
    def api_events(mobile: bool=False, user=Depends(require_user)):
        events = list(nvr.recordings)
        if mobile:
            cutoff_time = datetime.now() - timedelta(hours=4)
            end_times = [obj["end_time"] for obj in events]
            index = bisect.bisect_left(end_times, cutoff_time.timestamp())
            events = events[index:]
        return {"events": list(nvr.recordings)}

    @app.get("/api/logs")
    def get_logs(user=Depends(require_user)):
        html = "\n".join(event_log)
        return {"html": html}

    @app.get("/api/cameras/{camera}/settings")
    def get_camera_settings(camera: str):
        cam = nvr.cameras[camera]

        settings = {}

        settings = {
            "yolo_confidence_threshold": { 
                "value": cam.profile.yolo_confidence_threshold.value,
                "min": cam.profile.yolo_confidence_threshold.min,
                "max": cam.profile.yolo_confidence_threshold.max,
                "step": cam.profile.yolo_confidence_threshold.step
            },
            "motion_threshold": {
                "value": cam.profile.motion_threshold.value,
                "min": cam.profile.motion_threshold.min,
                "max": cam.profile.motion_threshold.max,
                "step": cam.profile.motion_threshold.step
            },
            "min_motion_confidence": {
                "value": cam.profile.min_motion_confidence.value,
                "min": cam.profile.min_motion_confidence.min,
                "max": cam.profile.min_motion_confidence.max,
                "step": cam.profile.min_motion_confidence.step
            },
            "min_motion_frames": {
                "value": cam.profile.min_motion_frames.value,
                "min": cam.profile.min_motion_frames.min,
                "max": cam.profile.min_motion_frames.max,
                "step": cam.profile.min_motion_frames.step
            },
            "min_sum_box_area": {
                "value": cam.profile.min_sum_box_area.value,
                "min": cam.profile.min_sum_box_area.min,
                "max": cam.profile.min_sum_box_area.max,
                "step": cam.profile.min_sum_box_area.step
            }
        }

        return settings

    @app.post("/api/cameras/{camera}/settings/{setting}")
    def update_camera_setting(camera: str, setting: str, payload: SettingValue):

        profile = nvr.cameras[camera].profile

        # Update the camera object directly
        attr = getattr(profile, setting)
        attr.value = payload.value

        return {
            "status": "ok",
            "camera": camera,
            "setting": setting,
            "value": payload.value
        }

    # camera debug
    @app.get("/api/settings/debug/{camera_name}")
    def get_camera_debug(camera_name: str):
        camera = nvr.cameras.get(camera_name)

        return {
            "camera": camera_name,
            "debug": camera.debug if camera else False
        }

    @app.post("/api/settings/debug/{camera_name}")
    async def set_camera_debug(camera_name: str, payload: SettingValue, user=Depends(require_user)):
        log_event(f"debug {payload.value}", camera=nvr.cameras[camera_name])
        cam = nvr.cameras.get(camera_name)
        if cam:
            cam.debug = payload.value
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

    app.mount("/recordings", AuthStaticFiles(directory=ctx.directory, check_dir=True), name="recordings")
    app.mount("/", AuthStaticFiles(directory="backend/frontend_dist", html=True), name="frontend")

    @app.get("/{path:path}")
    async def spa_fallback(path: str, user=Depends(require_user)):
        return FileResponse("backend/frontend_dist/index.html")

    return app
