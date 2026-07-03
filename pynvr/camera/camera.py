import os
import time
from dataclasses import dataclass
from queue import Queue

import numpy as np
from numpy.typing import NDArray

from .frame_buffers import FrameBuffers
from .motion_detector import MotionDetector
from ..utils import ConfigValue


@dataclass
class RecordingState:
    recording: bool = False
    recording_start_time: float = 0.0
    should_record: bool = False
    should_continue: bool = False

    # Used only by shadow filters, not recording logic
    white_ratio: float = 0.0


class CameraConfig:
    def __init__(self, config: dict,
                 name: str,
                 logs_dir: str,
                 recordings_dir: str):
        cfg = config["cameras"][name]

        width = cfg["resolution"]["width"]
        height = cfg["resolution"]["height"]
        self.yolo_confidence: ConfigValue = ConfigValue(default=cfg["yolo_confidence"], min=0.1, max=1.0, step=0.01)
        self.name = name
        self.max_pixels = width * height
        self.width = width
        self.height = height
        self.enabled = cfg["enabled"]
        self.debug = cfg["debug"]
        self.url = cfg["url"]
        self.render_annotations = cfg["render_annotations"]

        # Directories
        self.logs_dir = logs_dir
        self.recordings_dir = os.path.join(recordings_dir, name)
        self.segments_dir = os.path.join(recordings_dir, "segments", name)
        self.images_dir = os.path.join(recordings_dir, "images", name)
        self.metadata_dir = os.path.join(recordings_dir, "metadata", name)
        self.plates_dir = os.path.join(recordings_dir, "plates", name)

        # Ensure dirs exist
        os.makedirs(self.recordings_dir, exist_ok=True)
        os.makedirs(self.segments_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.plates_dir, exist_ok=True)


class Camera:
    """
    Camera wiring for ByteTrack-only motion:

    - Owns FrameBuffers (full + YOLO frame)
    - Owns MotionDetector (ByteTrack + velocity-based motion)
    - Owns RecordingState
    - Tracks night/day state
    - Holds latest frames for UI/debug
    """

    def __init__(
        self,
        width: int,
        height: int,
        config: dict,
        name: str,
        logs_dir: str,
        recordings_dir: str,
    ):
        self.width = width
        self.height = height
        self.start_time = time.time()

        # Per-camera config (paths, resolution, flags)
        self.config = CameraConfig(config, name, logs_dir, recordings_dir)

        # Frame buffers: full-res + optional YOLO-res
        self.buffers = FrameBuffers(config, width, height)

        # ByteTrack-only motion detector
        # Expects camera-specific config with:
        #   track_thresh, match_thresh, track_buffer,
        #   minimum_track_speed, yolo_confidence
        self.motion = MotionDetector(config["cameras"][name], name)

        # Recording state machine
        self.recording_state = RecordingState()

        # Debug flag
        self.debug: bool = config["cameras"][name]["debug"]

        # Latest-frame-wins buffers for UI/debug
        self.latest_frame: NDArray[np.uint8] | None = None
        self.yolo_frame: NDArray[np.uint8] | None = None
        self.debug_motion_image: NDArray[np.uint8] | None = None

        # Night/day state (used for color detection, UI, optional YOLO tweaks)
        self.is_night: bool = False

        # Optional: event queue / control messages
        self.events: "Queue[dict]" = Queue()

    # ----------------------------------------------------------------------
    # Night/day helpers (FrameProcessor will set is_night)
    # ----------------------------------------------------------------------
    def set_night(self, value: bool):
        self.is_night = value

    # ----------------------------------------------------------------------
    # Convenience hooks for FrameProcessor (optional)
    # ----------------------------------------------------------------------
    def update_latest_frame(self, frame_bgr: NDArray[np.uint8]):
        self.latest_frame = frame_bgr

    def update_yolo_frame(self, frame_bgr: NDArray[np.uint8]):
        self.yolo_frame = frame_bgr

    def update_debug_motion_image(self, frame_bgr: NDArray[np.uint8]):
        self.debug_motion_image = frame_bgr
