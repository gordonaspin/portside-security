from queue import Queue, Empty
from subprocess import Popen
from datetime import datetime
import os
import json
import time

import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO
from wcwidth import width

from logger.logger import log_event
from nvr.motion_profiles import MotionProfile, DayMotionProfile, NightMotionProfile
from nvr.motion_tuner import MotionProfileAutoTuner
from nvr.utils import make_ts_string
from nvr.frame_buffers import FrameBuffers
from nvr.motion_detector import MotionDetector, MotionResult
from nvr.auto_tuner_wrapper import AutoTunerWrapper
from nvr.recording_state import RecordingState

class CameraConfig:
    def __init__(self, cfg: dict, width: int, height: int,
                 name: str, logs_dir: str, recordings_dir: str):
        self.name = name
        self.max_pixels = width * height
        self.width = width
        self.height = height
        self.enabled = cfg["enabled"]
        self.debug = cfg["debug"]
        self.url = cfg["url"]

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
    def __init__(
        self,
        cfg: dict,
        width: int,
        height: int,
        name: str,
        logs_dir: str,
        recordings_dir: str,
        model: YOLO,
    ):
        self.config = CameraConfig(cfg, width, height, name, logs_dir, recordings_dir)
        self.buffers = FrameBuffers(width, height)
        self.motion = MotionDetector(cfg, self.config.max_pixels, model)
        self.tuner = AutoTunerWrapper(self.motion, self.config)
        self.recording_state = RecordingState()

        self.model: YOLO = model
        self.debug: bool = cfg["debug"]

        # --- Latest-frame-wins buffer ---
        self.latest_frame: NDArray[np.uint8] | None = None
        self.debug_motion_image: NDArray[np.uint8] | None = None

        # State variables
        self.is_night: bool = False

        # --- LPR ---
        if "lpr" in cfg and cfg["lpr"].get("enabled", False):
            self.lpr: LPR = LPR(cfg["lpr"])

    def is_lpr(self) -> bool:
        return hasattr(self, "lpr")

    def update_yolo_confidence_threshold(self, val):
        self.motion.profile.yolo_confidence_threshold.value = val

    def update_motion_threshold(self, val):
        self.motion.profile.motion_threshold.value = val

class LPR:
    def __init__(self, cfg: dict):

        # --- Config ---
        self.cfg: dict = cfg
        self.url: str = cfg["url"]
        self.left: int = cfg["left"]
        self.top: int = cfg["top"]
        self.width: int = cfg["width"]
        self.height: int = cfg["height"]

        # --- Frame queue ---
        self.queue: Queue[NDArray[np.uint8]] = Queue(maxsize=1)

        # --- Process state ---
        self.first_frame: bool = True

        # --- Buffers (typed as arrays, initialized as None) ---
        self.gray_buf: NDArray[np.uint8] | None = None
        self.equalized_buf: NDArray[np.uint8] | None = None
        self.preprocessed_buf: NDArray[np.uint8] | None = None

