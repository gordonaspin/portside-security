import os
import time
from queue import Queue

import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO

from backend.nvr.camera.auto_tuner_wrapper import AutoTunerWrapper
from backend.nvr.camera.frame_buffers import FrameBuffers
from backend.nvr.camera.motion_detector import MotionDetector
from backend.nvr.camera.recording_state import RecordingState

class CameraConfig:
    def __init__(self, config: dict,
                 name: str,
                 logs_dir: str,
                 recordings_dir: str):
        cfg = config["cameras"][name]

        width = cfg["resolution"]["width"]
        height = cfg["resolution"]["height"]
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
        self.config = CameraConfig(config, name, logs_dir, recordings_dir)
        self.buffers = FrameBuffers(config, width, height)
        self.motion = MotionDetector(config["cameras"][name], width, height)
        self.tuner = AutoTunerWrapper(self.motion, self.config)
        self.recording_state = RecordingState()

        self.debug: bool = config["cameras"][name]["debug"]

        # --- Latest-frame-wins buffer ---
        self.latest_frame: NDArray[np.uint8] | None = None
        self.yolo_frame: np.ndarray | None = None
        self.debug_motion_image: NDArray[np.uint8] | None = None

        # State variables
        self.is_night: bool = False

    def update_yolo_confidence_threshold(self, val):
        self.motion.profile.yolo_confidence_threshold.value = val

    def update_motion_threshold(self, val):
        self.motion.profile.motion_threshold.value = val
