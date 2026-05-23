from collections import deque, defaultdict
from queue import Queue, Empty
from subprocess import Popen
from datetime import datetime
import os
import json
import time

import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO

from logger.logger import log_event
from nvr.motion_profiles import MotionProfile, DayMotionProfile, NightMotionProfile
from nvr.motion_tuner import MotionProfileAutoTuner

class RollingAverage:
    def __init__(self, window_size=100):
        self.window = deque(maxlen=window_size)
        self.sum = 0.0

    def update(self, value):
        # If full, remove oldest from sum
        if len(self.window) == self.window.maxlen:
            self.sum -= self.window[0]

        self.window.append(value)
        self.sum += value

        return self.sum / len(self.window)
    
    def value(self):
        if not self.window:
            return 0.0
        return self.sum / len(self.window)
    
class Camera:
    def __init__(
        self,
        cfg: dict,
        max_pixels: int,
        name: str,
        logs_dir: str,
        recordings_dir: str,
        model: YOLO,
    ):

        # --- Basic config ---
        self.cfg: dict = cfg
        self.max_pixels = max_pixels
        self.name: str = name
        self.logs_dir: str = logs_dir
        self.recordings_dir: str = os.path.join(recordings_dir, self.name)
        self.segments_dir: str = os.path.join(recordings_dir, "segments", self.name)
        self.images_dir: str = os.path.join(recordings_dir, "images", self.name)
        self.metadata_dir: str = os.path.join(recordings_dir, "metadata", self.name)
        self.plates_dir: str = os.path.join(recordings_dir, "plates", self.name)
        self.model: YOLO = model
        self.debug: bool = cfg["debug"]

        # --- Stream state ---
        self.process: Popen | None = None
        self.frame_count: int = 0
        self.fail_count: int = 0

        # --- Latest-frame-wins buffer ---
        self.latest_frame: NDArray[np.uint8] | None = None
        self.frame_queue: Queue[NDArray[np.uint8]] = Queue(maxsize=1)

        # --- CV2 buffers (typed as arrays, initialized as None) ---
        self.background_buf: NDArray[np.float32] | None = None
        self.bg_frame_buf: NDArray[np.uint8] | None = None
        self.diff_blur_buf: NDArray[np.uint8] | None = None
        self.diff_buf: NDArray[np.uint8] | None = None
        self.diff_mask_buf: NDArray[np.uint8] | None = None
        self.diff_filtered_buf: NDArray[np.uint8] | None = None
        self.edges_buf: NDArray[np.uint8] | None = None
        self.gray_buf: NDArray[np.uint8] | None = None
        self.thresh_buf: NDArray[np.uint8] | None = None
        self.sobel_x_buf: NDArray[np.int16] | None = None
        self.sobel_y_buf: NDArray[np.int16] | None = None
        self.sobel_x_abs_buf: NDArray[np.uint8] | None = None
        self.sobel_y_abs_buf: NDArray[np.uint8] | None = None

        # --- FPS tracking ---
        self.total_frames: int = 0
        self.total_drops: int = 0
        self.dt: RollingAverage = RollingAverage(100)
        self.fps: RollingAverage = RollingAverage(100)
        self.drop_rate: float = 0.0
        self.last_frame_time: float = 0.0

        # --- UI / metadata ---
        self.status_text: str = "Not streaming"
        self.objects_text: str = ""

        # --- Logic state ---
        self.should_record: bool = False
        self.should_start: bool = False
        self.should_continue: bool = False
        self.last_night_time_check: float = time.time()
        self.last_motion_time: float = time.time()
        self.recording: bool = False
        self.is_night: bool = False
        self.recording_start_time: float = 0.0
        self.score: int = 0
        self.pixel_score: float = 0.0
        self.box_score: float = 0.0
        self.persist_score: float = 0.0
        self.edge_density: float = 0.0
        self.white_ratio: float = 0.0

        # --- Default Motion Profile (DAY) ---
        self.day_profile: MotionProfile = DayMotionProfile(
            max_pixels=self.max_pixels,
            yolo_confidence_threshold=cfg["yolo_confidence"],
            motion_threshold=cfg["motion_threshold"],
            min_motion_confidence=cfg["minimum_motion_confidence"],
            min_motion_frames=cfg["minimum_motion_frames"],
            min_sum_box_area=cfg["minimum_sum_box_area"]
        )
        self.night_profile = NightMotionProfile(
            max_pixels=self.max_pixels,
            yolo_confidence_threshold=cfg["yolo_confidence"] + 0.15,
            motion_threshold=cfg["motion_threshold"] * 1.5,
            min_motion_confidence=cfg["minimum_motion_confidence"] + 0.15,
            min_motion_frames=cfg["minimum_motion_frames"] + 2,
            min_sum_box_area=cfg["minimum_sum_box_area"]
            )
        self.profile = self.day_profile


        # --- Auto tuner ---
        self.auto_tuner: MotionProfileAutoTuner = MotionProfileAutoTuner()
        self.last_auto_adjust: float = 0.0

        # --- Motion detection ---
        self.noise: float = 0.0
        self.motion_boxes_list: list[tuple[int, int, int, int]] = []
        self.classes_in_frame_dict: dict[str, set[str]] = defaultdict(set)
        self.active_objects_dict: dict[str, set[str]] = defaultdict(set)
        self.active_segments_list: list[str] = []
        self.motion_confidence: float = 0.0
        self.motion_persistence: int = 0
        self.debug_motion_image: NDArray[np.uint8] | None = None
        self.keep_mask: list[bool] = []
        self.has_moving_object: bool = False

        # --- Apply config ---
        self.url: str = cfg["url"]
        self.enabled: bool = cfg["enabled"]

        # --- LPR ---
        if "lpr" in cfg and cfg["lpr"].get("enabled", False):
            self.lpr: LPR = LPR(cfg["lpr"])

        # --- Adaptive profile ---
        self.adaptive_profile: dict[str, float] = {
            "min_edge_density": self.profile.min_edge_density(0),
            "min_contour_area_ratio": self.profile.min_contour_area_ratio,
            "min_total_motion_area": self.profile.min_total_motion_area,
            "min_motion_frames": self.profile.min_motion_frames,
            "inflate_motion_boxes": self.profile.inflate_motion_boxes,
        }
        os.makedirs(self.segments_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.plates_dir, exist_ok=True)


    def profile_to_dict(self) -> dict:
        """
        Return a JSON‑safe snapshot of the motion profile.
        Skips callables (like min_edge_density lambdas).
        Uses adaptive_profile for values that back functions.
        """
        prof = self.profile
        ap = self.adaptive_profile  # your numeric backing store

        data = {}

        for k, v in vars(prof).items():
            # skip private attrs
            if k.startswith("_"):
                continue
            # skip callables (functions, lambdas, methods)
            if callable(v):
                continue
            data[k] = v

        # explicitly include the numeric base for min_edge_density
        if "min_edge_density" in ap:
            data["min_edge_density_base"] = ap["min_edge_density"]

        return data

    def is_lpr(self) -> bool:
        return hasattr(self, "lpr")

    def get_frame(self) -> NDArray[np.uint8] | None:
        """
        Retrieve the latest frame from the camera queue.
        This preserves the original behavior:
        - latest-frame-wins
        - drop frames if queue is full
        - timeout every 0.5s so thread can exit cleanly
        """
        try:
            frame: NDArray[np.uint8] = self.frame_queue.get(timeout=0.5)
            self.initialize_buffers(frame)
            return frame.copy()  # always work on a copy
        except Empty:
            return None

    def initialize_buffers(self, frame_bgr: NDArray[np.uint8]) -> None:
        """
        Initialize all OpenCV buffers on the first frame.
        This preserves your original logic exactly:
        - allocate all buffers once
        - reuse them forever
        - initialize background model from first gray frame
        """
        if self.frame_count:
            return

        self.frame_count = 1
        log_event(message="reading from stream", level="info", camera=self)

        h, w = frame_bgr.shape[:2]

        # Allocate all working buffers
        self.bg_frame_buf      = np.zeros((h, w), dtype=np.uint8)
        self.diff_blur_buf     = np.zeros((h, w), dtype=np.uint8)
        self.diff_buf          = np.zeros((h, w), dtype=np.uint8)
        self.diff_filtered_buf = np.zeros((h, w), dtype=np.uint8)
        self.diff_mask_buf     = np.zeros((h, w), dtype=np.uint8)
        self.edges_buf         = np.zeros((h, w), dtype=np.uint8)
        self.gray_buf          = np.zeros((h, w), dtype=np.uint8)
        self.thresh_buf        = np.zeros((h, w), dtype=np.uint8)
        self.sobel_x_buf       = np.zeros((h, w), dtype=np.int16)
        self.sobel_y_buf       = np.zeros((h, w), dtype=np.int16)
        self.sobel_x_abs_buf   = np.zeros((h, w), dtype=np.uint8)
        self.sobel_y_abs_buf   = np.zeros((h, w), dtype=np.uint8)

        # Background model starts as float32 version of first gray frame
        self.background_buf = self.gray_buf.astype("float32")

    def auto_adjust_if_needed(self, now: float) -> None:
        """
        Periodically auto-tune the motion profile.
        This preserves your original logic:
        - run auto_adjust_profile() every 60 seconds
        - log the tuning event
        """
        if now - self.last_auto_adjust <= 60:
            return

        self._auto_adjust_profile()
        self.last_auto_adjust = now


    def _auto_adjust_profile(self):
        tuner = self.auto_tuner
        stats = tuner.summarize()
        recs = tuner.recommend_adjustments()

        # nothing to do
        if not recs:
            tuner.reset()
            return

        before = self.profile_to_dict()
        ap = self.adaptive_profile
        prof = self.profile
        max_pixels = prof.max_pixels

        def clamp(v, lo, hi):
            return max(lo, min(hi, v))

        # apply SAFE recommendations

        if "min_edge_density" in recs:
            ap["min_edge_density"] += 0.002 * 0.2
            ap["min_edge_density"] = clamp(ap["min_edge_density"], 0.015, 0.04)

        if "min_motion_frames" in recs:
            ap["min_motion_frames"].value += max(1, int(2 * 0.2))
            ap["min_motion_frames"].value = clamp(ap["min_motion_frames"].value, 4, 16)

        if "min_total_motion_area" in recs:
            ap["min_total_motion_area"] += 0.001 * 0.2 * max_pixels
            ap["min_total_motion_area"] = clamp(
                ap["min_total_motion_area"],
                0.003 * max_pixels,
                0.006 * max_pixels,
            )

        # push adaptive values back into profile
        prof.min_total_motion_area = ap["min_total_motion_area"]
        prof.min_motion_frames.value = ap["min_motion_frames"].value
        prof.inflate_motion_boxes = ap["inflate_motion_boxes"]

        base = ap["min_edge_density"]
        prof.min_edge_density = lambda noise: base + noise * 0.0012

        after = self.profile_to_dict()

        # --- RESET TUNER ---
        tuner.reset()

        # --- WRITE JSON LOG ---
        log = {
            "timestamp": time.time(),
            "camera": self.name,
            "is_night": self.is_night,
            "before_profile": before,
            "after_profile": after,
            "tuner_stats": stats,
            "recommendations": recs,
        }

        timestamp_str = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(self.logs_dir, f"{timestamp_str}_{self.name}_tuner.json")

        with open(log_filename, "w") as f:
            json.dump(log, f, default=lambda o: o.__dict__, indent=4)

    def update_confidence(
        self,
        motion_boxes: list[tuple[int, int, int, int]],
        now: float
    ) -> None:
        """
        Compute pixel_score, box_score, persist_score, and motion_confidence.
        This preserves your original weighting:
        - pixel_score: 40%
        - box_score:   40%
        - persist:     20%
        """

        # --- PIXEL SCORE ---
        self.pixel_score = min(
            self.score / (self.profile.motion_threshold_pixels * 3.0),
            1.0
        )

        # --- BOX SCORE ---
        object_area = sum(
            (x2 - x1) * (y2 - y1)
            for (x1, y1, x2, y2) in motion_boxes
        )
        self.box_score = min(
            object_area / (self.profile.min_sum_box_area_pixels * 2.0),
            1.0
        )

        # --- PERSISTENCE SCORE ---
        self.persist_score = max(
            0.0,
            1.0 - ((now - self.last_motion_time) / self.profile.motion_persistence_time)
        )

        # --- FINAL MOTION CONFIDENCE ---
        self.motion_confidence = (
            (self.pixel_score * 0.4) +
            (self.box_score   * 0.4) +
            (self.persist_score * 0.2)
        )

    def update_persistence(
        self,
        motion_boxes: list[tuple[int, int, int, int]]
    ) -> None:
        """
        Update motion_persistence AFTER YOLO has determined:
        - camera.has_moving_object
        - object_area
        - edge_density
        - motion_confidence

        This preserves your original logic but moves it to the correct place.
        """

        object_area = sum(
            (x2 - x1) * (y2 - y1)
            for (x1, y1, x2, y2) in motion_boxes
        )

        is_object_like_motion = (
            motion_boxes and
            object_area >= self.profile.min_sum_box_area_pixels and
            self.edge_density >= 0.02 and
            self.has_moving_object and
            self.motion_confidence >= 0.15
        )

        if is_object_like_motion:
            self.motion_persistence += 1
        else:
            self.motion_persistence = max(0, self.motion_persistence - 1)

    def update_yolo_confidence_threshold(self, val):
        self.profile.yolo_confidence_threshold.value = val

    def update_motion_threshold(self, val):
        self.profile.motion_threshold.value = val

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
        self.process: Popen | None = None
        self.first_frame: bool = True

        # --- Buffers (typed as arrays, initialized as None) ---
        self.gray_buf: NDArray[np.uint8] | None = None
        self.equalized_buf: NDArray[np.uint8] | None = None
        self.preprocessed_buf: NDArray[np.uint8] | None = None

