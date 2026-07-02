import time
from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger

import numpy as np

from .byte_tracker import BYTETracker
from .config_value import ConfigValue

logger = getLogger("pynvr")

@dataclass
class MotionTrack:
    track_id: int
    cls: int
    conf: float
    tlbr: tuple[int, int, int, int]
    speed: float
    relative_speed: float


class MotionDetector:
    """
    ByteTrack-only motion detector.
    Responsibilities:
    - Run BYTETracker on YOLO detections
    - Compute per-track velocity
    - Decide has_moving_object based on minimum_track_speed_pxpf and linger time
    - Maintain classes_in_frame_dict and active_objects_dict
    - Maintain last_motion_time
    - Expose active_tracks for debug UI
    """

    def __init__(self, cfg: dict, name: str):
        self.name = name
        self.track_threshold: ConfigValue = ConfigValue(default=cfg["track_threshold"], min=0.1, max=1.0, step=0.01)
        self.match_threshold: ConfigValue = ConfigValue(default=cfg["match_threshold"], min=0.1, max=1.0, step=0.01)
        self.track_buffer: ConfigValue = ConfigValue(default=cfg["track_buffer"], min=30, max=300, step=1)
        # --- ByteTrack configuration ---
        self.tracker = BYTETracker(
            track_thresh=self.track_threshold.value,
            match_thresh=self.match_threshold.value,
            track_buffer=self.track_buffer.value,
        )

        # Minimum pixel velocity to consider an object "moving"
        self.minimum_relative_motion: ConfigValue = ConfigValue(default=cfg["minimum_relative_motion"], min=0.05, max=0.2, step=0.01)

        # --- Motion state ---
        self.has_moving_object: bool = False
        self.last_motion_time: float = time.time()

        # YOLO class + color metadata
        self.classes_in_frame_dict = defaultdict(set)
        self.active_objects_dict = defaultdict(set)

        # Track history for velocity computation
        self.last_positions: dict[int, tuple[float, float]] = {}

        # Active ByteTrack objects for debug UI
        self.active_tracks: list[MotionTrack] = []
        self.moving_track_ids: set[int] = set()
        self.smoothed_speeds = {}  # tid → smoothed relative speed

    # ----------------------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------------------
    def update(self, yolo_dets: np.ndarray, full_w: int, full_h: int, now: float):
        """
        Update motion state using YOLO detections and ByteTrack.
        yolo_dets: Nx6 array [x1, y1, x2, y2, score, cls]
        """

        # Run ByteTrack
        online_targets = self.tracker.update(
            yolo_dets,
            img_info=(full_h, full_w),
            img_size=(full_h, full_w),
        )

        moving = False
        active_tracks = []
        self.moving_track_ids.clear()

        for t in online_targets:
            tid = t.track_id
            x1, y1, x2, y2 = t.tlbr
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            size = max(w, h)

            # Compute velocity
            speed = 0.0
            relative_speed = 0.0
            smooth = 0.0
            if tid in self.last_positions:
                px, py = self.last_positions[tid]
                dx = cx - px
                dy = cy - py
                speed = float((dx * dx + dy * dy) ** 0.5)
                relative_speed = float(speed / size if size > 0 else 0)
                prev = self.smoothed_speeds.get(tid, 0.0)
                alpha = 0.25
                smooth = float(alpha * relative_speed + (1 - alpha) * prev)
                self.smoothed_speeds[tid] = smooth
                if smooth > self.minimum_relative_motion.value:
                    moving = True
                    self.moving_track_ids.add(tid)
                    #logger.debug(f"{self.name} tid: {tid}: speed {speed:.2f} relative_speed: {relative_speed:.2f}")
                else:
                    pass
                    # NEW: keep track alive even if stationary
                    #moving = moving or (now - self.last_motion_time < 3.0)

            self.last_positions[tid] = (cx, cy)

            active_tracks.append(MotionTrack(
                track_id=tid,
                cls=int(t.cls),
                conf=float(t.score),
                tlbr=(int(x1), int(y1), int(x2), int(y2)),
                speed=speed,
                relative_speed=smooth
            ))

        self.active_tracks = active_tracks
        self.has_moving_object = moving

        if moving:
            self.last_motion_time = now

    # ----------------------------------------------------------------------
    # CLASS + COLOR METADATA
    # ----------------------------------------------------------------------
    def clear_frame_classes(self):
        """Call this at the start of each frame before adding YOLO class/color info."""
        self.classes_in_frame_dict.clear()

    def add_class_color(self, class_name: str, color: str):
        """Called by FrameProcessor after color detection."""
        self.classes_in_frame_dict[class_name].add(color)

    def finalize_active_objects(self):
        """
        Merge classes_in_frame_dict into active_objects_dict while recording.
        Called by FrameProcessor during recording.
        """
        for cls, colors in self.classes_in_frame_dict.items():
            self.active_objects_dict[cls].update(colors)

    def reset_active_objects(self):
        """Called when recording stops."""
        self.active_objects_dict.clear()

    # ----------------------------------------------------------------------
    # DEBUG / INSPECTION
    # ----------------------------------------------------------------------
    def to_dict(self):
        """Minimal state for debug UI or API."""
        return {
            "has_moving_object": self.has_moving_object,
            "last_motion_time": self.last_motion_time,
            "active_tracks": [
                vars(t)
                for t in self.active_tracks
            ],
            "classes_in_frame": {k: list(v) for k, v in self.classes_in_frame_dict.items()},
            "active_objects": {k: list(v) for k, v in self.active_objects_dict.items()},
        }
