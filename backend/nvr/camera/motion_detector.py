import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from backend.nvr.camera.motion_profiles import DayMotionProfile, NightMotionProfile
from backend.utils.utils import make_readable_ts

@dataclass
class MotionResult:
    motion_boxes: list[tuple[int, int, int, int]]
    has_moving_object: bool
    edge_density: float
    motion_confidence: float
    motion_persistence: int
    classes_in_frame: dict[str, set[str]]
    active_objects: dict[str, set[str]]
    active_segments: list[str]
    score: int
    pixel_score: float
    box_score: float
    persist_score: float
    last_motion_time: float


class MotionDetector:
    def __init__(self, cfg: dict, width: int, height: int):
        self.width = width
        self.height = height
        self.max_pixels = self.width * self.height

        # Profiles
        self.day_profile = DayMotionProfile(
            width=width,
            height=height,
            max_pixels=self.max_pixels,
            yolo_confidence_threshold=cfg["yolo_confidence"],
            motion_threshold=cfg["motion_threshold"],
            min_motion_confidence=cfg["minimum_motion_confidence"],
            min_motion_frames=cfg["minimum_motion_frames"],
            min_sum_box_area=cfg["minimum_sum_box_area"],
        )
        self.night_profile = NightMotionProfile(
            width=width,
            height=height,
            max_pixels=self.max_pixels,
            yolo_confidence_threshold=cfg["yolo_confidence"],
            motion_threshold=cfg["motion_threshold"],
            min_motion_confidence=cfg["minimum_motion_confidence"],
            min_motion_frames=cfg["minimum_motion_frames"],
            min_sum_box_area=cfg["minimum_sum_box_area"],
        )
        self.profile = self.day_profile

        # Motion state
        self.noise = 0.0
        self.motion_boxes_list = []
        self.classes_in_frame_dict = defaultdict(set)
        self.active_objects_dict = defaultdict(set)
        self.active_segments_list = []

        self.motion_confidence = 0.0
        self.motion_persistence = 0
        self.edge_density = 0.0
        self.score = 0
        self.pixel_score = 0.0
        self.box_score = 0.0
        self.persist_score = 0.0

        # Authoritative motion timestamp
        self.last_motion_time = time.time()

        self.has_moving_object = False

    def update_confidence(self, motion_boxes, now):
        # PIXEL SCORE
        self.pixel_score = min(
            self.score / (self.profile.motion_threshold_pixels * 3.0),
            1.0,
        )

        # BOX SCORE
        object_area = sum((x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in motion_boxes)
        self.box_score = min(
            object_area / (self.profile.min_sum_box_area_pixels * 2.0),
            1.0,
        )

        # PERSISTENCE SCORE
        self.persist_score = max(
            0.0,
            1.0 - ((now - self.last_motion_time) / self.profile.motion_persistence_time),
        )

        # FINAL CONFIDENCE
        self.motion_confidence = (
            (self.pixel_score * 0.4)
            + (self.box_score * 0.4)
            + (self.persist_score * 0.2)
        )

    def update_persistence(self, motion_boxes):
        object_area = sum((x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in motion_boxes)

        is_object_like_motion = (
            motion_boxes
            and object_area >= self.profile.min_sum_box_area_pixels
            and self.edge_density >= 0.02
            and self.has_moving_object
            and self.motion_confidence >= 0.15
        )

        if is_object_like_motion:
            self.motion_persistence += 1
        else:
            self.motion_persistence = max(0, self.motion_persistence - 1)

    def profile_to_dict(self) -> dict:
        data: dict[str, Any] = {}
        data["noise"] = self.noise
        data["classes"] = {key: list(value) for key, value in self.classes_in_frame_dict.items()}
        data["active_objects"] = {key: list(value) for key, value in self.active_objects_dict.items()}

        data["motion_confidence"] = self.motion_confidence
        data["motion_persistence"] = self.motion_persistence
        data["edge_density"] = self.edge_density
        data["score"] = self.score
        data["pixel_score"] = self.pixel_score
        data["box_score"] = self.box_score
        data["persist_score"] = self.persist_score
        data["last_motion_time"] = make_readable_ts(self.last_motion_time)
        data["has_moving_object"] = self.has_moving_object
        data["profile"] = {}
        prof = self.profile
        for k, v in vars(prof).items():
            if k.startswith("_"):
                continue
            if callable(v):
                continue
            data["profile"][k] = v
        return data
