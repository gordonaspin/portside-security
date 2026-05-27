from dataclasses import dataclass
from collections import defaultdict

class ProfileValue:
    def __init__(self, default, min, max, step):
        self.default = default
        self.min = min
        self.value = default
        self.max = max
        self.step = step

class MotionProfile:
    def __init__(
        self,
        max_pixels: int,
        yolo_confidence_threshold: float,
        motion_threshold: float,
        min_motion_confidence: float,
        min_motion_frames: int,
        min_sum_box_area: float):

        self.max_pixels: int = max_pixels
        self.yolo_confidence_threshold: ProfileValue = ProfileValue(yolo_confidence_threshold, 0.1, 1.0, 0.01)
        self.motion_threshold: ProfileValue = ProfileValue(motion_threshold, 0.1, 1.0, 0.01)
        self.min_motion_confidence: ProfileValue = ProfileValue(min_motion_confidence, 0.1, 1.0, 0.01)
        self.min_motion_frames: ProfileValue = ProfileValue(min_motion_frames, 5, 20, 1)
        self.min_sum_box_area: ProfileValue = ProfileValue(min_sum_box_area, 0.1, 1.5, 0.01)

        self.min_box_width = 20
        self.min_box_height = 20

    def min_edge_density(self, noise: float):
        return 0.022 + (noise * 0.0012)
    
    def set_yolo_confidence_threshold(self, val):
        self.yolo_confidence_threshold.value = val

    def set_motion_threshold(self, val):
        self.motion_threshold.value = val

    def set_min_motion_confidence(self, val):
        self.min_motion_confidence.value = val

    def set_min_motion_frames(self, val):
        self.min_motion_frames.value = val

    def set_min_sum_box_area(self, val):
        self.min_sum_box_area.value = val

    @property
    def min_sum_box_area_pixels(self):
        return self.min_sum_box_area.value * self.max_pixels / 100
    
    @property
    def motion_threshold_pixels(self):
        return self.motion_threshold.value * self.max_pixels / 100


class DayMotionProfile(MotionProfile):
    def __init__(
        self,
        max_pixels,
        yolo_confidence_threshold,
        motion_threshold,
        min_motion_confidence,
        min_motion_frames,
        min_sum_box_area):
        super().__init__(
            max_pixels,
            yolo_confidence_threshold,
            motion_threshold,
            min_motion_confidence,
            min_motion_frames,
            min_sum_box_area)
        self.min_contour_solidity = 0.70
        self.min_contour_area_ratio = 0.0045
        self.max_allowed_aspect_ratio = 6.0
        self.min_total_motion_area = 0.006 * max_pixels
        self.inflate_motion_boxes = 13
        self.motion_persistence_time = 2.0


class NightMotionProfile(MotionProfile):
    def __init__(
        self,
        max_pixels,
        yolo_confidence_threshold,
        motion_threshold,
        min_motion_confidence,
        min_motion_frames,
        min_sum_box_area):
        super().__init__(
            max_pixels,
            yolo_confidence_threshold - 0.10,
            motion_threshold * 0.7,
            min_motion_confidence,
            min_motion_frames - 2,
            min_sum_box_area * 0.6)
        self.min_contour_solidity = 0.55
        self.min_contour_area_ratio = 0.0025
        self.max_allowed_aspect_ratio = 5.0
        self.min_total_motion_area = 0.0045 * max_pixels
        self.inflate_motion_boxes = 10
        self.motion_persistence_time = 3.0   # seconds




