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
    def __init__(self, max_pixels: int):
        self.max_pixels = max_pixels

    def min_edge_density(self, noise: float):
        return 0.022 + (noise * 0.0012)
    

class DayMotionProfile(MotionProfile):
    def __init__(self,
                 max_pixels,
                 yolo_confidence_threshold,
                 motion_threshold,
                 min_motion_confidence,
                 min_motion_frames,
                 min_sum_box_area):
        super().__init__(max_pixels)
        self.min_box_width = 20
        self.min_box_height = 20
        self.min_contour_solidity = 0.60
        self.min_contour_area_ratio = 0.0045
        self.max_allowed_aspect_ratio = 6.0
        self.min_total_motion_area = 0.006 * max_pixels
        self.inflate_motion_boxes = 13
        self.motion_persistence_time = 2.0
        self.yolo_confidence_threshold = ProfileValue(yolo_confidence_threshold, 0.1, 1.0, 0.01)
        self.motion_threshold = ProfileValue(motion_threshold, 0.1, 1.0, 0.01)
        self.min_motion_confidence = ProfileValue(min_motion_confidence, 0.1, 1.0, 0.01)
        self.min_motion_frames = ProfileValue(min_motion_frames, 5, 20, 1)
        self.min_sum_box_area = ProfileValue(min_sum_box_area, 0.1, 1.5, 0.05)

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
    
class NightMotionProfile(MotionProfile):
    def __init__(self,
                max_pixels,
                 yolo_confidence_threshold,
                 motion_threshold,
                 min_motion_confidence,
                 min_motion_frames,
                 min_sum_box_area):
        super().__init__(max_pixels)
        self.min_box_width = 24
        self.min_box_height = 24
        self.min_contour_solidity = 0.80                                                # stricter
        self.min_contour_area_ratio = 0.012                                             # larger min area of a single contour
        self.max_allowed_aspect_ratio = 5.0                                             # keeps square-ish objects, discards skinny long ones
        self.min_total_motion_area = 0.008 * max_pixels                                 # require more total motion
        self.inflate_motion_boxes = 10
        self.motion_persistence_time = 3.0   # seconds
        self.yolo_confidence_threshold = ProfileValue(yolo_confidence_threshold, 0.1, 1.0, 0.01)
        self.motion_threshold = ProfileValue(1.5 * motion_threshold, 0.1, 1.0, 0.1)
        self.min_motion_confidence = ProfileValue(min_motion_confidence + 0.15, 0.1, 1.0, 0.01)
        self.min_motion_frames = ProfileValue(min_motion_frames, 7, 20, 1)
        self.min_sum_box_area = ProfileValue(min_sum_box_area, 0.1, 1.5, 0.01)

    @property
    def min_sum_box_area_pixels(self):
        return self.min_sum_box_area.value * self.max_pixels
    
    @property
    def motion_threshold_pixels(self):
        return self.motion_threshold.value * 1.5 * self.max_pixels / 100

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


from dataclasses import dataclass
from collections import defaultdict

@dataclass
class MotionDecision:
    passed: bool
    reason: str
    details: dict


class MotionProfileAutoTuner:
    def __init__(self):
        self.decisions: list[MotionDecision] = []
        self.stats = defaultdict(int)

    def record(self, decision: MotionDecision, camera=None):
        # Ignore during recording – scene is too dynamic
        if camera is not None and camera.recording:
            return

        # Ignore when confidence is already good – motion was real
        if camera is not None:
            if camera.motion_confidence >= camera.profile.min_motion_confidence.value:
                return

        # YOLO overlap noise is not a motion error
        if decision.reason == "yolo_overlap_noise":
            return

        self.decisions.append(decision)
        self.stats[decision.reason] += 1

    def summarize(self):
        return dict(self.stats)

    def recommend_adjustments(self):
        rec: dict[str, str] = {}
        s = self.stats
        SCALE = 0.2  # 20% of original aggressiveness

        # Only keep SAFE rules

        # Too many shadow rejections → edge density slightly too low
        if s["shadow_low_edge"] > 100:
            rec["min_edge_density"] = f"increase by +{0.002 * SCALE:.4f}"

        # Too many short motions → min_motion_frames slightly too low
        if s["short_motion"] > 100:
            rec["min_motion_frames"] = f"increase by +{int(2 * SCALE) or 1}"

        # Too many low_total_area → min_total_motion_area slightly too low
        if s["low_total_area"] > 200:
            rec["min_total_motion_area"] = f"increase by +{0.001 * SCALE:.4f} * max_pixels"

        # DO NOT tune min_contour_area_ratio here – too dangerous
        # DO NOT tune min_motion_confidence here – keep manual

        return rec

    def reset(self):
        self.decisions.clear()
        self.stats.clear()
