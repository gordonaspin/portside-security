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
        scale = 0.2  # 20% of original aggressiveness

        # Only keep SAFE rules

        # Too many shadow rejections → edge density slightly too low
        if s["shadow_low_edge"] > 100:
            rec["min_edge_density"] = f"increase by +{0.002 * scale:.4f}"

        # Too many short motions → min_motion_frames slightly too low
        if s["short_motion"] > 100:
            rec["min_motion_frames"] = f"increase by +{int(2 * scale) or 1}"

        # Too many low_total_area → min_total_motion_area slightly too low
        if s["low_total_area"] > 200:
            rec["min_total_motion_area"] = f"increase by +{0.001 * scale:.4f} * max_pixels"

        # DO NOT tune min_contour_area_ratio here – too dangerous
        # DO NOT tune min_motion_confidence here – keep manual

        return rec

    def reset(self):
        self.decisions.clear()
        self.stats.clear()
