import time
import json
import os
from nvr.utils import make_ts_string
from nvr.motion_tuner import MotionProfileAutoTuner


class AutoTunerWrapper:
    def __init__(self, motion_detector, config):
        self.motion = motion_detector
        self.config = config
        self.tuner = MotionProfileAutoTuner()
        self.last_auto_adjust = 0.0

    def maybe_auto_adjust(self, now: float):
        if now - self.last_auto_adjust <= 60:
            return

        self._apply_adjustments()
        self.last_auto_adjust = now

    def _apply_adjustments(self):
        tuner = self.tuner
        stats = tuner.summarize()
        recs = tuner.recommend_adjustments()

        if not recs:
            tuner.reset()
            return

        before = self.motion.profile_to_dict()
        ap = self.motion.adaptive_profile
        prof = self.motion.profile
        max_pixels = prof.max_pixels

        def clamp(v, lo, hi):
            return max(lo, min(hi, v))

        # -----------------------------
        # SAFE ADJUSTMENTS
        # -----------------------------
        if "min_motion_frames" in recs:
            ap["min_motion_frames"].value -= 1
            ap["min_motion_frames"].value = clamp(ap["min_motion_frames"].value, 4, 16)

        if "min_total_motion_area" in recs:
            ap["min_total_motion_area"] += 0.001 * 0.2 * max_pixels
            ap["min_total_motion_area"] = clamp(
                ap["min_total_motion_area"],
                0.003 * max_pixels,
                0.006 * max_pixels,
            )

        # Push adaptive values back into profile
        prof.min_total_motion_area = ap["min_total_motion_area"]
        prof.min_motion_frames.value = ap["min_motion_frames"].value
        prof.inflate_motion_boxes = ap["inflate_motion_boxes"]

        after = self.motion.profile_to_dict()

        # Reset tuner
        tuner.reset()

        # -----------------------------
        # WRITE JSON LOG
        # -----------------------------
        log = {
            "timestamp": time.time(),
            "camera": self.config.name,
            "is_night": False,  # Camera will pass this later if needed
            "before_profile": before,
            "after_profile": after,
            "tuner_stats": stats,
            "recommendations": recs,
        }

        ts = make_ts_string()
        log_filename = os.path.join(
            self.config.logs_dir,
            f"{ts}_{self.config.name}_tuner.json"
        )

        with open(log_filename, "w") as f:
            json.dump(log, f, default=lambda o: o.__dict__, indent=4)
