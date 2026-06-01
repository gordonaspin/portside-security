class ProfileValue:
    def __init__(self, default, min, max, step):
        self.default = default
        self.min = min
        self.value = default
        self.max = max
        self.step = step

class MotionProfile:
    """
    Resolution‑independent motion profile.
    User-adjustable values use ProfileValue.
    Internal thresholds use instance attributes.
    """

    def __init__(
        self,
        width: int,
        height: int,
        max_pixels: int,
        yolo_confidence_threshold: float,
        motion_threshold: float,
        min_motion_confidence: float,
        min_motion_frames: int,
        min_sum_box_area: float
    ):
        self.width = width
        self.height = height
        self.max_pixels = max_pixels

        # -------------------------------
        # USER-ADJUSTABLE VALUES
        # -------------------------------
        self.yolo_confidence_threshold = ProfileValue(yolo_confidence_threshold, 0.1, 1.0, 0.01)
        self.motion_threshold = ProfileValue(motion_threshold, 0.1, 1.0, 0.01)
        self.min_motion_confidence = ProfileValue(min_motion_confidence, 0.1, 1.0, 0.01)
        self.min_motion_frames = ProfileValue(min_motion_frames, 5, 20, 1)
        self.min_sum_box_area = ProfileValue(min_sum_box_area, 0.1, 1.5, 0.01)

        # -------------------------------
        # INTERNAL THRESHOLDS (INSTANCE ATTRIBUTES)
        # These are NOT user-adjustable.
        # -------------------------------
        self.min_box_width_percent  = 1.5   # % of width
        self.min_box_height_percent = 1.5   # % of height
        self.min_total_motion_area_percent = 0.25  # % of total pixels

        # -------------------------------
        # DERIVED PIXEL THRESHOLDS
        # -------------------------------
        self.min_box_width  = int(self.min_box_width_percent  * width  / 100)
        self.min_box_height = int(self.min_box_height_percent * height / 100)
        self.min_total_motion_area = int(self.min_total_motion_area_percent * max_pixels / 100)

        # Day/Night profiles override these:
        self.min_contour_solidity = 0.60
        self.min_contour_area_ratio = 0.0035
        self.max_allowed_aspect_ratio = 6.0
        self.inflate_motion_boxes = 12
        self.motion_persistence_time = 2.0

    # -------------------------------
    # PROPERTIES
    # -------------------------------
    @property
    def min_sum_box_area_pixels(self):
        return self.min_sum_box_area.value * self.max_pixels / 100

    @property
    def motion_threshold_pixels(self):
        return self.motion_threshold.value * self.max_pixels / 100

    # -------------------------------
    # NOISE-ADAPTIVE EDGE DENSITY
    # -------------------------------
    def min_edge_density(self, noise: float):
        return 0.022 + (noise * 0.0012)


class DayMotionProfile(MotionProfile):
    def __init__(
        self,
        width, height, max_pixels,
        yolo_confidence_threshold,
        motion_threshold,
        min_motion_confidence,
        min_motion_frames,
        min_sum_box_area
    ):
        super().__init__(
            width, height, max_pixels,
            yolo_confidence_threshold,
            motion_threshold,
            min_motion_confidence,
            min_motion_frames,
            min_sum_box_area
        )

        # Override instance attributes
        self.min_box_width_percent  = 1.5
        self.min_box_height_percent = 1.5
        self.min_total_motion_area_percent = 0.60  # 0.60%

        # Recompute pixel thresholds
        self.min_box_width  = int(self.min_box_width_percent  * width  / 100)
        self.min_box_height = int(self.min_box_height_percent * height / 100)
        self.min_total_motion_area = int(self.min_total_motion_area_percent * max_pixels / 100)

        # Day-specific contour rules
        self.min_contour_solidity = 0.70
        self.min_contour_area_ratio = 0.0045
        self.max_allowed_aspect_ratio = 6.0
        self.inflate_motion_boxes = 13
        self.motion_persistence_time = 2.0


class NightMotionProfile(MotionProfile):
    def __init__(
        self,
        width, height, max_pixels,
        yolo_confidence_threshold,
        motion_threshold,
        min_motion_confidence,
        min_motion_frames,
        min_sum_box_area
    ):
        super().__init__(
            width, height, max_pixels,
            yolo_confidence_threshold - 0.10,
            motion_threshold * 0.7,
            min_motion_confidence,
            min_motion_frames - 2,
            min_sum_box_area * 0.6
        )

        # Override instance attributes
        self.min_box_width_percent  = 1.2
        self.min_box_height_percent = 1.2
        self.min_total_motion_area_percent = 0.45  # 0.45%

        # Recompute pixel thresholds
        self.min_box_width  = int(self.min_box_width_percent  * width  / 100)
        self.min_box_height = int(self.min_box_height_percent * height / 100)
        self.min_total_motion_area = int(self.min_total_motion_area_percent * max_pixels / 100)

        # Night-specific contour rules
        self.min_contour_solidity = 0.55
        self.min_contour_area_ratio = 0.0025
        self.max_allowed_aspect_ratio = 5.0
        self.inflate_motion_boxes = 10
        self.motion_persistence_time = 3.0




