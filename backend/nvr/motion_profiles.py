class MotionProfile:
    def __init__(self, max_pixels: int):
        self.max_pixels = max_pixels

    def min_edge_density(self, noise: float):
        return 0.015 + (noise * 0.001)
           

class DayMotionProfile(MotionProfile):
    def __init__(self, max_pixels, motion_threshold, yolo_confidence_threshold):
        super().__init__(max_pixels)
        self.yolo_confidence_threshold = yolo_confidence_threshold
        self.motion_threshold = int(motion_threshold * max_pixels / 100)
        self.min_box_width = 20
        self.min_box_height = 20
        self.min_contour_solidity = 0.60
        self.min_contour_area_ratio = 0.0025
        self.max_allowed_aspect_ratio = 6.0
        self.motion_confidence_min = 0.35
        self.min_total_motion_area = 0.003 * max_pixels
        self.min_sum_box_area = 0.005 * max_pixels
        self.inflate_motion_boxes = 30
        self.motion_persistence_time = 2.0   # seconds
    def set_motion_threshold(self, val):
            self.motion_threshold = int(val * self.max_pixels / 100)

    def set_yolo_confidence_threshold(self, val):
            self.yolo_confidence_threshold = val

class NightMotionProfile(MotionProfile):
    def __init__(self, max_pixels, motion_threshold, yolo_confidence_threshold):
        super().__init__(max_pixels)
        self.yolo_confidence_threshold = min(0.6, yolo_confidence_threshold + 0.15)     # stricter YOLO
        self.motion_threshold = int(motion_threshold * 1.2 * max_pixels / 100)          # 20% higher
        self.min_box_width = 15
        self.min_box_height = 15
        self.min_contour_solidity = 0.70                                                # stricter
        self.min_contour_area_ratio = 0.005                                             # larger min area of a single contour
        self.max_allowed_aspect_ratio = 4.0                                             # keeps square-ish objects, discards skinny long ones
        self.motion_confidence_min = 0.50                                               # require stronger motion
        self.min_total_motion_area = 0.005 * max_pixels                                 # require more total motion
        self.min_sum_box_area = 0.003 * max_pixels                                      # require sum of box areas to be larger
        self.inflate_motion_boxes = 40
        self.motion_persistence_time = 2.0   # seconds
          
    def set_motion_threshold(self, val):
            self.motion_threshold = int(val * 1.2 * self.max_pixels / 100)

    def set_yolo_confidence_threshold(self, val):
            self.yolo_confidence_threshold = min(0.6, val + 0.15)
