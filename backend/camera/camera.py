from collections import deque, defaultdict
from queue import Queue
from subprocess import Popen
import numpy as np
from numpy.typing import NDArray
from model.model import Model
from nvr.motion_profiles import MotionProfile, DayMotionProfile, MotionProfileAutoTuner

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
        recordings_dir: str,
        segments_dir: str,
        images_dir: str,
        metadata_dir: str,
        plates_dir: str,
        model: Model,
        debug: bool = False
    ):

        # --- Basic config ---
        self.cfg: dict = cfg
        self.max_pixels = max_pixels
        self.name: str = name
        self.recordings_dir: str = recordings_dir
        self.segments_dir: str = segments_dir
        self.images_dir: str = images_dir
        self.metadata_dir: str = metadata_dir
        self.plates_dir: str = plates_dir
        self.model: Model = model
        self.debug: bool = debug

        # --- Stream state ---
        self.process: Popen | None = None
        self.first_frame: bool = True
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
        self.last_recording_time: float = 0.0
        self.last_night_time_check: float = 0.0
        self.last_motion_time: float = 0.0
        self.should_record: bool = False
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
        self.profile: MotionProfile = DayMotionProfile(
            max_pixels=self.max_pixels,
            motion_threshold=cfg.get("motion_threshold", 1.0),
            yolo_confidence_threshold=cfg.get("yolo_confidence", 0.25)
        )

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

    def is_lpr(self) -> bool:
        return hasattr(self, "lpr")

    def auto_adjust_profile(self):
        tuner = self.auto_tuner
        stats = tuner.summarize()
        recs  = tuner.recommend_adjustments()

        prof = self.profile
        ap   = self.adaptive_profile

        # --- 1. Adjust min_edge_density ---
        if "min_edge_density" in recs:
            ap["min_edge_density"] += 0.002
            ap["min_edge_density"] = min(ap["min_edge_density"], 0.08)  # safety cap

        # --- 2. Adjust min_contour_area_ratio ---
        if "min_contour_area_ratio" in recs:
            ap["min_contour_area_ratio"] += 0.0005
            ap["min_contour_area_ratio"] = min(ap["min_contour_area_ratio"], 0.02)

        # --- 3. Adjust min_total_motion_area ---
        if "min_total_motion_area" in recs:
            ap["min_total_motion_area"] += 0.001 * self.max_pixels
            ap["min_total_motion_area"] = min(ap["min_total_motion_area"], 0.05 * self.max_pixels)

        # --- 4. Adjust min_motion_frames ---
        if "min_motion_frames" in recs:
            ap["min_motion_frames"] += 2
            ap["min_motion_frames"] = min(ap["min_motion_frames"], 20)

        # --- 5. Adjust inflate_motion_boxes ---
        if "inflate_motion_boxes" in recs:
            ap["inflate_motion_boxes"] -= 5
            ap["inflate_motion_boxes"] = max(ap["inflate_motion_boxes"], 5)

        # --- APPLY ADAPTIVE VALUES BACK TO PROFILE ---
        prof.min_contour_area_ratio = ap["min_contour_area_ratio"]
        prof.min_total_motion_area  = ap["min_total_motion_area"]
        prof.min_motion_frames      = ap["min_motion_frames"]
        prof.inflate_motion_boxes   = ap["inflate_motion_boxes"]

        # min_edge_density is a function → wrap it
        base = ap["min_edge_density"]
        prof.min_edge_density = lambda noise: base + noise * 0.0012

        # Reset tuner stats periodically
        tuner.decisions.clear()
        tuner.stats.clear()

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

