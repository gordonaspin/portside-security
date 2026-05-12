from collections import deque, defaultdict
import dataclasses
from queue import Queue
from subprocess import Popen
import time
import numpy as np
from numpy.typing import NDArray

from nvr.model import Model
from nvr.motion_profile import MotionProfile, DayMotionProfile, NightMotionProfile

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

@dataclasses.dataclass
class Camera:
    name: str
    url: str
    enabled: bool
    recordings_dir: str
    segments_dir: str
    images_dir: str
    metadata_dir: str
    model: Model
    debug: bool = False

    # stream state
    process: Popen = None
    first_frame: bool = True

    # latest-frame-wins buffer
    latest_frame: np.ndarray = None
    frame_queue: Queue = dataclasses.field(default_factory=lambda: Queue(maxsize=1))

    # buffers for cv2 frames
    background_buf: NDArray[np.float32] = None
    bg_frame_buf: NDArray[np.uint8] = None
    diff_blur_buf: NDArray[np.uint8] = None
    diff_buf: NDArray[np.uint8] = None
    diff_mask_buf: NDArray[np.uint8] = None
    diff_filtered_buf: NDArray[np.uint8] = None
    edges_buf: NDArray[np.uint8] = None
    gray_buf: NDArray[np.uint8] = None
    thresh_buf: NDArray[np.uint8] = None
    sobel_x_buf: NDArray[np.int16] = None
    sobel_y_buf: NDArray[np.int16] = None
    sobel_x_abs_buf: NDArray[np.int16] = None
    sobel_y_abs_buf: NDArray[np.int16] = None
    # FPS tracking
    total_frames: int = 0
    total_drops: int = 0
    dt: RollingAverage = dataclasses.field(default_factory=lambda: RollingAverage(100))
    fps: RollingAverage = dataclasses.field(default_factory=lambda: RollingAverage(100))
    drop_rate: float = 0.0
    last_frame_time: float = 0.0

    # UI / metadata
    status_text: str = "Not streaming"
    objects_text: str = ""

    # logic state
    last_recording_time: float = 0.0
    last_night_time_check: float = 0.0
    last_motion_time: float = 0.0
    should_record: bool = False
    recording: bool = False
    is_night: bool = False

    # profiles
    profile: MotionProfile = None

    # motion detection
    noise: float = 0.0
    motion_boxes_list: list = dataclasses.field(default_factory=list)
    classes_in_frame_dict: defaultdict = dataclasses.field(default_factory=lambda: defaultdict(set))
    active_objects_dict: defaultdict = dataclasses.field(default_factory=lambda: defaultdict(set))
    active_segments_list: list = dataclasses.field(default_factory=list)
    motion_condfidence: float = 0.0
    debug_motion_image: np.ndarray = None
    keep_mask: list = dataclasses.field(default_factory=list)
    has_moving_object: bool = False

