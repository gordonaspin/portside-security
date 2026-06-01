import numpy as np
from numpy.typing import NDArray

class FrameBuffers:
    def __init__(self, config: dict, width: int, height: int):

        # Buffers (all start as None)
        self.background_buf: NDArray[np.float32]    = np.zeros((height, width), dtype=np.float32)
        self.bg_frame_buf: NDArray[np.uint8]        = np.zeros((height, width), dtype=np.uint8)
        self.diff_blur_buf: NDArray[np.uint8]       = np.zeros((height, width), dtype=np.uint8)
        self.diff_buf: NDArray[np.uint8]            = np.zeros((height, width), dtype=np.uint8)
        self.diff_mask_buf: NDArray[np.uint8]       = np.zeros((height, width), dtype=np.uint8)
        self.diff_filtered_buf: NDArray[np.uint8]   = np.zeros((height, width), dtype=np.uint8)
        self.edges_buf: NDArray[np.uint8]           = np.zeros((height, width), dtype=np.uint8)
        self.gray_buf: NDArray[np.uint8]            = np.zeros((height, width), dtype=np.uint8)
        self.thresh_buf: NDArray[np.uint8]          = np.zeros((height, width), dtype=np.uint8)
        self.sobel_x_buf: NDArray[np.int16]         = np.zeros((height, width), dtype=np.int16)
        self.sobel_y_buf: NDArray[np.int16]         = np.zeros((height, width), dtype=np.int16)
        self.sobel_x_abs_buf: NDArray[np.uint8]     = np.zeros((height, width), dtype=np.uint8)
        self.sobel_y_abs_buf: NDArray[np.uint8]     = np.zeros((height, width), dtype=np.uint8)

        # Full-res frame buffer (NumPy)
        self.full_frame = np.empty((height, width, 3), dtype=np.uint8)
        self.full_frame_bytes = memoryview(self.full_frame).cast('B')
        self.full_frame_size = height * width * 3

        # Read buffer for full-res pipe
        self.read_buf_full = bytearray(self.full_frame_size)
        self.read_view_full = memoryview(self.read_buf_full)

        # YOLO buffer (only if dual-pipe)
        if (width != config["model"]["resolution"]["width"] or height != config["model"]["resolution"]["height"]):
            self.yolo_frame = np.empty((config["model"]["resolution"]["height"], config["model"]["resolution"]["width"], 3), dtype=np.uint8)
            self.yolo_frame_bytes = memoryview(self.yolo_frame).cast('B')
            self.yolo_frame_size = config["model"]["resolution"]["height"] * config["model"]["resolution"]["width"] * 3

            self.read_buf_yolo = bytearray(self.yolo_frame_size)
            self.read_view_yolo = memoryview(self.read_buf_yolo)
        else:
            self.yolo_frame = None
            self.yolo_frame_bytes = None
            self.yolo_frame_size = None
            self.read_buf_yolo = None
            self.read_view_yolo = None