import numpy as np
from numpy.typing import NDArray

class FrameBuffers:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.initialized = False

        # Buffers (all start as None)
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

    def initialize(self, frame_bgr: NDArray[np.uint8]):
        if self.initialized:
            return

        h, w = frame_bgr.shape[:2]

        self.bg_frame_buf      = np.zeros((h, w), dtype=np.uint8)
        self.diff_blur_buf     = np.zeros((h, w), dtype=np.uint8)
        self.diff_buf          = np.zeros((h, w), dtype=np.uint8)
        self.diff_filtered_buf = np.zeros((h, w), dtype=np.uint8)
        self.diff_mask_buf     = np.zeros((h, w), dtype=np.uint8)
        self.edges_buf         = np.zeros((h, w), dtype=np.uint8)
        self.gray_buf          = np.zeros((h, w), dtype=np.uint8)
        self.thresh_buf        = np.zeros((h, w), dtype=np.uint8)
        self.sobel_x_buf       = np.zeros((h, w), dtype=np.int16)
        self.sobel_y_buf       = np.zeros((h, w), dtype=np.int16)
        self.sobel_x_abs_buf   = np.zeros((h, w), dtype=np.uint8)
        self.sobel_y_abs_buf   = np.zeros((h, w), dtype=np.uint8)

        # Background model starts as float32 version of first gray frame
        self.background_buf = self.gray_buf.astype("float32")

        self.initialized = True