import fractions
import time
from logging import getLogger
from typing import List

import cv2
import numpy as np
from aiortc.mediastreams import VideoStreamTrack
from av import VideoFrame

from nvr.camera.camera import Camera

logger = getLogger("pynvr.webrtc")

class CameraTrack(VideoStreamTrack):
    """
    WebRTC track that streams a single camera's latest_frame.
    """
    kind = "video"

    def __init__(self, camera):
        super().__init__()
        self._camera = camera

    async def recv(self) -> VideoFrame:
        frame = self._camera.latest_frame

        if frame is None:
            # Provide a fallback frame so SDP negotiation succeeds
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = time.time_ns()
        video_frame.time_base = fractions.Fraction(1, 1_000_000_000)
        return video_frame

class MosaicTrack(VideoStreamTrack):
    """
    WebRTC track that streams a high-quality mosaic of multiple cameras.
    """
    kind = "video"

    def __init__(self, cameras: List[Camera], rows: int, cols: int):
        super().__init__()
        self._cameras = [camera for camera in cameras if camera.config.enabled]
        self.rows = rows
        self.cols = cols
        #self._max_cols = max_cols

        # 4K width, height computed dynamically to preserve 4:3 tiles
        self.MOSAIC_W = 3840

    async def recv(self) -> VideoFrame:
        # Collect frames
        frames = []
        for camera in self._cameras:
            frame = camera.latest_frame
            if frame is None:
                frame = np.zeros((480, 704, 3), dtype=np.uint8)
            frames.append(frame)

        if not frames:
            # No enabled cameras → return a black frame
            black = np.zeros((480, 704, 3), dtype=np.uint8)
            vf = VideoFrame.from_ndarray(black, format="bgr24")
            vf.pts = time.time_ns()
            vf.time_base = fractions.Fraction(1, 1_000_000_000)
            return vf
        
        total = len(frames)
        #cols = min(total, self._max_cols)
        #rows = int(np.ceil(total / cols))

        # Camera aspect ratio (704x480)
        CAM_ASPECT = 704 / 480

        # Tile width fixed by mosaic width
        TILE_W = self.MOSAIC_W // self.cols

        # Tile height computed to preserve 4:3
        TILE_H = int(TILE_W / CAM_ASPECT)

        # Pad with black tiles if needed
        needed = self.rows * self.cols
        while len(frames) < needed:
            frames.append(np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8))

        # Mosaic height computed from tile height
        self.MOSAIC_H = TILE_H * self.rows

        # Prepare mosaic canvas
        mosaic = np.zeros((self.MOSAIC_H, self.MOSAIC_W, 3), dtype=np.uint8)

        for idx, frame in enumerate(frames):
            src_h, src_w, _ = frame.shape

            # Compute tile grid position
            row = idx // self.cols
            col = idx % self.cols

            # Resize while preserving aspect ratio (no cropping)
            resized_frame = cv2.resize(
                frame,
                (TILE_W, TILE_H),
                interpolation=cv2.INTER_AREA if (src_w > TILE_W or src_h > TILE_H)
                else cv2.INTER_CUBIC
            )

            # Place tile
            y0 = row * TILE_H
            x0 = col * TILE_W
            mosaic[y0:y0+TILE_H, x0:x0+TILE_W] = resized_frame

        # Convert to VideoFrame
        video_frame = VideoFrame.from_ndarray(mosaic, format="bgr24")
        video_frame.pts = time.time_ns()
        video_frame.time_base = fractions.Fraction(1, 1_000_000_000)

        return video_frame

