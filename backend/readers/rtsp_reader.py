import os
import glob
import subprocess
from logging import getLogger
from queue import Queue, Empty
from threading import Event, Thread, current_thread
import time
from typing import override
from collections import deque, defaultdict


import numpy as np
from numpy.typing import NDArray

from camera.camera import Camera
from logger.logger import log_event
from utils.thread_safe import ThreadSafeSet
import constants

logger = getLogger("pynvr")

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


class SegmentCleaner():
    do_not_delete_set: ThreadSafeSet = ThreadSafeSet()

    def __init__(
        self,
        ):
        self.stop_event: Event = None
        self.segments_dir: list[str] = []
        self.cleaner_thread = Thread(target=self._cleanup_segments,daemon=True)

    def add_folder(self, folder: str):
        self.segments_dir.append(folder)

    def _cleanup_segments(self):
        """
        Thread that periodically deletes old segment files for all RTSPReaders
        """
        current_thread().name = "cleanup_segments"
            
        while True and not self.stop_event.is_set():
            try:
                for folder in self.segments_dir:
                    path = os.path.join(folder, "*.ts")
                    files = sorted(glob.glob(path))
                    if len(files) > constants.TS_FILE_RING_SECONDS:
                        for f in files[:-constants.TS_FILE_RING_SECONDS]:
                            if f not in SegmentCleaner.do_not_delete_set:
                                try: os.remove(f)
                                except: pass
                time.sleep(2)
            except Exception as e:
                log_event(message=f"exception in cleanup_segments {e}", level="error")


class Reader():
    def __init__(self):
        pass

    def start():
        pass

    def get_frame():
        pass

class RTSPReader(Reader):

    segment_cleaner: SegmentCleaner = SegmentCleaner()

    def __init__(
        self,
        camera: Camera,
        stop_event: Event,
    ):
        self.camera: Camera = camera
        self.process: subprocess.Popen = None
        self.stop_event: Event = stop_event
        self.reader_thread: Thread = None
        self.frame_queue: Queue[NDArray[np.uint8]] = Queue(maxsize=1)
        self.total_frames: int = 0
        self.total_drops: int = 0
        self.last_frame_time: float = 0.0
        self.fps: RollingAverage = RollingAverage(100)
        self.dt: RollingAverage = RollingAverage(100)

        if RTSPReader.segment_cleaner.stop_event is None:
            RTSPReader.segment_cleaner.stop_event = stop_event
            RTSPReader.segment_cleaner.cleaner_thread.start()
        RTSPReader.segment_cleaner.add_folder(self.camera.config.segments_dir)

    @override
    def start(self):
        self._open_stream()
        self.reader_thread = Thread(target=self._frame_reader, daemon=True)
        self.reader_thread.start()

    @override
    def get_frame(self) -> NDArray[np.uint8] | None:
        """
        Retrieve the latest frame from the camera queue.
        This preserves the original behavior:
        - latest-frame-wins
        - drop frames if queue is full
        - timeout every 0.5s so thread can exit cleanly
        """
        try:
            frame: NDArray[np.uint8] = self.frame_queue.get(timeout=0.5)
            self.camera.buffers.initialize(frame)
            return frame.copy()  # always work on a copy
        except Empty:
            return None

    def _open_stream(self):
        """
        Starts ffmpeg as a subprocess reading from the camera RTSP stream. The stream is split
        in two writing simultaneously to segment files and stdout. No re-encoding happens to the
        segment files. The frames written to stdout are resized for image processing by cv2. 
        """
        if not self.stop_event.is_set():
            log_event(message=f"starting camera", level="info", camera=self.camera)
            filespec = os.path.join(self.camera.config.segments_dir, "%Y%m%d_%H%M%S.ts")
            ffmpeg_cmd = [
                "ffmpeg",

                "-rtsp_transport", "tcp",           # Forces RTSP over TCP instead of UDP
                "-fflags", "nobuffer+genpts",       # Disables internal buffering, generates PTS
                "-flags", "low_delay",              # Tells decoder/demuxer to minimize delay (Reduces frame reordering buffers)
                "-i", self.camera.config.url,                   # RTSP stream from camera
                "-hide_banner",
                "-loglevel", "error",               # ONLY errors (no frame spam)
                "-nostats",
                
                "-filter_complex",                  # Split and reduce scale for raw only for OpenCV
                f"[0:v]scale={self.camera.config.width}:{self.camera.config.height},format=bgr24[raw]", # re-scale and raw BGR pixel format (OpenCV native)

                # ---- TS segments (NO RE-ENCODE) ----
                "-map", "0:v",                      # original stream, unaltered
                "-c", "copy",                       # No re-encoding (copy stream)
                "-f", "segment",                    # enable segment muxer
                "-segment_time", "1",               # target segment length 1 second
                "-reset_timestamps", "0",           # don't reset timestamps
                "-strftime", "1",                   # enable timestamp based filenames
                "-segment_format", "mpegts",        # force mpeg-ts container
                filespec,

                # ---- Raw frames (OpenCV) ----
                "-map", "[raw]",                    # selects filtered (scaled + BGR) stream
                "-f", "rawvideo",                   # outputs raw uncompressed frames
                "pipe:1"                            # sends raw bytes to stdout
            ]
            process =  subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0
            )
            self.process = process

    def _frame_reader(self):
        """
        Thread to read frames from the ffmpeg stdout stream and puts the frame on the camera queue.
        The queue length is 1, so if the queue is full that frame on the queue is dropped and
        replaced with the new frame. This means we drop frames to keep up. This is only for
        image processing, frames written to segments are not dropped
        """
        current_thread().name = f"{self.camera.config.name} _frame_reader"

        frame_size = self.camera.config.width * self.camera.config.height * 3
        fail_count = 0

        while not self.stop_event.is_set() and not self.process.stdout.closed:
            raw = self._read_exact(self.process.stdout, frame_size)

            if raw is None:
                log_event(message="reader failed", level="warn", camera=self.camera)
                if fail_count < 3:
                    fail_count += 1
                    self.restart()
                else:
                    log_event(message="stopping camera, too many failures, giving up", level="warn", camera=self.camera)
                    self.stop()
                continue

            frame = np.frombuffer(raw, np.uint8).reshape((self.camera.config.height, self.camera.config.width, 3))

            # FPS calculation
            now = time.perf_counter()
            if self.last_frame_time > 0:
                dt = now - self.last_frame_time

                # filter pipeline artifacts
                if 0.02 < dt < 0.2:
                    inst_fps = 1.0 / dt
                    self.dt.update(dt)
                    self.fps.update(1.0 / self.dt.value())

            self.last_frame_time = now

            # latest-frame-wins
            if self.frame_queue.full():
                self.frame_queue.get_nowait()
                self.total_drops += 1
            self.frame_queue.put(frame)
            self.total_frames += 1
            self.drop_rate = self.total_drops / self.total_frames

    def restart(self):
        """
        Stop and start unless we are shutting down
        """
        if not self.stop_event.is_set():
            log_event(message="restarting RTSP reader", level="warn", camera=self.camera)
            self.stop()
            self.start()

    def stop(self):
        """
        Stops the background ffmpeg process for the camera, closes pipes and resets the camera
        """
        if self.process is not None:
            ret = self.process.poll()
            log_event(message=f"stopping RTSP reader with ret {ret}", level="info", camera=self.camera)
            self.process.terminate()

            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process.stdout.close()
            self.reader_thread.join(timeout=2)
            self.camera.first_frame = True

    def _read_exact(self, pipe, size):
        """
        reads bytes from the pipe until the buffer size is reached
        """
        buf = b""
        while len(buf) < size:
            chunk = pipe.read(size - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf
