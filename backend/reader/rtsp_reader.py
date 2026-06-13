import os
import subprocess
import select
import time
from logging import getLogger
from queue import Queue, Empty
from threading import Event, Thread, current_thread
from typing import override

import cv2
import numpy as np
from numpy.typing import NDArray

from logger.logger import log_event
from nvr.camera.camera import Camera
from utils.utils import RollingAverage

logger = getLogger("pynvr.reader")


class Reader:
    def __init__(self):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def get_frame(self):
        pass


class RTSPReader(Reader):
    def __init__(
        self,
        camera: Camera,
        model_resolution: dict[str, int],
        produce_segments: bool,
        stop_event: Event,
    ):
        self.camera: Camera = camera
        self.model_width = model_resolution["width"]
        self.model_height = model_resolution["height"]
        self.produce_segments = produce_segments
        self.stop_event: Event = stop_event

        self.process: subprocess.Popen | None = None
        self.yolo_pipe: None | object = None  # file object for YOLO pipe
        self.yolo_fd_write: int | None = None
        self.log_filename: str | None = None
        self.log_file: object | None = None

        self.thread: Thread | None = None
        self.frame_queue: Queue[NDArray[np.uint8]] = Queue(maxsize=1)

        self.total_frames: int = 0
        self.total_drops: int = 0
        self.last_frame_time: float = 0.0
        self.fps: RollingAverage = RollingAverage()
        self.dt: RollingAverage = RollingAverage()

        # For corruption detection
        self._prev_full_mean: float = 0.0
        self._prev_full_frame: NDArray[np.uint8] | None = None

    @override
    def start(self):
        """
        Start autonomous reader thread.
        """
        if self.thread and self.thread.is_alive():
            return

        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        """
        Autonomous lifecycle:
        - open stream
        - read frames
        - on stall/error → stop and reopen
        - loop until stop_event is set
        """
        current_thread().name = f"{self.camera.config.name} RTSPReader"

        while not self.stop_event.is_set():
            try:
                self._open_stream()
            except Exception as e:
                log_event(
                    message=f"failed to open stream: {e}",
                    level="error",
                    camera=self.camera,
                )
                self._cleanup_process()
                time.sleep(2.0)
                continue

            # Reader loop: runs until stall, EOF, or stop_event
            self._frame_reader_loop()

            # Clean up process and pipes before next attempt
            self._cleanup_process()

            # Small backoff to avoid tight restart loops
            if not self.stop_event.is_set():
                time.sleep(1.0)

        logger.info(f"{self.camera.config.name} RTSPReader main loop exiting")

    def _cleanup_process(self):
        """
        Internal cleanup: terminate FFmpeg, close pipes/logs, reset camera state.
        """
        if self.process is not None:
            ret = self.process.poll()
            log_event(
                message=f"stopping RTSP reader with ret {ret}",
                level="info",
                camera=self.camera,
            )

            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                self.process.kill()

            if self.process.stdout:
                try:
                    self.process.stdout.close()
                except Exception:
                    pass

        if self.yolo_pipe is not None:
            try:
                self.yolo_pipe.close()
            except Exception:
                pass

        if self.log_file is not None:
            try:
                self.log_file.close()
            except Exception:
                pass

        self.process = None
        self.yolo_pipe = None
        self.yolo_fd_write = None
        self.log_file = None
        self.camera.first_frame = True

    @override
    def stop(self):
        """
        Public stop: signal thread and clean up process.
        """
        self.stop_event.set()
        self.thread.join()

    @override
    def get_frame(self) -> NDArray[np.uint8] | None:
        """
        Retrieve the latest frame from the camera queue.
        - latest-frame-wins
        - drop frames if queue is full
        - timeout every 0.5s so thread can exit cleanly
        """
        try:
            frame: NDArray[np.uint8] = self.frame_queue.get(timeout=0.5)
            return frame.copy()
        except Empty:
            return None

    def _open_stream(self):
        if self.stop_event.is_set():
            return

        # Determine if we need a second pipe for YOLO frames
        need_yolo_pipe = self._needs_yolo_pipe()

        # Segment file path (if enabled)
        filespec = (
            os.path.join(self.camera.config.segments_dir, "%Y%m%d_%H%M%S.ts")
            if self.produce_segments
            else None
        )

        # ------------------------------------------------------------
        # YOLO PIPE SETUP
        # ------------------------------------------------------------
        yolo_fd_read = None
        yolo_fd_write = None

        if need_yolo_pipe:
            yolo_fd_read, yolo_fd_write = self._make_yolo_pipe()
            self.yolo_fd_write = yolo_fd_write

        # ------------------------------------------------------------
        # BUILD FFMPEG COMMAND
        # ------------------------------------------------------------
        ffmpeg_cmd = self.build_ffmpeg_cmd(
            url=self.camera.config.url,
            filespec=filespec,
            segment_mode=self.produce_segments,
            yolo_fd=yolo_fd_write,
        )

        self.log_filename = os.path.join(
            self.camera.config.logs_dir,
            f"{self.camera.config.name}_ffmpeg.log",
        )
        self.log_file = open(self.log_filename, "a")
        for item in ffmpeg_cmd:
            if item.startswith("-"):
                self.log_file.write("\n")
            self.log_file.write(f"{item} ")
        self.log_file.write(
            f"\n--- Starting FFmpeg for {self.camera.config.name} "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
        )
        self.log_file.flush()

        if need_yolo_pipe:
            log_event(
                message="starting dual-pipe RTSP reader",
                level="info",
                camera=self.camera,
            )

            process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,      # actual-res frames
                stderr=self.log_file,             # FFmpeg logs
                pass_fds=(yolo_fd_write,),   # yolo-res frames on separate pipe
                bufsize=0,
            )

            # Parent does not write to YOLO pipe
            os.close(yolo_fd_write)
            self.yolo_fd_write = None

            # Open read-end for Python
            self.yolo_pipe = os.fdopen(yolo_fd_read, "rb", buffering=0)
        else:
            log_event(
                message="starting single-pipe RTSP reader",
                level="info",
                camera=self.camera,
            )

            process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,    # actual-res frames (also used for YOLO in single-pipe mode)
                stderr=self.log_file,             # FFmpeg logs
                bufsize=0,
            )

            self.yolo_pipe = None

        self.process = process

    def _frame_reader_loop(self):
        """
        Reads frames from FFmpeg until:
        - stall (repeated timeouts)
        - EOF / process exit
        - stop_event set
        Then returns to let _run() restart.
        """
        if self.process is None or self.process.stdout is None:
            return

        current_thread().name = f"{self.camera.config.name} frame reader"

        fail_count = 0

        while (
            not self.stop_event.is_set()
            and self.process is not None
            and self.process.stdout is not None
            and not self.process.stdout.closed
        ):

            # ------------------------------------------------------------
            # 1. Read FULL-RES frame (always required)
            # ------------------------------------------------------------
            raw_full = self._read_exact(
                self.process.stdout.fileno(),
                self.camera.buffers.read_view_full,
                self.camera.buffers.full_frame_size,
            )

            if raw_full is None:
                # Soft failure: timeout or short read
                fail_count += 1

                # If FFmpeg actually died, bail out and let restart logic handle it
                if self.process.poll() is not None:
                    log_event(
                        message="ffmpeg exited, breaking reader loop",
                        level="warn",
                        camera=self.camera,
                    )
                    break

                # Too many consecutive soft failures → restart
                if fail_count >= 10:
                    log_event(
                        message="full-res reader stalled, restarting after repeated timeouts",
                        level="warn",
                        camera=self.camera,
                    )
                    break

                continue

            # Got a good frame → reset failure counter
            fail_count = 0

            # Copy raw bytes into preallocated NumPy frame
            self.camera.buffers.full_frame_bytes[:] = raw_full
            full_frame = self.camera.buffers.full_frame

            # Optional corruption detection
            if self._looks_corrupted(full_frame):
                continue

            # ------------------------------------------------------------
            # 2. Read YOLO frame (only in dual-pipe mode)
            # ------------------------------------------------------------
            if (
                self.yolo_pipe is not None
                and self.camera.buffers.read_view_yolo is not None
            ):
                raw_yolo = self._read_exact(
                    self.yolo_pipe.fileno(),
                    self.camera.buffers.read_view_yolo,
                    self.camera.buffers.yolo_frame_size,
                )

                if raw_yolo is None:
                    # Soft failure: timeout or short read
                    fail_count += 1

                    # If FFmpeg actually died, bail out and let restart logic handle it
                    if self.process.poll() is not None:
                        log_event(
                            message="ffmpeg exited, breaking YOLO reader loop",
                            level="warn",
                            camera=self.camera,
                        )
                        break

                    # Too many consecutive soft failures → restart
                    if fail_count >= 10:
                        log_event(
                            message="yolo reader stalled, restarting after repeated timeouts",
                            level="warn",
                            camera=self.camera,
                        )
                        break

                    continue

                # Got a good frame → reset failure counter
                fail_count = 0

                self.camera.buffers.yolo_frame_bytes[:] = raw_yolo
                self.camera.buffers.yolo_frame = self.camera.buffers.yolo_frame
            else:
                # Single-pipe mode: YOLO frame == full frame
                self.camera.buffers.yolo_frame = full_frame

            # ------------------------------------------------------------
            # 3. FPS calculation
            # ------------------------------------------------------------
            now = time.perf_counter()
            if self.last_frame_time > 0:
                dt = now - self.last_frame_time
                if 0.02 < dt < 0.2:
                    self.dt.update(dt)
                    self.fps.update(1.0 / self.dt.value())
            self.last_frame_time = now

            # ------------------------------------------------------------
            # 4. Queue handling (latest-frame-wins)
            # ------------------------------------------------------------
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                    self.total_drops += 1
                except Exception:
                    pass

            self.frame_queue.put(full_frame)
            self.total_frames += 1
            self.drop_rate = self.total_drops / self.total_frames

        logger.info(f"{self.camera.config.name} RTSPReader loop exiting")

    def _read_exact(self, fd, view, size, timeout=2.0):
        """
        Reads exactly `size` bytes into the provided memoryview `view`.
        Returns:
            view[:size] on success
            None on timeout or EOF
        """

        total = 0
        deadline = time.time() + timeout

        while total < size:
            if time.time() > deadline:
                return None

            r, _, _ = select.select([fd], [], [], timeout)
            if not r:
                return None

            try:
                chunk = os.read(fd, size - total)
            except OSError:
                return None

            if not chunk:
                return None

            n = len(chunk)
            view[total : total + n] = chunk
            total += n

        return view[:size]

    def _looks_corrupted(self, frame: NDArray[np.uint8]) -> bool:
        """
        Lightweight corruption detector to skip obviously bad frames
        caused by partial reads or mixed-frame boundaries.
        """
        mean = float(frame.mean())

        # All-black or all-white frames are suspicious
        if mean < 1.0 or mean > 254.0:
            logger.warning(
                f"{self.camera.config.name} mean={mean:.2f} - Detected corrupted frame, dropping"
            )
            return True

        # Exposure jump / I-frame jump
        if hasattr(self, '_prev_mean') and abs(mean - self._prev_mean) > 80:
            logger.warning(f"{self.camera.config.name} exposure jump > 80 - Detected exposure jump, dropping")
            return True

        if hasattr(self, "_prev_frame") and self._prev_frame is not None:
            diff = cv2.absdiff(frame, self._prev_frame)
            if diff.mean() > 120.0:
                logger.warning(
                    f"{self.camera.config.name} diff.mean={diff.mean():.2f} - Detected corrupted frame, dropping"
                )
                return True

            if not hasattr(self, "_diff_buf_row"):
                self._diff_buf_row = np.zeros((frame.shape[1], 3), dtype=np.uint8)

            cv2.absdiff(frame[0], self._prev_frame[0], dst=self._diff_buf_row)
            continuity_diff = self._diff_buf_row.mean()
            if continuity_diff > 150:
                logger.warning(
                    f"{self.camera.config.name} continuity_diff.mean={continuity_diff:.2f} - Detected corrupted frame, dropping"
                )
                return True

        self._prev_mean = mean
        self._prev_frame = frame.copy()
        return False

    def build_ffmpeg_cmd(
        self,
        url: str,
        filespec: str | None,
        segment_mode: bool,
        yolo_fd: int | None,
    ) -> list[str]:
        actual_w = self.camera.width
        actual_h = self.camera.height
        yolo_w = self.model_width
        yolo_h = self.model_height

        need_yolo_pipe = not (actual_w == yolo_w and actual_h == yolo_h)

        if need_yolo_pipe and yolo_fd is None:
            raise ValueError("yolo_fd must be provided when YOLO pipe is required")

        full_block = actual_w * actual_h * 3
        yolo_block = yolo_w * yolo_h * 3

        base = [
            "ffmpeg",
            "-rtsp_transport",
            "tcp",
            "-max_delay", "0",
            "-fflags",
            "nobuffer+genpts+discardcorrupt",
            "-flags",
            "low_delay",
            "-avioflags",
            "direct",
            "-i",
            url,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostats",
        ]

        filters = []
        filters.append("[0:v]format=bgr24[full]")

        if need_yolo_pipe:
            if actual_w == yolo_w and actual_h == yolo_h:
                filters.append("[0:v]format=bgr24[yolo]")
            else:
                filters.append(f"[0:v]scale={yolo_w}:{yolo_h},format=bgr24[yolo]")

        filter_graph = ";".join(filters)

        cmd = base + [
            "-filter_complex",
            filter_graph,
            "-map",
            "[full]",
            "-f",
            "rawvideo",
            "-blocksize",
            str(full_block),
            "pipe:1",
        ]

        if filespec is not None:
            cmd += [
                "-map",
                "0:v",
                "-c:v",
                "copy",
                "-f",
                "segment",
                "-segment_time",
                "1",
                "-reset_timestamps",
                "0",
                "-strftime",
                "1",
                "-segment_format",
                "mpegts",
                filespec,
            ]

        if need_yolo_pipe:
            cmd += [
                "-map",
                "[yolo]",
                "-f",
                "rawvideo",
                "-blocksize",
                str(yolo_block),
                f"pipe:{yolo_fd}",
            ]

        return cmd

    def _needs_yolo_pipe(self) -> bool:
        return not (
            self.camera.width == self.model_width
            and self.camera.height == self.model_height
        )

    def _make_yolo_pipe(self) -> tuple[int, int]:
        read_fd, write_fd = os.pipe()
        logger.debug(
            f"{self.camera.config.name} created YOLO pipe "
            f"read fd={read_fd}, write fd={write_fd}"
        )
        return read_fd, write_fd
