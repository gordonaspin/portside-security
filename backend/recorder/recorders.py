from turtle import fd

import av
import json
import os
import struct
import subprocess
import tempfile
import threading
import time
import traceback
import uuid
from collections import defaultdict, deque
from copy import deepcopy
from datetime import timedelta
from fractions import Fraction
from logging import getLogger
from typing import override

import cv2
from paddle.device import stream

from constants import PRE_RECORD_DURATION, TS_FILE_RING_SECONDS

from logger.logger import log_event
from nvr.camera.camera import Camera
from nvr.file_cleaner import FileCleaner
from utils.utils import make_readable_ts, make_ts_string, tags_to_str, make_ts_string_precise, RollingAverage

logger = getLogger("pynvr.recorder")

class Recorder:
    def __init__(self, camera: Camera, pre_record_duration=PRE_RECORD_DURATION):
        self.camera: Camera = camera
        self.pre_record_duration: float = pre_record_duration
        self.max_pre_frames: int = int(20 * pre_record_duration)
        self.uuid: str = str(uuid.uuid4())
        # Rolling buffer (automatically discards frames past the time limit)
        self.rolling_buffer: deque = deque(maxlen=self.max_pre_frames)
        
        # Recording queue (unlimited size, used during active recording)
        self.record_queue: deque = deque()
        
        self.last_frame_time: float = time.time()
        # Thread management states
        self.event: threading.Event = threading.Event()
        self.lock: threading.Lock = threading.Lock()
        self.thread: threading.Thread | None = None

        self.temporary_media_filename: str = None
        self.temporary_log_filename: str = None

        self.final_fps: int = None
        self.final_tags: dict | None = None
        self.final_tags_str: str = None
        self.final_timestamp_tags_str: str = None
        self.final_start_time: float | None = None
        self.final_end_time: float | None = None
        self.final_media_filename: str = None
        self.final_log_filename: str = None
        self.final_metadata_filename: str = None
        self.final_timestamp_name_tags: str = None

        self.fps: RollingAverage = RollingAverage(100)
    
    def should_add_frame(self):
        return True
    
    def add_frame(self, frame):
        now = time.time()
        if now - self.last_frame_time > 0:
            self.fps.update(1 / (now - self.last_frame_time))
        self.last_frame_time = now

        """Call this inside your main loop for every single incoming frame."""
        with self.lock:
            # Always make a copy to prevent OpenCV reference overwrites
            copied_frame = frame.copy()
            
            # Keep updating the rolling pre-buffer history
            now = time.monotonic_ns() // 1000  # Convert to microseconds for better precision in ffmpeg timestamps
  # microseconds
            precise = make_ts_string_precise(now)
            self.rolling_buffer.append((now, copied_frame))
            
            # If actively recording, append subsequent frames to the stream
            if self.camera.recording_state.recording:
                self.record_queue.append((now, copied_frame))


    def can_start(self):
        with self.lock:
            if len(self.rolling_buffer) == 0:
                return False
        return True
    
    def start_recording(self):
        if not self.can_start():
            return

        """ gather metadata and setup for recording, then spawn the recording thread """
        self.final_start_time = self.camera.recording_state.recording_start_time - self.pre_record_duration
        timestamp_str = make_ts_string(self.final_start_time)
        self.final_fps = self.fps.as_int()
        self.final_tags = deepcopy(self.camera.motion.active_objects_dict)
        self.final_tags_str = tags_to_str(self.final_tags)
        self.final_timestamp_tags_str = timestamp_str + "_" + self.final_tags_str
        self.final_media_filename = os.path.join(self.camera.config.recordings_dir, self.final_timestamp_tags_str + ".mp4")
        self.final_metadata_filename = os.path.join(self.camera.config.metadata_dir, self.final_timestamp_tags_str + ".json")
        self.final_log_filename = os.path.join(self.camera.config.logs_dir, self.final_timestamp_tags_str + ".log")
        self.final_profile = self.camera.motion.profile_to_dict()
        self.final_tuner_stats = self.camera.tuner.tuner.summarize()
        self.final_tuner_recs = self.camera.tuner.tuner.recommend_adjustments()

        self.temporary_media_filename  = tempfile.NamedTemporaryFile(
            "w+b",
            dir=self.camera.config.recordings_dir,
            suffix=".mp4",
            delete=False).name
        
        self.temporary_log_filename = self.temporary_media_filename + ".log"

        self.seed_buffer()
        self.start_thread()

    def seed_buffer(self):
        # Seed the recording queue with a snapshot of the current pre-buffer
        self.record_queue = deque(list(self.rolling_buffer))
            
    def start_thread(self):
        # Spawn the thread
        self.thread = threading.Thread(
            target=self._async_writer_worker, 
            daemon=False,  # Not daemon because we want to guarantee it finishes writing
        )
        self.thread.start()
        logger.debug(f"recorder started")

    def stop_recording(self):
        """Signals the recording to stop. Main thread can keep running."""
        self.event.set()

        self.final_end_time = time.time()

        # We join the thread to guarantee the file is written and safe on disk
        if self.thread:
            self.thread.join()

        self._finalize_files()

    def _get_additional_metadata(self):
        return {}

    def _finalize_files(self):
        self._finalize_media_file()
        self._finalize_log_file()

        cap = cv2.VideoCapture(self.final_media_filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = int(self.fps.as_int())

        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        duration_seconds = 0
        if frame_count and fps:
            duration_seconds = frame_count / fps

        self.formatted_duration = str(timedelta(seconds=int(duration_seconds)))

        metadata = self._create_metadata()

        with open(self.final_metadata_filename, "w") as f:
            json.dump(metadata, f, default=lambda o: o.__dict__, indent=4)

        self.report_complete()

    def report_complete(self):
        log_event(message=f"recording available {self.formatted_duration}", level="record", camera=self.camera, file_path=self.final_metadata_filename)

    def _finalize_media_file(self):
        os.rename(self.temporary_media_filename, self.final_media_filename)

    def _finalize_log_file(self):
        os.rename(self.temporary_log_filename, self.final_log_filename)

    def _create_metadata(self):
        # Convert to a standard dict and sets to lists
        serializable_tags = {k: list(v) for k, v in self.final_tags.items()}


        json_data = {
            "camera": self.camera.config.name,
            "fps": self.final_fps,
            "tags": serializable_tags,
            "media_filename": self.final_media_filename,
            "log_filename": self.final_log_filename,
            "start_time": self.final_start_time,
            "end_time": self.final_end_time,
            "start_fmt": make_readable_ts(self.final_start_time),
            "end_fmt": make_readable_ts(self.final_end_time),
            "metadata_filename": self.final_metadata_filename,
            "profile": self.final_profile,
            "tuner_stats": self.final_tuner_stats,
            "recommendations": self.final_tuner_recs,
        }
        return json_data | self._get_additional_metadata()


class OpenCVFrameRecorder(Recorder):
    def __init__(self, camera: Camera, pre_record_duration=PRE_RECORD_DURATION):
        super().__init__(camera=camera, pre_record_duration=pre_record_duration)
        logger.debug(f"{self.camera.config.name} initialized OpenCVFrameRecorder with pre-record duration {pre_record_duration}s and max pre-frames {self.max_pre_frames}")


    def _async_writer_worker(self):
        """Background thread that continuously drains the queue and writes to disk."""

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            self.temporary_media_filename,
            fourcc,
            self.fps.as_int(),
            (self.camera.config.width, self.camera.config.height)
            )
        
        try:
            while True:
                frame = None
                with self.lock:
                    # Check if there are frames ready to be written
                    if len(self.record_queue) > 0:
                        timestamp_μs, frame = self.record_queue.popleft()
                    # If queue is empty AND not recording, we are completely done
                    elif self.event.is_set() and len(self.record_queue) == 0:
                        break
                
                if frame is not None:
                    # High CPU overhead compression happens here safely on the background thread
                    video_writer.write(frame)
                else:
                    # Queue is momentarily empty, yield CPU execution to the main thread
                    time.sleep(0.01)
        finally:
            video_writer.release()

    @override
    def _finalize_media_file(self):
        """Moves the moov atom to the front of an MP4 file for browser streaming."""
        # Define the FFmpeg command as a list of strings
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            self.temporary_media_filename,
            "-c:v",
            "libx264",  # Explicitly convert the mp4v video to H.264
            "-c:a",
            "copy",  # Copy audio if present without changes
            "-movflags",
            "+faststart",  # Move metadata to the front for streaming
            self.final_media_filename,
        ]

        try:
            log_file = open(self.temporary_log_filename, "w")
            # Launch the process safely using a context manager
            with subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=log_file, text=True
            ) as process:
                # Wait for completion and capture outputs
                process.communicate()
                if process.returncode != 0:
                    logger.debug(f"ffmpeg Error in OpenCVFrameRecorder")
                else:
                    logger.debug("video is ready for web streaming")
                os.remove(self.temporary_media_filename)

        except FileNotFoundError:
            logger.error("ffmpeg is not installed or not in your system PATH.")

        finally:
            log_file.close()

    @override
    def start_recording(self):
        logger.debug(f"{self.camera.config.name} cv2 frame recorder started")
        super().start_recording()  # This will set up the filename and log_filename

    @override
    def _get_additional_metadata(self):
        return {
            "recorder_type": "OpenCV frame",
        }
    
    @override
    def report_complete(self):
        log_event(message=f"OpenCV frame recording available {self.formatted_duration}", level="record", camera=self.camera, file_path=self.final_metadata_filename)

class AVFFmpegFrameRecorder(Recorder):
    def __init__(self, camera: Camera, pre_record_duration=PRE_RECORD_DURATION):
        super().__init__(camera=camera, pre_record_duration=pre_record_duration)
        logger.debug(f"{self.camera.config.name} initialized AVFFmpegFrameRecorder with pre-record duration {pre_record_duration}s and max pre-frames {self.max_pre_frames}")


    def _async_writer_worker(self):
        try:
            #log_file = open(self.temporary_log_filename, "w")
            output = av.open(
                str(self.temporary_media_filename),
                mode="w",
                format="mp4",
                options={
                    "movflags": "faststart",
                }
            )
            stream = output.add_stream("libx264") # , rate=self.fps.as_int())
            stream.options = {
                "preset": "ultrafast",
                "tune": "zerolatency",
            }
            stream.width = self.camera.config.width
            stream.height = self.camera.config.height
            stream.pix_fmt = "yuv420p"
            stream.codec_context.max_b_frames = 0

            first_pts_μs = None
            frame_number = 0
            timebase = Fraction(1, 1_000_000)
            while True:
                frame = None
                with self.lock:
                    if len(self.record_queue) > 0:
                        pts_μs, frame = self.record_queue.popleft()
                    elif self.event.is_set() and len(self.record_queue) == 0:
                        break

                if frame is None:
                    time.sleep(0.01)
                    continue

                if first_pts_μs is None:
                    first_pts_μs = pts_μs

                pts_μs -= first_pts_μs  # Normalize to start at 0

                video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
                video_frame.pts = pts_μs
                video_frame.time_base = timebase
                frame_number += 1
                for packet in stream.encode(video_frame):
                    output.mux(packet)

            for packet in stream.encode():
                output.mux(packet)

            output.close()

        except Exception as e:
            logger.error(f"AVError in AVFFmpegFrameRecorder: {e}")
            traceback.print_exc()

    @override
    def _finalize_log_file(self):
        # For ffmpeg frame recorder, we write logs directly in the recording thread, so no need to rename
        pass

    @override
    def start_recording(self):
        logger.debug(f"{self.camera.config.name} av frame recorder started")
        super().start_recording()  # This will set up the filename and log_filename

    @override
    def _get_additional_metadata(self):
        return {
            "recorder_type": "AVFFmpeg frame",
        }

    @override
    def report_complete(self):
        log_event(message=f"AVFFmpeg recording available {self.formatted_duration}", level="record", camera=self.camera, file_path=self.final_metadata_filename)


class FFmpegFrameRecorder(Recorder):
    def __init__(self, camera: Camera, pre_record_duration=PRE_RECORD_DURATION):
        super().__init__(camera=camera, pre_record_duration=pre_record_duration)
        logger.debug(f"{self.camera.config.name} initialized FFmpegFrameRecorder with pre-record duration {pre_record_duration}s and max pre-frames {self.max_pre_frames}")

    def _async_writer_worker(self):
        """Background thread that continuously drains the queue and writes to disk."""
        command = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",       # Matches OpenCV format
            "-s", f"{self.camera.config.width}x{self.camera.config.height}",
            "-r", f"{self.fps.as_int()}",            # Framerate
            "-i", "-",                 # Input from Python pipe
            
            # --- Compression & Compatibility Settings ---
            "-c:v", "libx264",         # H.264 codec (universal browser support)
            "-pix_fmt", "yuv420p",     # Crucial for browser playback
            "-crf", "23",              # Balanced quality vs file size (lower = higher quality)
            "-preset", "medium",       # Balanced encoding speed vs file size
            
            # --- Browser Optimization ---
            "-movflags", "+faststart", # Puts index at front for instant browser playback
            self.temporary_media_filename
        ]
        
        try:
            log_file = open(self.temporary_log_filename, "w")

            # Run the process
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=log_file,
                )
            
            first_pts_μs = 0
            frame_number = 0
            delta_μs = 0
            prev_pts_μs = 0
            while True:
                frame = None
                with self.lock:
                    if len(self.record_queue) > 0:
                        pts_μs, frame = self.record_queue.popleft()
                    elif self.event.is_set() and len(self.record_queue) == 0:
                        break

                if frame is None:
                    time.sleep(0.01)
                    continue

                if first_pts_μs == 0:
                    first_pts_μs = pts_μs

                pts_μs -= first_pts_μs
                delta_μs = pts_μs - prev_pts_μs
                prev_pts_μs = pts_μs

                #logger.debug(f"FfmpegFrameRecorder writing frame {frame_number} with pts {pts_μs} delta {delta_μs} to ffmpeg")
                sleep_time = (delta_μs / 1_000_000) * 0.9
                if sleep_time > 0:
                    time.sleep(sleep_time)
                process.stdin.write(frame.tobytes())
                frame_number += 1

        except Exception as e:
            pass
        finally:
            log_file.close()
            process.stdin.close()
            process.wait()

    @override
    def _finalize_log_file(self):
        # For ffmpeg frame recorder, we write logs directly in the recording thread, so no need to rename
        pass

    @override
    def start_recording(self):
        logger.debug(f"{self.camera.config.name} ffmpeg frame recorder started")
        super().start_recording()  # This will set up the filename and log_filename

    @override
    def _get_additional_metadata(self):
        return {
            "recorder_type": "FFmpeg frame",
        }

    @override
    def report_complete(self):
        log_event(message=f"FFmpeg frame recording available {self.formatted_duration}", level="record", camera=self.camera, file_path=self.final_metadata_filename)


class FFmpegSegmentRecorder(Recorder):
    def __init__(self, camera: Camera, pre_record_duration=PRE_RECORD_DURATION):
        super().__init__(camera=camera, pre_record_duration=pre_record_duration)
        self.segments: list[str] = []
        FileCleaner.add(self.camera.config.segments_dir, "*.ts", timedelta(seconds=TS_FILE_RING_SECONDS), timedelta(seconds=5))
        FileCleaner.add(self.camera.config.segments_dir, "*.list", timedelta(seconds=TS_FILE_RING_SECONDS), timedelta(seconds=5))
        logger.debug(f"{self.camera.config.name} initialized FfmpegSegmentRecorder with pre-record duration {pre_record_duration}s and max pre-frames {self.max_pre_frames}")

    @override
    def should_add_frame(self):
        return False  # Segment recorder doesn't use individual frames, it uses ffmpeg segments directly from disk

    @override
    def can_start(self):
        # Segment recorder can always "start" (it just arms the worker)
        return True

    @override
    def seed_buffer(self):
        pass

    @override
    def start_recording(self):
        """
        For segment recording, 'start' just means:
        - start the merge worker thread (it will wait on self.event)
        """
        logger.debug(f"{self.camera.config.name} segment recorder started. Waiting for event to capture segments")
        super().start_recording()  # This will set up the filename and log_filename

    @override
    def stop_recording(self):
        """
        Capture everything needed to finalize this recording,
        without ever reading camera state again later.
        """
        self.final_end_time = time.time()

        # Resolve segments for this event window
        self.segments = self._get_segments(
            start_time=self.final_start_time,
            end_time=self.final_end_time,
        )

        if self.segments:
            logger.debug(f"{self.camera.config.name} captured {len(self.segments)} segments for recording")
            super().stop_recording()  # This will signal the worker to start merging


    def _async_writer_worker(self):
        """
        Runs ffmpeg merge in a separate thread.
        When the process finishes, write metadata and finalize files.
        """
        # Wait until stop_recording() has populated segments + snapshots
        self.event.wait()
        self.event.clear()

        if not self.segments:
            # Nothing to merge → nothing to record
            return
        
        self.list_filename = self.final_media_filename + ".list"

        # Protect these segments from cleanup while we merge
        FileCleaner.do_not_delete_set.update(self.segments)

        # Wait until the last segment in the window is fully written
        last_segment = self.segments[-1]
        self._wait_for_last_segment_to_finish(last_segment)

        # Build concat list file
        with open(self.list_filename, "w") as f:
            for segment_file in self.segments:
                try:
                    stat_entry = os.stat(segment_file)
                    if stat_entry.st_size > 0:
                        f.write(f"file '{os.path.abspath(segment_file)}'\n")
                except FileNotFoundError:
                    continue

        # Run ffmpeg concat to merge segments into a single MP4, with the moov atom at the front for streaming
        ffmpeg_cmd1 = [
            "ffmpeg",
            "-y",
            "-fflags", "+genpts",
            "-f", "concat",
            "-safe", "0",
            "-i", self.list_filename,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-preset", "veryfast",
            "-crf", "23",
            "-vsync", "cfr",
            "-r", f"{self.fps.as_int()}",
            "-video_track_timescale", "90000",
            self.temporary_media_filename,
        ]
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-fflags", "+genpts",
            "-f", "concat",
            "-safe", "0",
            "-i", self.list_filename,
            "-c", "copy",
            "-movflags", "+faststart",

            self.temporary_media_filename,
        ]

        try:
            log_file = open(self.temporary_log_filename, "w")
            logger.debug(f"{self.camera.config.name} merging {len(self.segments)} segments")
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=log_file,
            )
            process.communicate()
        finally:
            log_file.close()
            FileCleaner.do_not_delete_set -= set(self.segments)

    @override
    def _get_additional_metadata(self):
        # Use the segments captured for this recording
        return {
            "segments": self.segments,
            "list_filename": self.list_filename,
            "recorder_type": "FFmpeg segment",
            }
    
    def _get_segments(self, start_time: float, end_time: float) -> list[str]:
        """
        Return all .ts segments whose timestamp falls within [start_time, end_time].
        """
        selected: list[tuple[str, float]] = []
        for f in os.scandir(self.camera.config.segments_dir):
            if f.name.endswith(".ts"):
                try:
                    stat_entry = f.stat()
                    if start_time <= stat_entry.st_mtime <= end_time:
                        selected.append(
                            (os.path.join(self.camera.config.segments_dir, f.name), stat_entry.st_mtime)
                        )
                except Exception:
                    pass
        selected.sort(key=lambda x: x[1])
        return [f[0] for f in selected]

    def _wait_for_last_segment_to_finish(
        self,
        last_segment: str,
        timeout: float = 5.0,
        stable_time: float = 3.0,
    ):
        """
        Wait until the last segment *within the event window* is fully written.
        """
        deadline = time.time() + timeout
        last_size = -1
        stable_since = None

        while time.time() < deadline:
            try:
                size = os.path.getsize(last_segment)
            except FileNotFoundError:
                time.sleep(0.05)
                continue

            if size != last_size:
                last_size = size
                stable_since = time.time()
            else:
                if stable_since is not None and time.time() - stable_since >= stable_time:
                    return

            time.sleep(0.05)

        # Timeout → proceed anyway
        return

    @override
    def report_complete(self):
        log_event(message=f"FFmpeg segment recording available {self.formatted_duration}", level="record", camera=self.camera, file_path=self.final_metadata_filename)

