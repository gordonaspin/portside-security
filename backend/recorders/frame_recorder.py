import time
import subprocess
import threading
import collections
import json
import os
import numpy as np
from logging import getLogger
from collections import defaultdict
from datetime import datetime, timedelta
import tempfile
from typing import override

import cv2

from logger.logger import log_event
from camera.camera import Camera
from constants import PRE_RECORD_DURATION
from nvr.utils import make_readable_ts, make_ts_string, tags_to_str
from readers.rtsp_reader import SegmentCleaner

logger = getLogger("nvr")

class FrameRecorderFactory:
    @staticmethod
    def create(camera: Camera, factory_name: str, pre_record_duration=PRE_RECORD_DURATION):
        if factory_name == "OpenCVFrameRecorderFactory":
            return OpenCVFrameRecorderFactory
        elif factory_name == "FfmpegFrameRecorderFactory":
            return FfmpegFrameRecorderFactory
        elif factory_name == "FfmpegSegmentRecorderFactory":
            return FfmpegSegmentRecorderFactory
        else:
            logger.warning(f"Unknown recorder factory '{factory_name}' for camera {camera.name}. Defaulting to FfmpegFrameRecorderFactory.")
            return FfmpegFrameRecorderFactory

class OpenCVFrameRecorderFactory:
    @staticmethod
    def create(camera: Camera, pre_record_duration=PRE_RECORD_DURATION):
        return OpenCVFrameRecorder(camera=camera, pre_record_duration=pre_record_duration)

class FfmpegFrameRecorderFactory:
    @staticmethod
    def create(camera: Camera, pre_record_duration=PRE_RECORD_DURATION):
        return FfmpegFrameRecorder(camera=camera, pre_record_duration=pre_record_duration)
    
class FfmpegSegmentRecorderFactory:
    @staticmethod
    def create(camera: Camera, pre_record_duration=PRE_RECORD_DURATION):
        return FfmpegSegmentRecorder(camera=camera, pre_record_duration=pre_record_duration)

class Recorder:
    def __init__(self, camera: Camera, pre_record_duration=PRE_RECORD_DURATION):
        self.camera = camera
        self.pre_record_duration = pre_record_duration
        self.max_pre_frames = int(20 * pre_record_duration)

        # 1. Rolling buffer (automatically discards frames past the time limit)
        self.rolling_buffer = collections.deque(maxlen=self.max_pre_frames)
        
        # 2. Recording queue (unlimited size, used during active recording)
        self.record_queue = collections.deque()
        
        # Thread management states
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.writer_thread = None
        self.filename = None

    def add_frame(self, frame):
        """Call this inside your main loop for every single incoming frame."""
        with self.lock:
            # Always make a copy to prevent OpenCV reference overwrites
            copied_frame = frame.copy()
            
            # Keep updating the rolling pre-buffer history
            self.rolling_buffer.append(copied_frame)
            
            # If actively recording, append subsequent frames to the stream
            if self.camera.recording:
                self.record_queue.append(copied_frame)

    def can_start(self):
        with self.lock:
            if len(self.rolling_buffer) == 0:
                return False
        return True
    
    def start_recording(self):
        """Locks in the pre-buffer and begins appending live footage."""
        if not self.can_start():
            return

        self.filename  = tempfile.NamedTemporaryFile(
            "w+b",
            dir=self.camera.recordings_dir,
            suffix=".mp4",
            delete=False).name

        # Seed the recording queue with a snapshot of the current pre-buffer
        self.record_queue = collections.deque(list(self.rolling_buffer))
            
        # Spawn the thread
        self.writer_thread = threading.Thread(
            target=self._async_writer_worker, 
            args=(self.filename,),
            daemon=True
        )
        self.writer_thread.start()
        logger.debug(f"Frame recorder STARTED. Pre-buffer locked: {len(self.record_queue)} frames.")


    def stop_recording(self, tags: defaultdict[set]):
        """Signals the recording to stop. Main thread can keep running."""
        
        self.stop_event.set()
        # We join the thread to guarantee the file is written and safe on disk
        if self.writer_thread:
            self.writer_thread.join()

        adjusted_start_time = self.camera.recording_start_time - self.pre_record_duration
        tags_str = tags_to_str(tags)
        timestamp_str = make_ts_string(adjusted_start_time)
        timestamp_name_tags = timestamp_str + "_" + tags_str
        metadata_filename = os.path.join(self.camera.metadata_dir, timestamp_name_tags + ".json")
        media_filename = os.path.join(self.camera.recordings_dir, timestamp_name_tags + ".mp4")
        
        self._finalize_files(self.filename, media_filename, timestamp_name_tags)

        cap = cv2.VideoCapture(media_filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = self._stabilize_fps()

        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        duration_seconds = 0
        if frame_count and fps:
            duration_seconds = frame_count / fps
        formatted_duration = str(timedelta(seconds=int(duration_seconds)))

        metadata = self._create_metadata(tags, media_filename, metadata_filename, adjusted_start_time, time.time())
        with open(metadata_filename, "w") as f:
            json.dump(metadata, f, default=lambda o: o.__dict__, indent=4)

        log_event(message=f"frame recording available {formatted_duration}", level="record", camera=self.camera, file_path=metadata_filename)

    def _get_additional_metadata(self):
        return {}

    def _finalize_files(self, input_file: str, media_filname: str, timestamp_name_tags: str):
        pass

    def _create_metadata(self, tags: defaultdict, media_filename, metadata_filename, start_time, end_time):
        # Convert to a standard dict and sets to lists
        serializable_tags = {k: list(v) for k, v in tags.items()}
        profile = self.camera.profile_to_dict()
        stats = self.camera.auto_tuner.summarize()
        recs = self.camera.auto_tuner.recommend_adjustments()

        json_data = {
            "camera": self.camera.name,
            "tags": serializable_tags,
            "media_filename": media_filename,
            "start_time": start_time,
            "end_time": end_time,
            "start_fmt": make_readable_ts(start_time),
            "end_fmt": make_readable_ts(end_time),
            "metadata_filename": metadata_filename,
            "profile": profile,
            "tuner_stats": stats,
            "recommendations": recs,
        }
        return json_data | self._get_additional_metadata()

    def _stabilize_fps(self):
        fps = int(self.camera.fps.value())
        if fps <= 0:
            fps = 18
        return fps


class OpenCVFrameRecorder(Recorder):
    def __init__(self, camera: Camera, pre_record_duration=PRE_RECORD_DURATION):
        super().__init__(camera=camera, pre_record_duration=pre_record_duration)


    def _async_writer_worker(self, filename):
        """Background thread that continuously drains the queue and writes to disk."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(filename, fourcc, self._stabilize_fps(), (self.camera.width, self.camera.height))
        
        try:
            while True:
                frame = None
                with self.lock:
                    # Check if there are frames ready to be written
                    if len(self.record_queue) > 0:
                        frame = self.record_queue.popleft()
                    # If queue is empty AND not recording, we are completely done
                    elif self.stop_event.is_set() and len(self.record_queue) == 0:
                        break
                
                if frame is not None:
                    # High CPU overhead compression happens here safely on the background thread
                    video_writer.write(frame)
                else:
                    # Queue is momentarily empty, yield CPU execution to the main thread
                    time.sleep(0.01)
        finally:
            video_writer.release()


    def _finalize_files(self, input_file: str, media_filename: str, timestamp_name_tags: str):
        """Moves the moov atom to the front of an MP4 file for browser streaming."""
        # Define the FFmpeg command as a list of strings
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_file,
            "-c:v",
            "libx264",  # Explicitly convert the mp4v video to H.264
            "-c:a",
            "copy",  # Copy audio if present without changes
            "-movflags",
            "+faststart",  # Move metadata to the front for streaming
            media_filename,
        ]

        try:
            # Launch the process safely using a context manager
            with subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            ) as process:
                # Wait for completion and capture outputs
                stdout, stderr = process.communicate()

                if process.returncode != 0:
                    logger.debug(f"FFmpeg Error:\n{stderr}")
                else:
                    logger.debug("Success! Video is now ready for web streaming.")
                os.remove(self.filename)

        except FileNotFoundError:
            ("Error: FFmpeg is not installed or not in your system PATH.")


class FfmpegFrameRecorder(Recorder):
    def __init__(self, camera: Camera, pre_record_duration=PRE_RECORD_DURATION):
        super().__init__(camera=camera, pre_record_duration=pre_record_duration)


    def _async_writer_worker(self, filename):
        """Background thread that continuously drains the queue and writes to disk."""
        fps = self._stabilize_fps()

        command = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",       # Matches OpenCV format
            "-s", f"{self.camera.width}x{self.camera.height}",
            "-r", f"{fps}",            # Framerate
            "-i", "-",                 # Input from Python pipe
            
            # --- Compression & Compatibility Settings ---
            "-c:v", "libx264",         # H.264 codec (universal browser support)
            "-pix_fmt", "yuv420p",     # Crucial for browser playback
            "-crf", "23",              # Balanced quality vs file size (lower = higher quality)
            "-preset", "medium",       # Balanced encoding speed vs file size
            
            # --- Browser Optimization ---
            "-movflags", "+faststart", # Puts index at front for instant browser playback
            filename
        ]
        
        try:
            # Run the process
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
                )
            while True:
                frame = None
                with self.lock:
                    # Check if there are frames ready to be written
                    if len(self.record_queue) > 0:
                        frame = self.record_queue.popleft()
                    # If queue is empty AND not recording, we are completely done
                    elif self.stop_event.is_set() and len(self.record_queue) == 0:
                        break
                
                if frame is not None:
                    # High CPU overhead compression happens here safely on the background thread
                    process.stdin.write(frame.tobytes())
                else:
                    # Queue is momentarily empty, yield CPU execution to the main thread
                    time.sleep(0.01)
        except Exception as e:
            pass
        finally:
            process.stdin.close()
            process.wait()
    
    def _finalize_files(self, input_file: str, media_filename: str, timestamp_name_tags: str):
        # FFmpeg already produced a web‑ready MP4 at input_file
        os.rename(input_file, media_filename)


class FfmpegSegmentRecorder(Recorder):
    def __init__(self, camera: Camera, pre_record_duration=PRE_RECORD_DURATION):
        super().__init__(camera=camera, pre_record_duration=pre_record_duration)
        self.event: threading.Event = threading.Event()
        self.tags: defaultdict[str, set] | None = None
        self.segments: list[str] = []
        self.list_filename: str | None = None
        self.log_filename: str | None = None
        self.adjusted_start_time: float | None = None

        # Snapshots of mutable camera state (captured at stop_recording time)
        self.profile_snapshot = None
        self.tuner_stats_snapshot = None
        self.tuner_recs_snapshot = None
        self.fps_snapshot: float | None = None

    @override
    def add_frame(self, frame):
        # Segment recorder does not consume frames directly
        return

    @override
    def can_start(self):
        # Segment recorder can always "start" (it just arms the worker)
        return True

    @override
    def start_recording(self):
        """
        For segment recording, 'start' just means:
        - allocate a temp output filename
        - start the merge worker thread (it will wait on self.event)
        """
        self.filename = tempfile.NamedTemporaryFile(
            "w+b",
            dir=self.camera.recordings_dir,
            suffix=".mp4",
            delete=False
        ).name

        self.writer_thread = threading.Thread(
            target=self._async_writer_worker,
            args=(self.filename,),
            daemon=True,
        )
        self.writer_thread.start()

    @override
    def stop_recording(self, tags: defaultdict[set]):
        """
        Capture everything needed to finalize this recording,
        without ever reading camera state again later.
        """
        end_time = time.time()

        # Snapshot all mutable camera state we care about
        self.tags = tags
        self.adjusted_start_time = self.camera.recording_start_time - self.pre_record_duration
        self.profile_snapshot = self.camera.profile_to_dict()
        self.tuner_stats_snapshot = self.camera.auto_tuner.summarize()
        self.tuner_recs_snapshot = self.camera.auto_tuner.recommend_adjustments()
        self.fps_snapshot = self.camera.fps.value()

        # Resolve segments for this event window
        self.segments = self._get_segments(
            start_time=self.adjusted_start_time,
            end_time=end_time,
        )

        # Wake the worker
        self.event.set()

    def _async_writer_worker(self, filename: str):
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

        self.list_filename = filename + ".list"
        self.log_filename = filename + ".log"

        # Protect these segments from cleanup while we merge
        SegmentCleaner.do_not_delete_set.update(self.segments)

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

        if self.camera.debug:
            log_event(
                message=f"merging {len(self.segments)} segments to {filename}",
                level="debug",
                camera=self.camera,
            )

        # Run ffmpeg concat
        log_file = open(self.log_filename, "w")
        ffmpeg_cmd = [
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
            "-r", "20",
            "-video_track_timescale", "90000",
            filename,
        ]

        try:
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=log_file,
            )
            process.communicate()
        finally:
            log_file.close()
            SegmentCleaner.do_not_delete_set -= set(self.segments)

        # At this point, `filename` is the merged MP4.
        # Now we finalize filenames + metadata, using ONLY our snapshots.

        tags_str = tags_to_str(self.tags)
        timestamp_str = make_ts_string(self.adjusted_start_time)
        timestamp_name_tags = timestamp_str + "_" + tags_str

        metadata_filename = os.path.join(
            self.camera.metadata_dir,
            timestamp_name_tags + ".json",
        )
        media_filename = os.path.join(
            self.camera.recordings_dir,
            timestamp_name_tags + ".mp4",
        )

        # Move/rename files (also moves log/list)
        self._finalize_files(self.filename, media_filename, timestamp_name_tags)

        # Compute duration using the merged file
        cap = cv2.VideoCapture(media_filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = self._stabilize_fps()
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        duration_seconds = 0
        if frame_count and fps:
            duration_seconds = frame_count / fps
        formatted_duration = str(timedelta(seconds=int(duration_seconds)))

        # Build metadata using snapshots (no camera reads here)
        metadata = self._create_metadata(
            self.tags,
            media_filename,
            metadata_filename,
            self.adjusted_start_time,
            time.time(),
        )

        with open(metadata_filename, "w") as f:
            json.dump(metadata, f, default=lambda o: o.__dict__, indent=4)

        log_event(
            message=f"segment recording available {formatted_duration}",
            level="record",
            camera=self.camera,
            file_path=metadata_filename,
        )

    def _get_segments(self, start_time: float, end_time: float) -> list[str]:
        """
        Return all .ts segments whose timestamp falls within [start_time, end_time].
        """
        selected: list[tuple[str, float]] = []
        for f in os.scandir(self.camera.segments_dir):
            if f.name.endswith(".ts"):
                try:
                    stat_entry = f.stat()
                    if start_time <= stat_entry.st_mtime <= end_time:
                        selected.append(
                            (os.path.join(self.camera.segments_dir, f.name), stat_entry.st_mtime)
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

    def _finalize_files(self, input_file: str, media_filename: str, timestamp_name_tags: str):
        # FFmpeg already produced a web‑ready MP4 at input_file
        os.rename(input_file, media_filename)
        os.rename(
            self.log_filename,
            os.path.join(self.camera.logs_dir, timestamp_name_tags + "_merge.log"),
        )
        os.rename(
            self.list_filename,
            os.path.join(self.camera.recordings_dir, timestamp_name_tags + "_merge.list"),
        )

    @override
    def _get_additional_metadata(self):
        # Use the segments captured for this recording
        return {"segments": self.segments}

    def _stabilize_fps(self):
        # Prefer snapshot if available, otherwise fall back to camera
        if self.fps_snapshot and self.fps_snapshot > 0:
            return self.fps_snapshot
        return super()._stabilize_fps()
