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

import cv2

from logger.logger import log_event
from camera.camera import Camera
from recorders.recorder import Recorder
from constants import PRE_RECORD_DURATION

logger = getLogger("nvr")

class AsyncFrameRecorder(Recorder):
    def __init__(self, camera: Camera, pre_record_duration=PRE_RECORD_DURATION):
        super().__init__(camera=camera, pre_record_duration=pre_record_duration)
        self.max_pre_frames = int(20 * pre_record_duration)

        # 1. Rolling buffer (automatically discards frames past the time limit)
        self.rolling_buffer = collections.deque(maxlen=self.max_pre_frames)
        
        # 2. Recording queue (unlimited size, used during active recording)
        self.record_queue = collections.deque()
        
        # Thread management states
        self.lock = threading.Lock()
        self.writer_thread = None

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

    def start_recording(self):
        """Locks in the pre-buffer and begins appending live footage."""
        with self.lock:
            if len(self.rolling_buffer) == 0:
                return

            tmpfile = tempfile.NamedTemporaryFile("w+b", dir=self.camera.recordings_dir, suffix='.mp4', delete=False)
            self.filename = os.path.join(self.camera.recordings_dir, tmpfile.name)

            # Seed the recording queue with a snapshot of the current pre-buffer
            self.record_queue = collections.deque(list(self.rolling_buffer))
            
        # Spawn the continuous background consumer thread
        self.writer_thread = threading.Thread(
            target=self._async_writer_worker, 
            args=(self.filename,),
            daemon=True
        )
        self.writer_thread.start()
        logger.debug(f"Frame recorder STARTED. Pre-buffer locked: {len(self.record_queue)} frames.")

    def stop_recording(self, tags: defaultdict[set]):
        """Signals the recording to stop. Main thread can keep running."""
            
        # We join the thread to guarantee the file is written and safe on disk
        if self.writer_thread:
            self.writer_thread.join()

        adjusted_start_time = self.camera.recording_start_time - self.pre_record_duration
        tags_str = self._tags_to_str(tags)
        timestamp_str = datetime.fromtimestamp(adjusted_start_time).strftime("%Y%m%d_%H%M%S")
        timestamp_name_tags = timestamp_str + "_" + tags_str
        metadata_filename = os.path.join(self.camera.metadata_dir, timestamp_name_tags + ".json")
        media_filename = os.path.join(self.camera.recordings_dir, timestamp_name_tags + "_fr.mp4")
        metadata_filename = os.path.join(self.camera.recordings_dir, timestamp_name_tags + "_fr.json")
        
        self.convert_for_streaming(self.filename, media_filename)
        os.remove(self.filename)

        cap = cv2.VideoCapture(media_filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        duration_seconds = 0
        if frame_count and fps:
            duration_seconds = frame_count / fps
        formatted_duration = str(timedelta(seconds=int(duration_seconds)))

        metadata = self.create_metadata(tags, media_filename, metadata_filename, adjusted_start_time, time.time())
        with open(metadata_filename, "w") as f:
            json.dump(metadata, f, default=lambda o: o.__dict__, indent=4)

        log_event(message=f"frame recording available {formatted_duration}", level="record", camera=self.camera, file_path=metadata_filename)

    def _async_writer_worker(self, filename):
        """Background thread that continuously drains the queue and writes to disk."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(filename, fourcc, int(self.camera.fps.value()), (self.camera.width, self.camera.height))
        
        try:
            while True:
                frame = None
                with self.lock:
                    # Check if there are frames ready to be written
                    if len(self.record_queue) > 0:
                        frame = self.record_queue.popleft()
                    # If queue is empty AND not recording, we are completely done
                    elif not self.camera.recording and len(self.record_queue) == 0:
                        break
                
                if frame is not None:
                    # High CPU overhead compression happens here safely on the background thread
                    video_writer.write(frame)
                else:
                    # Queue is momentarily empty, yield CPU execution to the main thread
                    time.sleep(1 / self.camera.fps.value())
        finally:
            video_writer.release()


    def convert_for_streaming(self, input_file: str, output_file: str):
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
            output_file,
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

        except FileNotFoundError:
            ("Error: FFmpeg is not installed or not in your system PATH.")
