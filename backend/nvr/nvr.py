import glob
import json
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from logging import getLogger
from threading import Thread, Event, current_thread

import cv2
import numpy as np
from numpy.typing import NDArray

from backend.nvr.camera.camera import Camera
from backend.logger.logger import log_event
from backend.nvr.file_cleaner import FileCleaner
from backend.nvr.processor import FrameProcessor
from backend.reader.rtsp_reader import Reader, RTSPReader
from backend.recorder.factory import FrameRecorderFactory
from backend.utils.thread_safe import ThreadSafeList
from backend.utils.utils import make_readable_ts, make_ts_string_precise
from backend.utils.utils import get_camera_resolution

logger = getLogger("pynvr")

# =========================
# NVR ENGINE
# =========================
class NVR:
    def __init__(self, config: dict):

        self.recordings_dir: str = config["recordings_directory"]
        self.logs_dir: str = config["logs_directory"]
        self.debug: bool = config["debug"]
        self.stop_event: Event = Event()
        self.thread: Thread = None

        self.cameras: dict[str, Camera] = {}
        self.frame_readers: dict[str, Reader] = {}
        self.frame_processors: dict[str, FrameProcessor] = {}
        self.recordings: ThreadSafeList = ThreadSafeList()

        camera_resolutions = self.get_all_camera_resolutions(config["cameras"])
        for name, _ in config["cameras"].items():
            if not config["cameras"][name]["enabled"]:
                log_event(message=f"{name} camera is disabled", level="info")
                continue
            actual_width, actual_height = camera_resolutions[name]
            if actual_width is None or actual_height is None:
                actual_width = config["cameras"][name]["resolution"]["width"]
                actual_height = config["cameras"][name]["resolution"]["height"]
                log_event(message=f"could not get resolution, falling back to configured resolution {actual_width}x{actual_height}", level="warn", camera=name)
            self.cameras[name] = Camera(name=name,
                                        width=actual_width,
                                        height=actual_height,
                                        config=config,
                                        logs_dir=self.logs_dir,
                                        recordings_dir=self.recordings_dir,
                                        )
    
            self.frame_readers[name] = RTSPReader(
                self.cameras[name],
                config["model"]["resolution"],
                config["cameras"][name]["recorder"] == "FFmpegSegment",
                self.stop_event)
            self.frame_processors[name] = FrameProcessor(
                camera=self.cameras[name],
                reader=self.frame_readers[name],
                recorder_factory=FrameRecorderFactory.create(self.cameras[name], config["cameras"][name]["recorder"]),
                model_cfg=config["model"],
                stop_event=self.stop_event,
                recordings=self.add_recording
                )
        FileCleaner.stop_event = self.stop_event
        FileCleaner.add(self.recordings_dir, "*.mp4", timedelta(**config["keep_recordings_timedelta"]), timedelta(minutes=5))
        FileCleaner.add(self.recordings_dir, "*.jpg", timedelta(**config["keep_recordings_timedelta"]), timedelta(minutes=5))
        FileCleaner.add(self.recordings_dir, "*.json", timedelta(**config["keep_recordings_timedelta"]), timedelta(minutes=5))
        FileCleaner.add(self.recordings_dir, "*.log", timedelta(**config["keep_logs_timedelta"]), timedelta(minutes=5))
        FileCleaner.add(self.logs_dir, "*.log", timedelta(**config["keep_logs_timedelta"]), timedelta(minutes=5))

    def start(self):
        """
        Start the NVR processes. Threads created are:
        1 ffmpeg reader thread for each camera, writing to segment files and stdout
        1 ffmpeg frame reader thread for each camera reading from stdout and writing frames to a queue
        1 frame processor thread to read frames from the queue and do image processing
        """
        self._load_events()

        if not self.stop_event.is_set():
            for camera in self.cameras.values():
                if camera.config.enabled:
                    self.frame_readers[camera.config.name].start()
                    self.frame_processors[camera.config.name].start()

            self.thread = Thread(target=self._watch_cameras_and_load_events,daemon=True)
            self.thread.start()

    def stop(self):
        """
        Stop the NVR
        """
        log_event(message="stopping NVR processors", level="info")
        for processor in self.frame_processors.values():
            processor.stop()
        log_event(message="stopping NVR readers", level="info")
        for reader in self.frame_readers.values():
            reader.stop()


    def threads(self):
        threads = []
        for camera in self.cameras.values():
            name = camera.config.name
            if self.frame_readers[name].thread is not None:
                threads.append(self.frame_readers[name].thread)
            if self.frame_processors[name].thread is not None:
                threads.append(self.frame_processors[name].thread)
            if self.frame_processors[name].recorder.thread is not None:
                threads.append(self.frame_processors[name].recorder.thread)
        if FileCleaner.thread is not None:
            threads.append(FileCleaner.thread)
        if self.thread is not None:
            threads.append(self.thread)
        return threads

    def _watch_cameras_and_load_events(self):
        """
        load events into recordings list and check each ffmpeg process
        every 5 seconds and restart if necessary
        """
        current_thread().name = "event loader"

        while not self.stop_event.is_set():
            time.sleep(5)
            for reader in self.frame_readers.values():
                if reader.process and reader.process.poll() is not None:
                    log_event("reader process ended", "error", camera=reader.camera)
                    reader.restart()
            pass

    def add_recording(self, metadata_file: str):
        self.recordings.append(self._load_event(metadata_file=metadata_file))

    def _load_event(self, metadata_file:str) -> dict:
        event = None

        with open(metadata_file) as fp:
            try:
                event = json.load(fp)
                for k in list(event):
                    # only send necessary event data to gui
                    if k not in ["camera",
                                    "tags",
                                    "media_filename",
                                    "start_time",
                                    "end_time",
                                    "start_fmt",
                                    "end_fmt",
                                    "metadata_filename",
                                    "recorder_type"
                                    ]:
                        event.pop(k, None)

            except json.JSONDecodeError:
                logger.warning(f"invalid JSON in file {metadata_file}, deleting the file")
                os.remove(metadata_file)
        return event

    def _load_events(self):
        events = []

        start = time.time()
        for camera in self.cameras.values():
            if camera.config.enabled:
                for f in glob.glob(f"{camera.config.metadata_dir}/*.json"):
                    try:
                        events.append(self._load_event(f))

                    except FileNotFoundError:
                        pass # it's possible a clean-up job whacked the file

        # Sort globally by start_time
        events.sort(key=lambda x: x["start_time"])
        self.recordings.extend(events)

        logger.debug(f"loaded {len(events)} events in {(time.time() - start):.2f} seconds")


    def get_all_camera_resolutions(self,camera_config):
        results = {}

        def task(name, url):
            w, h = get_camera_resolution(url)
            return name, (w, h)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(task, name, cfg["url"])
                for name, cfg in camera_config.items() if cfg["enabled"]
            ]

            for f in as_completed(futures):
                name, res = f.result()
                log_event(message=f"{name} camera resolution detected as {res[0]}x{res[1]}", level="info")
                results[name] = res

        return results