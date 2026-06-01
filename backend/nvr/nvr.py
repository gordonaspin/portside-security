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

from nvr.camera.camera import Camera
from logger.logger import log_event
from nvr.file_cleaner import FileCleaner
from nvr.lpr import LicensePlateRecognition, VideoProcessor
from nvr.processor import FrameProcessor
from reader.rtsp_reader import Reader, RTSPReader
from recorder.factory import FrameRecorderFactory
from utils.thread_safe import ThreadSafeList
from utils.utils import make_readable_ts, make_ts_string_precise
from utils.utils import get_camera_resolution

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

        self.recordings: ThreadSafeList = ThreadSafeList()

        self.cameras: dict[str, Camera] = {}
        self.frame_readers: dict[str, Reader] = {}
        self.frame_processors: dict[str, FrameProcessor] = {}

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
                config["cameras"][name]["recorder_factory"] == "FfmpegSegmentRecorderFactory",
                self.stop_event)
            self.frame_processors[name] = FrameProcessor(
                camera=self.cameras[name],
                reader=self.frame_readers[name],
                recorder_factory=FrameRecorderFactory.create(self.cameras[name], config["cameras"][name]["recorder_factory"]),
                model_cfg=config["model"],
                stop_event=self.stop_event,
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
        current_thread().name = "event_loader"

        while not self.stop_event.is_set():
            self.recordings = ThreadSafeList(self._load_events())
            time.sleep(5)
            for reader in self.frame_readers.values():
                if reader.process and reader.process.poll() is not None:
                    log_event("reader process ended", "error", camera=reader.camera)
                    reader.restart()
            pass

    def _load_events(self):
        files = []

        for camera in self.cameras.values():
            if camera.config.enabled:
                for f in glob.glob(f"{camera.config.metadata_dir}/*.json"):
                    try:
                        with open(f) as fp:
                            try:
                                event = json.load(fp)
                            except json.JSONDecodeError:
                                logger.warning(f"invalid JSON in file {f}, deleting the file")
                                os.remove(f)
                                continue
                            event.pop("segments", None)
                            event.pop("profile", None)
                            event.pop("tuner_stats", None)
                            event.pop("recommendations", None)
                            files.append(event)
                    except FileNotFoundError:
                        pass # it's possible a clean-up job whacked the file

        # Sort globally by start_time
        files.sort(key=lambda x: x["start_time"])

        return files


    def _lpr_frame_reader(self, camera: Camera):
        """
        Thread to read frames from the ffmpeg stdout stream and puts the frame on the camera queue.
        The queue length is 1, so if the queue is full that frame on the queue is dropped and
        replaced with the new frame. This means we drop frames to keep up. This is only for
        image processing, frames written to segments are not dropped
        """
        current_thread().name = f"{camera.config.name} _lpr_frame_reader"

        frame_size = camera.lpr.width * camera.lpr.height * 3

        while not self.stop_event.is_set():
            raw = self._read_exact(camera.lpr.process.stdout, frame_size)

            if raw is None:
                log_event(message="lpr reader failed", level="warn", camera=camera)
                continue

            frame = np.frombuffer(raw, np.uint8).reshape((camera.lpr.height, camera.lpr.width, 3))

            # latest-frame-wins
            if camera.lpr.queue.full():
                camera.lpr.queue.get_nowait()
            camera.lpr.queue.put(frame)


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