from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
import glob
import json
from logging import getLogger
import os
import queue
import subprocess
from threading import Thread, Event, current_thread
import time
from math import sqrt

import cv2
import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO
from ultralytics.engine.results import Results

from nvr.utils import make_readable_ts, make_ts_string_precise
from camera.camera import Camera
import constants as constants
from logger.logger import log_event
from nvr.motion_tuner import MotionDecision
from utils.thread_safe import ThreadSafeSet, ThreadSafeList
from nvr.lpr import LicensePlateRecognition, VideoProcessor
from recorders.frame_recorder import FrameRecorderFactory
from readers.rtsp_reader import Reader, RTSPReader
from nvr.processor import FrameProcessor

logger = getLogger("nvr")

# =========================
# NVR ENGINE
# =========================
class NVR:
    def __init__(self, config: dict):
        self.width: int = config["resolution"]["width"]
        self.height: int = config["resolution"]["height"]
        self.max_pixels = self.width * self.height
        self.recordings_dir: str = config["recordings_directory"]
        self.logs_dir: str = config["logs_directory"]
        self.lpr_model: str = config["yolo"]["lpr_model"]
        self.debug: bool = config["debug"]
        
        yolo = YOLO(config["yolo"]["model"])
        classname_to_classindex: dict = {v: k for k, v in yolo.names.items()}
        self.selected_classes: list[int] = [classname_to_classindex[n] for n in config["yolo"]["classes"]]
        self.stop_event: Event = Event()

        self.recordings: ThreadSafeList = ThreadSafeList()

        self.cameras: dict[str, Camera] = {}
        self.frame_readers: dict[str, Reader] = {}
        self.frame_processors: dict[str, FrameProcessor] = {}

        for name, cfg in config["cameras"].items():
            self.cameras[name] = Camera(name=name,
                                        cfg=cfg,
                                        width=self.width,
                                        height=self.height,
                                        logs_dir=self.logs_dir,
                                        recordings_dir=self.recordings_dir,
                                        model=YOLO(config["yolo"]["model"]),
                                        )
            self.frame_readers[name] = RTSPReader(self.cameras[name], self.stop_event)
            self.frame_processors[name] = FrameProcessor(
                camera=self.cameras[name],
                reader=self.frame_readers[name],
                recorder_factory=FrameRecorderFactory.create(self.cameras[name], config["cameras"][name]["recorder_factory"]),
                model=YOLO(config["yolo"]["model"]),
                selected_classes=self.selected_classes,
                stop_event=self.stop_event,
                )

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

            Thread(target=self._watch_cameras_and_load_events,daemon=True).start()

    def stop(self):
        """
        Stop the NVR
        """
        log_event(message="stopping NVR readers", level="info")
        for reader in self.frame_readers.values():
            reader.stop()
        log_event(message="stopping NVR processors", level="info")
        for processor in self.frame_processors.values():
            processor.stop()

    def threads(self):
        threads = []
        for camera in self.cameras.values():
            threads.append(self.frame_readers[camera.config.name].reader_thread)
            threads.append(self.frame_processors[camera.config.name].processor_thread)
        return threads

    def _watch_cameras_and_load_events(self):
        """
        load events into recordings list and check each ffmpeg process
        every 5 seconds and restart if necessary
        """
        while not self.stop_event.is_set():
            self.recordings = ThreadSafeList(self._load_events())
            time.sleep(5)
            for reader in self.frame_readers.values():
                if reader.process and reader.process.poll() is not None:
                    log_event("ffmpeg died, restarting", "error", camera=reader.camera)
                    reader.restart()
            pass

    def _load_events(self):
        files = []

        for camera in self.cameras.values():
            if camera.config.enabled:
                for f in glob.glob(f"{camera.config.metadata_dir}/*.json"):
                    try:
                        with open(f) as fp:
                            event = json.load(fp)
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
                #self._restart_camera(camera)
                continue

            frame = np.frombuffer(raw, np.uint8).reshape((camera.lpr.height, camera.lpr.width, 3))

            # latest-frame-wins
            if camera.lpr.queue.full():
                camera.lpr.queue.get_nowait()
            camera.lpr.queue.put(frame)


    def _process_lpr_frames(self, camera: Camera):
        """
        Thread to process lpr frames from the camera queue.
        """
        
        def write_json(camera: Camera, plate, ts, epoch, image_path, metadata_path):
            with open(metadata_path, "w") as f:
                f.write(
                    json.dumps(
                        {
                            "camera": camera.config.name,
                            "tags":  {
                                "license": [plate]
                            },
                            "media_filename": image_path,
                            "start_time": epoch,
                            "end_time": epoch + 5.0, # fudge a duration so we can feed to timeline
                            "start_fmt": make_readable_ts(epoch),
                            "end_fmt": make_readable_ts(epoch + 5.0),
                            "metadata_filename": metadata_path,
                        }
                    )
                )
        
        current_thread().name = f"{camera.config.name} _process__lpr_frames"
        lpr = LicensePlateRecognition(self.lpr_model)
        vp = VideoProcessor(lpr)

        while not self.stop_event.is_set():
            # get latest frame (non-blocking)
            try:
                frame: NDArray[np.uint8] = camera.lpr.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if camera.lpr.first_frame:
                camera.lpr.gray_buf = np.zeros((camera.lpr.height, camera.lpr.width), dtype=np.uint8)
                camera.lpr.equalized_buf = np.zeros((camera.lpr.height, camera.lpr.width), dtype=np.uint8)
                camera.lpr.preprocessed_buf = np.zeros((camera.lpr.height, camera.lpr.width), dtype=np.uint8)

                log_event(message=f"reading from lpr stream", level="info", camera=camera)
                camera.lpr.first_frame = False

            if camera.recording_state.recording:
                frame, detected_texts = vp.process_frame(camera, frame)
                now: float = time.time()
                ts = datetime.now().isoformat()
                timestamp_str = make_ts_string_precise(now)
                license_image_path = os.path.join(camera.config.plates_dir, f"{timestamp_str}_plate") + ".jpg"
                license_metadata_path = os.path.join(camera.config.metadata_dir, f"{timestamp_str}_plate") + ".json"
                cv2.imwrite(license_image_path, frame)
                write_json(camera, "", ts, now, license_image_path, license_metadata_path)
                log_event(message=f"License plate logged", level="info", camera=camera, file_path=license_metadata_path)
                if detected_texts:
                    tags = '_'.join(detected_texts)
                    license_image_path = os.path.join(camera.config.plates_dir, f"{timestamp_str}_{tags}") + ".jpg"
                    license_metadata_path = os.path.join(camera.config.metadata_dir, f"{timestamp_str}_{tags}") + ".json"
                    cv2.imwrite(license_image_path, frame)
                    log_event(message=f"License plate identified {tags}", level="info", camera=camera, file_path=license_metadata_path)
                    write_json(camera, tags, ts, now, license_image_path, license_metadata_path)

