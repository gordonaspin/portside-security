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
from typing import Tuple
import time
from math import sqrt

import cv2
import numpy as np
from numpy.typing import NDArray
from ultralytics.engine.results import Results

from camera.camera import Camera
import constants as constants
from context import Context
from logger.logger import log_event
from model.model import Model
from nvr.motion_profiles import DayMotionProfile, NightMotionProfile, MotionDecision
from utils.thread_safe import ThreadSafeSet, ThreadSafePathDict, ThreadSafeList
from nvr.lpr import LicensePlateRecognition, VideoProcessor

logger = getLogger("nvr")

# =========================
# NVR ENGINE
# =========================
class NVR:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self._width: int = ctx.resolution[0]
        self._height: int = ctx.resolution[1]
        self._max_pixels = self._width * self._height
        
        self.model = Model(ctx.model)
        self.yolo_confidence_threshold = ctx.confidence_threshold
        self.motion_threshold = ctx.motion_threshold
        self.stop_event: Event = Event()
        self.debug: bool = self.ctx.debug
        self.debug_files: bool = self.ctx.debug_files
        self.selected_classes: list[int] = self.model.class_to_index(ctx.classes)

        self._recordings_dir: str = ctx.directory
        self._segments_dir: str = os.path.join(self._recordings_dir, "segments")
        self._images_dir: str = os.path.join(self._recordings_dir, "images")
        self._metadata_dir: str = os.path.join(self._recordings_dir, "metadata")
        self._plates_dir: str = os.path.join(self._recordings_dir, "plates")
        self._do_not_delete_set: ThreadSafeSet = ThreadSafeSet()
        self.recordings: ThreadSafeList = ThreadSafeList()
        self.cameras: dict[str, Camera] = {}
        for name, cfg in ctx.camera_config.items():
            self.cameras[name] = Camera(name=name,
                                        cfg=cfg,
                                        max_pixels=self._max_pixels,
                                        recordings_dir=os.path.join(self._recordings_dir, name),
                                        segments_dir=os.path.join(self._segments_dir, name),
                                        images_dir=os.path.join(self._images_dir, name),
                                        metadata_dir=os.path.join(self._metadata_dir, name),
                                        plates_dir=os.path.join(self._plates_dir, name),
                                        model=Model(ctx.model)
                                        )

    def update_yolo_confidence_threshold(self, val):
        self.yolo_confidence_threshold = val
        for camera in self.cameras.values():
            self.set_camera_motion_profile(camera)

    def update_motion_threshold(self, val):
        self.motion_threshold = val
        for camera in self.cameras.values():
            self.set_camera_motion_profile(camera)

    def set_camera_motion_profile(self, camera: Camera):
        if camera.is_night:
            camera.profile = NightMotionProfile(self._max_pixels, self.motion_threshold, self.yolo_confidence_threshold)
        else:
            camera.profile = DayMotionProfile(self._max_pixels, self.motion_threshold, self.yolo_confidence_threshold)

    def start(self):
        """
        Start the NVR processes. Threads created are:
        1 ffmpeg reader thread for each camera, writing to segment files and stdout
        1 ffmpeg frame reader thread for each camera reading from stdout and writing frames to a queue
        1 frame processor thread to read frames from the queue and do image processing
        """
        if not self.stop_event.is_set():
            for camera in self.cameras.values():
                if camera.enabled:
                    os.makedirs(camera.recordings_dir, exist_ok=True)
                    os.makedirs(camera.segments_dir, exist_ok=True)
                    os.makedirs(camera.images_dir, exist_ok=True)
                    os.makedirs(camera.metadata_dir, exist_ok=True)
                    os.makedirs(camera.plates_dir, exist_ok=True)
                    _, lpr = self._start_camera(camera=camera)
                    Thread(target=self._frame_reader, args=(camera,), daemon=True).start()
                    Thread(target=self._process_frames,args=(camera,), daemon=True).start()
                    if lpr:
                        Thread(target=self._lpr_frame_reader, args=(camera,), daemon=True).start()
                        Thread(target=self._process_lpr_frames,args=(camera,), daemon=True).start()

            Thread(target=self._cleanup_segments,daemon=True).start()
            Thread(target=self._watch_cameras_and_load_events,daemon=True).start()

    def stop(self):
        """
        Stop the NVR
        """
        for camera in self.cameras.values():
            if camera.enabled and camera.process is not None:
                self._stop_camera(camera)

    def _restart_camera(self, camera):
        """
        Stop and start the camera unless we are shutting down
        """
        if not self.stop_event.is_set():
            log_event(message="restarting camera", level="warn", camera=camera)
            self._stop_camera(camera)
            self._start_camera(camera)

    def _stop_camera(self, camera):
        """
        Stops the background ffmpeg process for the camera, closes pipes and resets the camera
        """
        if camera.enabled and camera.process is not None:
            ret = camera.process.poll()
            log_event(message=f"stopping camera with ret {ret}", level="info", camera=camera)
            camera.process.terminate()

            try:
                camera.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                camera.process.kill()
            camera.process.stdout.close()
            camera.first_frame = True

    def _start_camera(self, camera: Camera):
        """
        Starts ffmpeg as a subprocess reading from the camera RTSP stream. The stream is split
        in two writing simultaneously to segment files and stdout. No re-encoding happens to the
        segment files. The frames written to stdout are resized for image processing by cv2. 
        """
        if not self.stop_event.is_set():
            log_event(message=f"starting recorder", level="info", camera=camera)
            filespec = os.path.join(camera.segments_dir, "%Y%m%d_%H%M%S.ts")
            ffmpeg_cmd = [
                "ffmpeg",

                "-rtsp_transport", "tcp",           # Forces RTSP over TCP instead of UDP
                "-fflags", "nobuffer+genpts",       # Disables internal buffering, generates PTS
                "-flags", "low_delay",              # Tells decoder/demuxer to minimize delay (Reduces frame reordering buffers)
                "-i", camera.url,                   # RTSP stream from camera
                "-hide_banner",
                "-loglevel", "error",               # ONLY errors (no frame spam)
                "-nostats",
                
                "-filter_complex",                  # Split and reduce scale for raw only for OpenCV
                f"[0:v]scale={self._width}:{self._height},format=bgr24[raw]", # re-scale and raw BGR pixel format (OpenCV native)

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
            camera.process = process

            lpr_process = None
            if camera.is_lpr():
                ffmpeg_lpr_cmd = [
                    "ffmpeg",
                    "-rtsp_transport", "tcp",
                    "-fflags", "nobuffer+genpts",
                    "-flags", "low_delay",
                    "-i", camera.lpr.url,                # 4K stream
                    "-hide_banner",
                    "-loglevel", "error",
                    "-nostats",

                    "-vf", f"crop={camera.lpr.width}:{camera.lpr.height}:{camera.lpr.left}:{camera.lpr.top},format=bgr24",
                    "-f", "rawvideo",
                    "pipe:1"
                ]
                lpr_process =  subprocess.Popen(
                    ffmpeg_lpr_cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=0
                )
                camera.lpr.process = lpr_process

            return process, lpr_process

    def _watch_cameras_and_load_events(self):
        """
        load events into recordings list and check each ffmpeg process
        every 5 seconds and restart if necessary
        """
        while not self.stop_event.is_set():
            self.recordings = ThreadSafeList(self._load_events())
            time.sleep(5)
            for camera in self.cameras.values():
                if camera.process and camera.process.poll() is not None:
                    log_event("ffmpeg died, restarting", "error", camera=camera)
                    self._restart_camera(camera)
            pass

    def _cleanup_segments(self):
        """
        Thread that periodically deletes old segment files for all cameras
        """
        current_thread().name = "cleanup_segments"
            
        while True and not self.stop_event.is_set():
            try:
                for camera in self.cameras.values():
                    if camera.enabled:
                        path = os.path.join(camera.segments_dir, "*.ts")
                        files = sorted(glob.glob(path))
                        if len(files) > constants.BUFFER_SECONDS:
                            for f in files[:-constants.BUFFER_SECONDS]:
                                if f not in self._do_not_delete_set:
                                    try: os.remove(f)
                                    except: pass
                time.sleep(1)
            except Exception as e:
                log_event(message=f"exception in cleanup_segments {e}", level="error")

    def _get_segments(self, camera: Camera, end_time: float):
        """
        Return all .ts segments whose timestamp falls within camera.recording_start_time and now.
        Don't add edge-case files of length zero (partial .ts file)
        """
        files = sorted(glob.glob(os.path.join(camera.segments_dir, "*.ts")))
        selected = []

        for f in files:
            ts_str = os.path.basename(f).split(".")[0]  # "20260508_221056"
            try:
                ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S").timestamp()
            except ValueError:
                continue

            if camera.recording_start_time <= ts <= end_time:
                selected.append(f)

        return selected
    
    def _merge_segments_async(self, camera: Camera, tags: defaultdict[set], end_time: float):
        """
        Runs ffmpeg merge in a separate thread. When the process finishes,
        the log the event and delete the listing file.
        """

        segments = self._get_segments(camera=camera, end_time=end_time)
        tags_str = self._tags_to_str(tags)
        timestamp_str = datetime.fromtimestamp(camera.recording_start_time).strftime("%Y%m%d_%H%M%S")
        timestamp_name_tags = timestamp_str + "_" + tags_str
        list_filename = os.path.join(self.ctx.log_directory, f"{camera.name}_{timestamp_name_tags}.txt")
        metadata_filename = os.path.join(camera.metadata_dir, timestamp_name_tags + ".json")
        mp4_filename = os.path.join(camera.recordings_dir, timestamp_name_tags + ".mp4")
        merge_log_filename = os.path.join(self.ctx.log_directory, timestamp_name_tags + "_merge.log")
        self._do_not_delete_set.update(segments)

        def worker():
            time.sleep(4.0) # we sleep for a few seconds to allow ffmpeg to finish the last .ts file
            if segments:
                with open(list_filename,"w") as f:
                    for segment_file in segments:
                        try:
                            stat_entry = os.stat(segment_file)
                            if stat_entry.st_size > 0:
                                f.write(f"file '{os.path.abspath(segment_file)}'\n")
                            else:
                                continue
                        except FileNotFoundError as e:
                            continue
                if self.debug:
                    log_event(message=f"merging {len(segments)} segments {tags_str} to {mp4_filename}", level="debug", camera=camera, file_path=mp4_filename)
                
                # Convert to a standard dict and sets to lists
                serializable_tags = {k: list(v) for k, v in tags.items()}

                with open(metadata_filename, "w") as f:
                    json_data = {
                        "camera": camera.name,
                        "tags": serializable_tags,
                        "segments": segments,
                        "output": mp4_filename,
                        "start_time": camera.recording_start_time,
                        "end_time": end_time,
                        "metadata": metadata_filename,
                    }
                    f.write(json.dumps(json_data, indent=4))

                log_file = open(merge_log_filename, "w")
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-fflags", "+genpts",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", list_filename,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-preset", "veryfast",
                    "-crf", "23",
                    "-vsync", "cfr",
                    "-r", "20",
                    "-video_track_timescale", "90000",
                    mp4_filename
                ]
                try:
                    process = subprocess.Popen(
                        ffmpeg_cmd,
                        stdout=subprocess.PIPE,
                        stderr=log_file
                    )

                    stdout, stderr = process.communicate()

                    if process.returncode != 0:
                        # You can log or handle errors here if needed
                        pass

                finally:
                    # This runs when the thread finishes (success or failure)
                    log_file.close()
                    self._do_not_delete_set -= set(segments)
                    self._merge_complete(camera=camera, tags=tags, media_path=mp4_filename, metadata_path=metadata_filename)
            else:
                log_event(message=f"nothing to merge: {len(segments)} segments {tags_str} to {mp4_filename}", level="debug", camera=camera, file_path=mp4_filename)

        thread = Thread(target=worker, daemon=True)
        thread.start()
        return thread

    def _tags_to_str(self, tags: defaultdict[set]):
        if not tags:
            return ""

        parts = []
        for obj, colors in tags.items():
            object_str = obj
            color_str = ":".join(colors)
            parts.append(f"{object_str}({color_str})")
        return ",".join(parts)

    def _merge_complete(self, camera: Camera, tags: defaultdict[set], media_path: str, metadata_path: str):
        """
        logs the merge completion event and deletes recording if too short
        """
        cap = cv2.VideoCapture(media_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration_seconds = frame_count / fps
        formatted_duration = str(timedelta(seconds=int(duration_seconds)))
        if frame_count < constants.RECORDING_FRAME_COUNT_MINIMUM and os.path.isfile(media_path):
            os.remove(media_path)
            os.remove(metadata_path)
            log_event(message=f"recording auto-deleted with {frame_count} frames", level="info", camera=camera, file_path=media_path)
        else:
            log_event(message=f"recording available {formatted_duration}", level="record", camera=camera, file_path=metadata_path)

    def _load_events(self):
        flat = []

        for camera in self.cameras.values():
            if camera.enabled:
                for f in glob.glob(f"{camera.metadata_dir}/*.json"):
                    try:
                        with open(f) as fp:
                            event = json.load(fp)
                            event.pop("segments", None)
                            flat.append(event)
                    except FileNotFoundError:
                        pass # it's possible a clean-up job whacked the file

        # Sort globally by start_time
        flat.sort(key=lambda x: x["start_time"])

        return flat

    def _frame_reader(self, camera: Camera):
        """
        Thread to read frames from the ffmpeg stdout stream and puts the frame on the camera queue.
        The queue length is 1, so if the queue is full that frame on the queue is dropped and
        replaced with the new frame. This means we drop frames to keep up. This is only for
        image processing, frames written to segments are not dropped
        """
        current_thread().name = f"{camera.name} _frame_reader"

        frame_size = self._width * self._height * 3

        while not self.stop_event.is_set():
            raw = self._read_exact(camera.process.stdout, frame_size)

            if raw is None:
                log_event(message="reader failed", level="warn", camera=camera)
                if camera.fail_count < 3:
                    camera.fail_count += 1
                    self._restart_camera(camera)
                else:
                    log_event(message="stopping camera, too many failures, giving up", level="warn", camera=camera)
                    self._stop_camera(camera=camera)
                continue

            frame = np.frombuffer(raw, np.uint8).reshape((self._height, self._width, 3))

            # FPS calculation
            now = time.perf_counter()
            if camera.last_frame_time > 0:
                dt = now - camera.last_frame_time

                # filter pipeline artifacts
                if 0.02 < dt < 0.2:
                    inst_fps = 1.0 / dt
                    camera.dt.update(dt)
                    camera.fps.update(1.0 / camera.dt.value())

            camera.last_frame_time = now

            # latest-frame-wins
            if camera.frame_queue.full():
                camera.frame_queue.get_nowait()
                camera.total_drops += 1
            camera.frame_queue.put(frame)
            camera.total_frames += 1
            camera.drop_rate = camera.total_drops / camera.total_frames

    def _lpr_frame_reader(self, camera: Camera):
        """
        Thread to read frames from the ffmpeg stdout stream and puts the frame on the camera queue.
        The queue length is 1, so if the queue is full that frame on the queue is dropped and
        replaced with the new frame. This means we drop frames to keep up. This is only for
        image processing, frames written to segments are not dropped
        """
        current_thread().name = f"{camera.name} _lpr_frame_reader"

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

    def draw_text(self, frame, text, position, font, font_scale, color, thickness, bg_color):
        """
        Utility function to draw text on a frame
        """
        if not text:
            return
        x, y = position
        text_size, _ = cv2.getTextSize(text, font, 0.7, thickness)
        text_w, text_h = text_size
        # rectangle: (x, y) top-left, (x + text_w, y + text_h) bottom-right
        # text: (x, y + text_h) bottom-left corner
        cv2.rectangle(frame, (x, y-2), (x + text_w + 2, y + text_h + 6), bg_color, -1)
        cv2.putText(frame, text, (x+1, y + text_h), font, font_scale, color, thickness)

    def _process_frames(self, camera: Camera):
        """
        Thread to process frames from the camera queue. Processing is as follows:
        - get the frame from the queue (latest frame processing, some frames are dropped)
        - convert to grayscale for image processing (faster than color)
        - blur the gray (better for motion detection)
        - calculate the difference between this gray frame and the previous one (for motion detection)
        - calculate a theshold image based on the difference and score (count) the white pixels
        - if the score is above threshold, compute the motion contours and rectangles from the threshold image
        - draw status, contours and rectangles on a debug copy of the image. Red for movement that is too small, green for movement that we care about
        - if we have movement we care about, run YOLO and check if movement and detected objects intersect
        - if movement and objects intersect, start recording if we have seen motion for a number of frames, get a list of pre-record segments
        - keep recording while there is motion, add to the segment list
        - stop recording after motion is not detected for a number of frames
        - get the list of segment files that correlate to the recording period
        - merge the segment files in to a video file, asynchronously
        - if there were YOLO results and movement, write the objects on to the image
        - if there was motion, merge the image and overlay
        - store the image and status in the camera object, the GUI will read this image
        """
        current_thread().name = f"{camera.name} _process_frames"

        while not self.stop_event.is_set():
            try:
                frame: NDArray[np.uint8] = camera.frame_queue.get(timeout=0.5)
                frame_bgr: NDArray[np.uint8] = frame.copy()
            except queue.Empty:
                continue

            if camera.first_frame:
                log_event(message=f"reading from stream", level="info", camera=camera)
                h, w = frame_bgr.shape[:2]
                camera.bg_frame_buf      = np.zeros((h, w), dtype=np.uint8)
                camera.diff_blur_buf     = np.zeros((h, w), dtype=np.uint8)
                camera.diff_buf          = np.zeros((h, w), dtype=np.uint8)
                camera.diff_filtered_buf = np.zeros((h, w), dtype=np.uint8)
                camera.diff_mask_buf     = np.zeros((h, w), dtype=np.uint8)
                camera.edges_buf         = np.zeros((h, w), dtype=np.uint8)
                camera.gray_buf          = np.zeros((h, w), dtype=np.uint8)
                camera.thresh_buf        = np.zeros((h, w), dtype=np.uint8)
                camera.sobel_x_buf       = np.zeros((h, w), dtype=np.int16)
                camera.sobel_y_buf       = np.zeros((h, w), dtype=np.int16)
                camera.sobel_x_abs_buf   = np.zeros((h, w), dtype=np.uint8)
                camera.sobel_y_abs_buf   = np.zeros((h, w), dtype=np.uint8)
                camera.background_buf    = camera.gray_buf.astype("float32")
                camera.first_frame = False

            now: float = time.time()

            # periodic night/day check
            if now - camera.last_night_time_check > constants.PERIODIC_CHECK_INTERVAL:
                camera.is_night = self._is_night_time(frame_bgr, constants.NIGHT_TIME_THRESHOLD)
                self.set_camera_motion_profile(camera)
                camera.last_night_time_check = time.time()

            # periodic auto-tuning of motion profile
            if now - camera.last_auto_adjust > 60:
                log_event(f"tuning profile", level="info", camera=camera)
                camera.auto_adjust_profile()
                camera.last_auto_adjust = now

            # --- GRAY + BLUR ---
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY, dst=camera.gray_buf)
            cv2.GaussianBlur(camera.gray_buf, (21, 21), 0, dst=camera.gray_buf)

            # initialize background model
            if camera.background_buf is None:
                camera.background_buf = camera.gray_buf.astype("float32")
                continue

            # update background model
            cv2.accumulateWeighted(
                camera.gray_buf,
                dst=camera.background_buf,
                alpha=0.12 if camera.is_night else 0.02
            )

            cv2.convertScaleAbs(camera.background_buf, dst=camera.bg_frame_buf)

            # --- MOTION DIFF ---
            cv2.absdiff(camera.bg_frame_buf, camera.gray_buf, dst=camera.diff_buf)

            # --- NOISE-ADAPTIVE LOW-INTENSITY FILTERING ---
            camera.noise = np.std(camera.diff_buf)
            cutoff = max(8, min(20, camera.noise * 1.5))

            cv2.threshold(camera.diff_buf, cutoff, 255, cv2.THRESH_BINARY, dst=camera.diff_mask_buf)
            cv2.bitwise_and(camera.diff_buf, camera.diff_mask_buf, dst=camera.diff_filtered_buf)

            # --- BLUR TO REDUCE HIGH-FREQUENCY NOISE ---
            cv2.GaussianBlur(camera.diff_filtered_buf, (7, 7), 0, dst=camera.diff_blur_buf)

            # --- OTSU THRESHOLD ON CLEANED DIFF ---
            cv2.threshold(
                camera.diff_blur_buf, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                dst=camera.thresh_buf
            )

            camera.score = cv2.countNonZero(camera.thresh_buf)
            camera.white_ratio = camera.score / camera.max_pixels

            # --- GLOBAL SHADOW SWEEP ---
            if camera.white_ratio > 0.10 and camera.edge_density < camera.profile.min_edge_density(camera.noise):
                camera.auto_tuner.record(MotionDecision(
                    passed=False,
                    reason="shadow_low_edge",
                    details={"white_ratio": camera.white_ratio, "edge_density": camera.edge_density}
                ))
                camera.motion_boxes_list.clear()
                continue

            # --- SOBEL ---
            cv2.Sobel(camera.gray_buf, cv2.CV_16S, 1, 0, dst=camera.sobel_x_buf)
            cv2.Sobel(camera.gray_buf, cv2.CV_16S, 0, 1, dst=camera.sobel_y_buf)
            cv2.convertScaleAbs(camera.sobel_x_buf, dst=camera.sobel_x_abs_buf)
            cv2.convertScaleAbs(camera.sobel_y_buf, dst=camera.sobel_y_abs_buf)
            cv2.addWeighted(camera.sobel_x_abs_buf, 0.5, camera.sobel_y_abs_buf, 0.5, 0, dst=camera.edges_buf)
            camera.edge_density = cv2.countNonZero(camera.edges_buf) / camera.max_pixels

            # --- LOW-EDGE SHADOW ---
            if camera.white_ratio > 0.10 and camera.edge_density < 0.02:
                camera.auto_tuner.record(MotionDecision(
                    passed=False,
                    reason="shadow_low_edge2",
                    details={"white_ratio": camera.white_ratio, "edge_density": camera.edge_density}
                ))
                camera.motion_boxes_list.clear()
                continue

            # --- FIND MOTION BOXES ---
            krs, kcs, dsrs, dscs, dars, dacs = [], [], [], [], [], []
            camera.motion_boxes_list.clear()

            if camera.score > camera.profile.motion_threshold:
                krs, kcs, dsrs, dscs, dars, dacs = self._find_motion_boxes(camera)
                camera.motion_boxes_list.extend(krs)

            # --- TOTAL MOTION AREA ---
            total_motion_area = sum(
                (x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in camera.motion_boxes_list
            )

            if total_motion_area < camera.profile.min_total_motion_area:
                camera.auto_tuner.record(MotionDecision(
                    passed=False,
                    reason="low_total_area",
                    details={"total_motion_area": total_motion_area}
                ))
                camera.motion_boxes_list.clear()

            # --- MOTION CONFIDENCE (PIXEL + BOX + PERSISTENCE) ---
            camera.pixel_score = min(camera.score / (camera.profile.motion_threshold * 3.0), 1.0)

            object_area = sum(
                (x2 - x1) * (y2 - y1)
                for (x1, y1, x2, y2) in camera.motion_boxes_list
            )
            camera.box_score = min(object_area / (camera.profile.min_sum_box_area * 2.0), 1.0)

            camera.persist_score = max(
                0.0,
                1.0 - ((now - camera.last_motion_time) / camera.profile.motion_persistence_time)
            )

            camera.motion_confidence = (
                (camera.pixel_score * 0.4) +
                (camera.box_score   * 0.4) +
                (camera.persist_score * 0.2)
            )

            # --- YOLO ---
            result = None
            camera.classes_in_frame_dict.clear()
            camera.has_moving_object = False
            moving_yolo_indices: set[int] = set()

            if camera.debug or (camera.motion_boxes_list and camera.motion_confidence > 0.05):
                result: Results = camera.model.model.predict(
                    frame_bgr,
                    conf=camera.profile.yolo_confidence_threshold,
                    classes=self.selected_classes if self.selected_classes else None,
                    verbose=False,
                    imgsz=512,
                )[0]

                yolo_boxes = []
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    yolo_boxes.append((int(x1), int(y1), int(x2), int(y2)))

                inflated_motion_boxes = [
                    self._inflate_box(b, camera.profile.inflate_motion_boxes)
                    for b in camera.motion_boxes_list
                ]

                for i, yb in enumerate(yolo_boxes):
                    for mb in inflated_motion_boxes:
                        if self._boxes_overlap(mb, yb):
                            moving_yolo_indices.add(i)
                            break

                keep_mask = [i in moving_yolo_indices for i in range(len(yolo_boxes))]
                camera.keep_mask = keep_mask
                camera.has_moving_object = any(keep_mask)

                # --- TUNER: YOLO overlap noise ---
                if camera.motion_boxes_list and not camera.has_moving_object:
                    camera.auto_tuner.record(MotionDecision(
                        passed=False,
                        reason="yolo_overlap_noise",
                        details={"motion_boxes": len(camera.motion_boxes_list)}
                    ))
                    # collapse confidence a bit when YOLO sees nothing overlapping
                    camera.motion_confidence = min(camera.motion_confidence, 0.10)

                result.boxes = result.boxes[camera.keep_mask]

                for i, box in enumerate(result.boxes):
                    class_name = self.model.model.names[int(box.cls)]
                    roi = self.yolo_box_to_roi(frame_bgr, box)
                    if roi.size > 0:
                        color = self._detect_object_color(roi)
                        camera.classes_in_frame_dict[class_name].add(color)

                # --- LOG EVENT (preserved semantics) ---
                if camera.debug and self.debug and camera.has_moving_object:
                    log_event(
                        message=f"moving objects detected: {self._tags_to_str(camera.classes_in_frame_dict)}",
                        level="debug",
                        camera=camera
                    )

            # --- HARD RESET WHEN MOTION DISAPPEARS ---
            if not camera.motion_boxes_list:
                # Reset persistence immediately
                camera.motion_persistence = 0
                camera.persist_score = 0.0

                # Collapse confidence quickly to allow fast stop
                camera.motion_confidence = min(camera.motion_confidence, 0.05)

                # Mark last_motion_time so POST_RECORD_DURATION starts now
                camera.last_motion_time = now

            # --- OBJECT-LIKE PERSISTENCE (AFTER YOLO) ---
            is_object_like_motion = (
                camera.motion_boxes_list and
                object_area >= camera.profile.min_sum_box_area and
                camera.edge_density >= 0.02 and
                camera.has_moving_object and
                camera.motion_confidence >= 0.15
            )

            if is_object_like_motion:
                camera.motion_persistence += 1
            else:
                camera.motion_persistence = max(0, camera.motion_persistence - 1)

            # --- DRAW STATUS TEXT ---
            self.draw_text(
                frame_bgr, camera.status_text, (0, 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255) if camera.recording else (0, 255, 0),
                2, (32, 32, 32)
            )
            self.draw_text(
                frame_bgr, camera.objects_text, (0, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, (32, 32, 32)
            )

            # --- DEBUG PANELS / UI ---
            if camera.debug:
                camera.debug_motion_image = self.draw_debug_panels(
                    camera,
                    frame_bgr,
                    result,
                    krs, kcs, dsrs, dscs, dars, dacs
                )

            # --- RECORDING DECISION ---
            motion_is_persistent = (
                camera.motion_persistence >= camera.profile.min_motion_frames
            )

            object_motion = (
                motion_is_persistent and
                camera.has_moving_object and
                len(moving_yolo_indices) > 0
            )

            object_area_ok = (
                object_area >= camera.profile.min_sum_box_area
            )

            START_CONF = camera.profile.motion_confidence_min
            STOP_CONF  = START_CONF * 0.5

            should_start = (
                object_motion and
                object_area_ok and
                camera.motion_confidence >= START_CONF
            )

            should_continue = (
                camera.recording and
                camera.motion_confidence >= STOP_CONF
            )

            # --- ADDITIONAL CONFIDENCE DECAY WHEN YOLO SEES NOTHING ---
            if camera.recording and not camera.has_moving_object:
                camera.motion_confidence *= 0.5
                if camera.motion_confidence > STOP_CONF:
                    camera.motion_confidence *= 0.5

            camera.should_record = should_start or should_continue

            # only treat as "last motion" when we actually have motion + objects
            if camera.motion_boxes_list and camera.has_moving_object:
                camera.last_motion_time = now

            # --- TUNER: insufficient persistence ---
            if camera.motion_boxes_list and not motion_is_persistent:
                camera.auto_tuner.record(MotionDecision(
                    passed=False,
                    reason="short_motion",
                    details={"persistence": camera.motion_persistence}
                ))

            # --- TUNER: insufficient confidence ---
            if camera.motion_boxes_list and camera.motion_confidence < START_CONF:
                camera.auto_tuner.record(MotionDecision(
                    passed=False,
                    reason="low_confidence",
                    details={"confidence": camera.motion_confidence}
                ))

            # --- START RECORDING ---
            if should_start and not camera.recording:
                camera.recording = True
                camera.recording_start_time = now - constants.PRE_RECORD_DURATION
                camera.active_objects_dict = deepcopy(camera.classes_in_frame_dict)
                camera.last_recording_time = now

                camera.auto_tuner.record(MotionDecision(
                    passed=True,
                    reason="recording_start",
                    details={"confidence": camera.motion_confidence}
                ))

                log_event(
                    message=f"recording start {self._tags_to_str(camera.active_objects_dict)}",
                    level="info",
                    camera=camera
                )

            # --- CONTINUE RECORDING ---
            if camera.recording:
                for item, colors in camera.classes_in_frame_dict.items():
                    camera.active_objects_dict[item].update(colors)

                camera.auto_tuner.record(MotionDecision(
                    passed=True,
                    reason="recording_continue",
                    details={"confidence": camera.motion_confidence}
                ))

            # --- STOP RECORDING ---
            if camera.recording and not should_continue:
                if now - camera.last_motion_time > constants.POST_RECORD_DURATION:
                    camera.recording = False
                    tags = deepcopy(camera.active_objects_dict)
                    self._merge_segments_async(camera, tags, now)

                    camera.auto_tuner.record(MotionDecision(
                        passed=True,
                        reason="recording_stop",
                        details={"confidence": camera.motion_confidence}
                    ))

                    camera.classes_in_frame_dict.clear()
                    camera.active_objects_dict.clear()
                    camera.motion_frames = 0
                    camera.no_motion_frames = 0
                    camera.motion_persistence = 0

            # --- FINAL FRAME OUTPUT ---
            if result is not None:
                img_bgr = result.plot(pil=False)  # pil=False returns BGR
            else:
                img_bgr = frame_bgr

            if camera.debug and camera.debug_motion_image is not None:
                img_bgr = camera.debug_motion_image

            camera.latest_frame = img_bgr

            parts = [self._make_status(camera)]
            if camera.is_night:
                parts.append("Night")

            parts.append(f"FPS {int(camera.fps.value())}:{camera.drop_rate:.2f}")
            camera.objects_text = self._tags_to_str(camera.active_objects_dict)
            camera.status_text = " | ".join(parts)


    def _process_lpr_frames(self, camera: Camera):
        """
        Thread to process lpr frames from the camera queue.
        """
        
        def write_json(camera: Camera, plate, ts, epoch, image_path, metadata_path):
            with open(metadata_path, "w") as f:
                f.write(
                    json.dumps(
                        {
                            "camera": camera.name,
                            "tags":  {
                                "license": [plate]
                            },
                            "output": image_path,
                            "start_time": epoch,
                            "end_time": epoch + 5.0, # fudge a duration so we can feed to timeline
                            "metadata": metadata_path
                        }
                    )
                )
        
        current_thread().name = f"{camera.name} _process__lpr_frames"
        lpr = LicensePlateRecognition(self.ctx.lpr_model)
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

            if camera.recording:
                frame, detected_texts = vp.process_frame(camera, frame)
                now: float = time.time()
                ts = datetime.now().isoformat()
                timestamp_str = datetime.fromtimestamp(now).strftime("%Y%m%d_%H%M%S_%f")
                license_image_path = os.path.join(camera.plates_dir, f"{timestamp_str}_plate") + ".jpg"
                license_metadata_path = os.path.join(camera.metadata_dir, f"{timestamp_str}_plate") + ".json"
                cv2.imwrite(license_image_path, frame)
                write_json(camera, "", ts, now, license_image_path, license_metadata_path)
                log_event(message=f"License plate logged", level="info", camera=camera, file_path=license_metadata_path)
                if detected_texts:
                    tags = '_'.join(detected_texts)
                    license_image_path = os.path.join(camera.plates_dir, f"{timestamp_str}_{tags}") + ".jpg"
                    license_metadata_path = os.path.join(camera.metadata_dir, f"{timestamp_str}_{tags}") + ".json"
                    cv2.imwrite(license_image_path, frame)
                    log_event(message=f"License plate identified {tags}", level="info", camera=camera, file_path=license_metadata_path)
                    write_json(camera, tags, ts, now, license_image_path, license_metadata_path)

    def _make_status(self, camera: Camera):
        """
        creates a string that represents the status (red/green for recording/live)
        """
        idx = int(time.time() * 4) % 4

        record_cycle = ["*", "*", " ", " "]

        pulse = record_cycle[idx] if camera.recording else ""

        return f"{pulse}{'REC' if camera.recording else 'LIVE'}"
    

    def _find_motion_boxes(self, camera: Camera):
        """
        Find motion boxes using contour analysis with solidity filtering.
        Returns:
            kept_rects, kept_contours,
            small_rects, small_contours,
            angular_rects, angular_contours
        """

        tuner = camera.auto_tuner  # shorthand

        # Profile thresholds
        min_solidity = camera.profile.min_contour_solidity
        min_w = camera.profile.min_box_width
        min_h = camera.profile.min_box_height
        min_edge_density = camera.profile.min_edge_density(camera.noise)
        max_aspect = camera.profile.max_allowed_aspect_ratio
        min_contour_area_ratio = camera.profile.min_contour_area_ratio

        thresh = camera.thresh_buf
        edges = camera.edges_buf

        h, w = thresh.shape[:2]
        frame_area = w * h
        min_area = frame_area * min_contour_area_ratio

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        kept_rects = []
        kept_contours = []

        small_rects = []
        small_contours = []

        angular_rects = []
        angular_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1:
                continue

            # --- MINIMUM AREA FILTER ---
            if area < min_area:
                x, y, w0, h0 = cv2.boundingRect(cnt)
                small_rects.append((x, y, x + w0, y + h0))
                small_contours.append(cnt)

                tuner.record(MotionDecision(
                    passed=False,
                    reason="small_contour",
                    details={"area": area, "min_area": min_area}
                ))
                continue

            # --- SOLIDITY FILTER ---
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue

            solidity = float(area) / hull_area
            if solidity < min_solidity:
                x, y, w0, h0 = cv2.boundingRect(cnt)
                angular_rects.append((x, y, x + w0, y + h0))
                angular_contours.append(cnt)

                tuner.record(MotionDecision(
                    passed=False,
                    reason="low_solidity",
                    details={"solidity": solidity, "min_solidity": min_solidity}
                ))
                continue

            # --- MINIMUM WIDTH/HEIGHT FILTER ---
            x, y, w0, h0 = cv2.boundingRect(cnt)
            if w0 < min_w or h0 < min_h:
                small_rects.append((x, y, x + w0, y + h0))
                small_contours.append(cnt)

                tuner.record(MotionDecision(
                    passed=False,
                    reason="small_dimensions",
                    details={"w": w0, "h": h0, "min_w": min_w, "min_h": min_h}
                ))
                continue

            # --- SOBEL EDGE DENSITY FILTER ---
            roi_edges = edges[y:y+h0, x:x+w0]
            edge_count = cv2.countNonZero(roi_edges)
            edge_density = edge_count / max(1, (w0 * h0))

            if edge_density < min_edge_density:
                angular_rects.append((x, y, x + w0, y + h0))
                angular_contours.append(cnt)

                tuner.record(MotionDecision(
                    passed=False,
                    reason="low_edge_density",
                    details={"edge_density": edge_density, "min_edge_density": min_edge_density}
                ))
                continue

            # --- ASPECT RATIO FILTER ---
            aspect = max(w0, h0) / max(1, min(w0, h0))
            if aspect > max_aspect:
                angular_rects.append((x, y, x + w0, y + h0))
                angular_contours.append(cnt)

                tuner.record(MotionDecision(
                    passed=False,
                    reason="high_aspect_ratio",
                    details={"aspect": aspect, "max_aspect": max_aspect}
                ))
                continue

            # --- ACCEPTED MOTION BOX ---
            kept_rects.append((x, y, x + w0, y + h0))
            kept_contours.append(cnt)

            tuner.record(MotionDecision(
                passed=True,
                reason="accepted_contour",
                details={
                    "area": area,
                    "solidity": solidity,
                    "edge_density": edge_density,
                    "aspect": aspect
                }
            ))

        return (
            kept_rects, kept_contours,
            small_rects, small_contours,
            angular_rects, angular_contours
        )


    def _is_night_time(self, frame, brightness_threshold=50):
        """
        determines if we are looking at a night time image.
        Converts the frame to HSV and computes the mean value of intensity channel
        it's night time if below the threshold, else it's day time
        """
        # Convert to HSV (Hue, Saturation, Intensity)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate average brightness (V channel - Intensity)
        mean_brightness = np.mean(hsv[:,:,2])
        
        # If brightness is low, it's likely night time
        return mean_brightness < brightness_threshold

    def _inflate_box(self, box, inflate_px: int) -> tuple[int, int, int, int]:
        """Inflate a box by inflate_px in all directions, clamped to frame."""
        x1, y1, x2, y2 = box
        x1 = max(0, x1 - inflate_px)
        y1 = max(0, y1 - inflate_px)
        x2 = min(self._width - 1, x2 + inflate_px)
        y2 = min(self._height - 1, y2 + inflate_px)
        return (x1, y1, x2, y2)

    def _boxes_overlap(self, a, b) -> bool:
        """Return True if two boxes (x1,y1,x2,y2) overlap."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

    def yolo_box_to_roi(self, frame_bgr, box):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Clamp to image bounds
        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))

        roi = frame_bgr[y1:y2, x1:x2].copy()
        return roi

    def _detect_object_color(self, roi_bgr, k=2):
        if roi_bgr is None or roi_bgr.size == 0:
            return "unknown"

        # Smooth noise
        roi = cv2.GaussianBlur(roi_bgr, (5, 5), 0)

        # Convert to LAB (OpenCV LAB)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Convert OpenCV LAB → true LAB
        lab[:, :, 1] -= 128.0   # a channel shift
        lab[:, :, 2] -= 128.0   # b channel shift

        # Flatten for k-means
        pixels = lab.reshape((-1, 3))

        # K-means clustering
        _, labels, centers = cv2.kmeans(
            pixels, k, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
            3,
            cv2.KMEANS_PP_CENTERS
        )

        counts = np.bincount(labels.flatten())
        sorted_idx = np.argsort(-counts)

        total = len(pixels)

        for idx in sorted_idx:
            # Ignore tiny clusters (noise, highlights)
            if counts[idx] < 0.05 * total:
                continue

            lab_color = centers[idx]
            color_name = self._classify_color_lab(lab_color)

            if color_name != "unknown":
                return color_name

        return "unknown"


    def _classify_color_lab(self, lab_color):
    # LAB-based color classifier
        L, a, b = lab_color
        chroma = sqrt(a*a + b*b)

        # Neutral detection
        if L < 30:
            return "black"
        if chroma < 10:
            return "white" if L > 200 else "gray"

        # Metallic detection
        if 60 < L < 95 and chroma < 25:
            return "silver"
        if 60 < L < 95 and 25 <= chroma < 45 and b > 20:
            return "gold"

        # Earth tone detection
        if 30 < L < 70 and 10 < chroma < 40:
            if b > 25:
                return "tan"
            if 10 < b <= 25:
                return "beige"
            if b <= 10:
                return "brown"

        # Standard color classification
        best = None
        best_dist = 1e9

        for name, ref in constants.REF_COLORS.items():
            dist = np.linalg.norm(lab_color - ref)
            if dist < best_dist:
                best_dist = dist
                best = name

        return best

    def draw_debug_panels(self,
                      camera: Camera,
                      frame_bgr: NDArray[np.uint8],
                      result: Results,
                      krs: list[Tuple[int, int, int, int]],
                      kcs: list[NDArray[np.int32]],
                      dsrs: list[Tuple[int, int, int, int]],
                      dscs: list[NDArray[np.int32]],
                      dars: list[Tuple[int, int, int, int]],
                      dacs: list[NDArray[np.int32]]
                      ):

        # --- BUILD 4-PANEL DEBUG COMPOSITE ---

        # 1. Original frame (with YOLO annotations via result.plot)
        if result is not None:
            orig_panel = result.plot(pil=False).copy()
        else:
            orig_panel = frame_bgr.copy()

        cv2.putText(orig_panel, "Original Frame (YOLO)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 2. Background model
        bg_panel = cv2.convertScaleAbs(camera.background_buf)
        bg_panel = cv2.cvtColor(bg_panel, cv2.COLOR_GRAY2BGR)
        cv2.putText(bg_panel, "Background Model", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 3. Diff (filtered)
        diff_panel = cv2.cvtColor(camera.diff_filtered_buf, cv2.COLOR_GRAY2BGR)
        cv2.putText(diff_panel, "Diff (Filtered)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 4. Threshold panel
        thresh_panel = cv2.cvtColor(camera.thresh_buf, cv2.COLOR_GRAY2BGR)

        # --- DRAW MOTION BOXES ON THRESH PANEL ---
        for (x1, y1, x2, y2) in krs:
            cv2.rectangle(thresh_panel, (x1, y1), (x2, y2), (0, 255, 0), 2)

        #for (x1, y1, x2, y2) in dsrs:
        #    cv2.rectangle(thresh_panel, (x1, y1), (x2, y2), (0, 165, 255), 2)

        #for (x1, y1, x2, y2) in dars:
        #    cv2.rectangle(thresh_panel, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # --- PER-CONTOUR METRICS ---
        for cnt in kcs:# + dscs + dacs:
            x, y, w0, h0 = cv2.boundingRect(cnt)

            area = cv2.contourArea(cnt)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            roi_edges = camera.edges_buf[y:y+h0, x:x+w0]
            edge_density = cv2.countNonZero(roi_edges) / max(1, (w0 * h0))

            aspect = max(w0, h0) / max(1, min(w0, h0))

            if any(cnt is kc for kc in kcs):
                color = (0, 255, 0)
            elif any(cnt is dsc for dsc in dscs):
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)

            cv2.rectangle(thresh_panel, (x, y), (x + w0, y + h0), color, 2)

            text = f"S:{solidity:.2f} E:{edge_density:.2f} A:{aspect:.1f}"
            cv2.putText(thresh_panel, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)

        # --- YOLO ANNOTATIONS ON THRESH PANEL ---
        if result is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = f"{self.model.model.names[cls_id]} {conf:.2f}"

                cv2.rectangle(thresh_panel, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(thresh_panel, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2)

        # --- DEBUG TEXT ---
        self.draw_combined_debug_layout(camera, thresh_panel)

        # --- RESIZE PANELS ---
        h, w = frame_bgr.shape[:2]
        half_w = w // 2
        half_h = h // 2

        p1 = cv2.resize(orig_panel, (half_w, half_h))
        p2 = cv2.resize(bg_panel, (half_w, half_h))
        p3 = cv2.resize(diff_panel, (half_w, half_h))
        p4 = cv2.resize(thresh_panel, (half_w, half_h))

        # --- STACK INTO 4-PANEL COMPOSITE ---
        top = np.hstack((p1, p2))
        bottom = np.hstack((p3, p4))
        composite = np.vstack((top, bottom))

        return composite

    def draw_tuner_dashboard(self, camera: Camera, panel):
        """
        Draws a live tuner dashboard overlay on the given panel.
        Shows rule hit counts, last decisions, and recommendations.
        """

        tuner = camera.auto_tuner
        stats = tuner.summarize()
        recs  = tuner.recommend_adjustments()

        # --- Dashboard box ---
        x0, y0 = 10, 10
        w, h = 420, 260
        cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (32, 32, 32), -1)
        cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (255, 255, 255), 2)

        # --- Stats Section ---
        y = y0 + 30
        cv2.putText(panel, "Rule Hits:", (x0 + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        y += 25

        # Show top 6 most frequent rules
        for rule, count in sorted(stats.items(), key=lambda x: -x[1])[:6]:
            color = (0, 255, 0) if "accepted" in rule else (0, 165, 255)
            cv2.putText(panel, f"{rule}: {count}",
                        (x0 + 20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        color, 2)
            y += 22

        # --- Recommendations Section ---
        y += 10
        cv2.putText(panel, "Recommendations:",
                    (x0 + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 2)
        y += 25

        if not recs:
            cv2.putText(panel, "All thresholds stable",
                        (x0 + 20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
        else:
            for k, v in recs.items():
                cv2.putText(panel, f"{k}: {v}",
                            (x0 + 20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 2)
                y += 22

        return panel

    def draw_combined_debug_layout(self, camera: Camera, thresh_panel):
        """
        Left column  = dbg() motion stats
        Right column = auto-tuner dashboard
        """

        h, w = thresh_panel.shape[:2]

        # -----------------------------
        # LEFT COLUMN: dbg() output
        # -----------------------------
        xL = 10
        yL = 20
        spacing = 20

        def dbg(text, color=(0,255,255)):
            nonlocal yL
            cv2.putText(
                thresh_panel, text, (xL, yL),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 2
            )
            yL += spacing

        # --- Your existing dbg() content ---
        dbg(f"recording={camera.recording}")
        dbg(f"should_record={camera.should_record}")
        dbg(f"has_moving_object={camera.has_moving_object}")
        dbg(f"score={camera.score} / {camera.profile.motion_threshold}")
        dbg(f"motion_confidence={camera.motion_confidence:.2f} / {camera.profile.motion_confidence_min}")
        dbg(f"motion_persistence={camera.motion_persistence} / {camera.profile.min_motion_frames}")
        dbg(f"total_motion_boxes={sum(map(len, [camera.motion_boxes_list]))}")
        dbg(f"pixel_score={camera.pixel_score:.2f}")
        dbg(f"box_score={camera.box_score:.2f}")
        dbg(f"persist_score={camera.persist_score:.2f}")
        dbg(f"objects={self._tags_to_str(camera.active_objects_dict)}")

        # -----------------------------
        # RIGHT COLUMN: tuner dashboard
        # -----------------------------
        tuner = camera.auto_tuner
        stats = tuner.summarize()
        recs  = tuner.recommend_adjustments()

        # Dashboard box
        dash_w = 420
        dash_h = 420
        xR = w - dash_w - 10
        yR = 10

        #cv2.rectangle(thresh_panel, (xR, yR), (xR + dash_w, yR + dash_h), (32, 32, 32), -1)
        cv2.rectangle(thresh_panel, (xR, yR), (xR + dash_w, yR + dash_h), (255, 255, 255), 2)

        # Title
        cv2.putText(thresh_panel, "AUTO-TUNER DASHBOARD",
                    (xR + 10, yR + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)

        # Rule hits
        y = yR + 60
        cv2.putText(thresh_panel, "Rule Hits:",
                    (xR + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 2)
        y += 25

        for rule, count in sorted(stats.items(), key=lambda x: -x[1])[:6]:
            color = (0, 255, 0) if "accepted" in rule else (0, 165, 255)
            cv2.putText(thresh_panel, f"{rule}: {count}",
                        (xR + 20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        color, 2)
            y += 22

        # Recommendations
        y += 10
        cv2.putText(thresh_panel, "Recommendations:",
                    (xR + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 2)
        y += 25

        if not recs:
            cv2.putText(thresh_panel, "All thresholds stable",
                        (xR + 20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 2)
        else:
            for k, v in recs.items():
                cv2.putText(thresh_panel, f"{k}: {v}",
                            (xR + 20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 255), 2)
                y += 22

        return thresh_panel


