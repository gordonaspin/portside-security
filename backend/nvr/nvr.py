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
        self.width: int = ctx.resolution[0]
        self.height: int = ctx.resolution[1]
        self.max_pixels = self.width * self.height
        
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
                                        max_pixels=self.max_pixels,
                                        logs_dir=self.ctx.log_directory,
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
            camera.profile = NightMotionProfile(self.max_pixels, self.motion_threshold, self.yolo_confidence_threshold)
        else:
            camera.profile = DayMotionProfile(self.max_pixels, self.motion_threshold, self.yolo_confidence_threshold)

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
                f"[0:v]scale={self.width}:{self.height},format=bgr24[raw]", # re-scale and raw BGR pixel format (OpenCV native)

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

    def _get_segments(self, camera: Camera, start_time: float, end_time: float):
        """
        Return all .ts segments whose timestamp falls within camera.recording_start_time and now.
        Don't add edge-case files of length zero (partial .ts file)
        """
        selected = sorted([(os.path.join(camera.segments_dir, f.name), f.stat().st_mtime)
                            for f in os.scandir(camera.segments_dir)
                            if f.name.endswith(".ts")
                            and start_time <= f.stat().st_mtime <= end_time],
                            key=lambda x: x[1])
        return [f[0] for f in selected]
    
    def _merge_segments_async(self, camera: Camera, tags: defaultdict[set], end_time: float):
        """
        Runs ffmpeg merge in a separate thread. When the process finishes,
        the log the event and delete the listing file.
        """
        adjusted_start_time = camera.recording_start_time - constants.PRE_RECORD_DURATION
        segments = self._get_segments(camera=camera, start_time=adjusted_start_time, end_time=end_time)
        tags_str = self._tags_to_str(tags)
        timestamp_str = datetime.fromtimestamp(adjusted_start_time).strftime("%Y%m%d_%H%M%S")
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
                profile = camera.profile_to_dict()
                stats = camera.auto_tuner.summarize()
                recs = camera.auto_tuner.recommend_adjustments()

                with open(metadata_filename, "w") as f:
                    json_data = {
                        "camera": camera.name,
                        "tags": serializable_tags,
                        "output": mp4_filename,
                        "start_time": adjusted_start_time,
                        "end_time": end_time,
                        "start_time_hms": datetime.fromtimestamp(adjusted_start_time - constants.PRE_RECORD_DURATION).strftime("%Y%m%d_%H%M%S"),
                        "end_time_hms": datetime.fromtimestamp(end_time).strftime("%Y%m%d_%H%M%S"),
                        "metadata": metadata_filename,
                        "segments": segments,
                        "profile": profile,
                        "tuner_stats": stats,
                        "recommendations": recs,
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
                            event.pop("profile", None)
                            event.pop("tuner_stats", None)
                            event.pop("recommendations", None)
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

        frame_size = self.width * self.height * 3

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

            frame = np.frombuffer(raw, np.uint8).reshape((self.height, self.width, 3))

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

    def _update_night_day(self, camera: Camera, frame_bgr: NDArray[np.uint8], now: float) -> None:
        """
        Periodically check whether the camera is in night mode.
        This preserves the original behavior:
        - check every PERIODIC_CHECK_INTERVAL seconds
        - update camera.is_night
        - apply the correct motion profile (day/night)
        """
        if now - camera.last_night_time_check <= constants.PERIODIC_CHECK_INTERVAL:
            return

        camera.is_night = self._is_night_time(frame_bgr, constants.NIGHT_TIME_THRESHOLD)
        self.set_camera_motion_profile(camera)
        camera.last_night_time_check = now

    def _compute_gray_and_blur(self, camera: Camera, frame_bgr: NDArray[np.uint8]) -> None:
        """
        Convert frame to grayscale and apply Gaussian blur.
        This preserves the original behavior:
        - grayscale is faster for motion detection
        - blur reduces high-frequency noise
        """
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY, dst=camera.gray_buf)
        cv2.GaussianBlur(camera.gray_buf, (21, 21), 0, dst=camera.gray_buf)

    def _update_background(self, camera: Camera) -> None:
        """
        Maintain a running background model using accumulateWeighted.
        - Night mode uses a faster alpha (0.12)
        - Day mode uses a slower alpha (0.02)
        """
        if camera.background_buf is None:
            camera.background_buf = camera.gray_buf.astype("float32")
            return

        cv2.accumulateWeighted(
            camera.gray_buf,
            dst=camera.background_buf,
            alpha=0.12 if camera.is_night else 0.02
        )

        cv2.convertScaleAbs(camera.background_buf, dst=camera.bg_frame_buf)

    def _compute_motion_diff(self, camera: Camera) -> None:
        """
        Compute absolute difference between background and current frame.
        Apply noise-adaptive thresholding and filtering.
        """
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

    def _apply_shadow_filters(self, camera: Camera) -> bool:
        """
        Apply shadow suppression filters.
        Returns True if motion should be discarded and processing should continue.
        """

        # --- GLOBAL SHADOW SWEEP ---
        if camera.white_ratio > 0.10 and camera.edge_density < camera.profile.min_edge_density(camera.noise):
            #camera.auto_tuner.record(MotionDecision(
            #    passed=False,
            #    reason="shadow_low_edge",
            #    details={"white_ratio": camera.white_ratio, "edge_density": camera.edge_density}
            #))
            camera.motion_boxes_list.clear()
            return True

        # --- SOBEL EDGES ---
        cv2.Sobel(camera.gray_buf, cv2.CV_16S, 1, 0, dst=camera.sobel_x_buf)
        cv2.Sobel(camera.gray_buf, cv2.CV_16S, 0, 1, dst=camera.sobel_y_buf)
        cv2.convertScaleAbs(camera.sobel_x_buf, dst=camera.sobel_x_abs_buf)
        cv2.convertScaleAbs(camera.sobel_y_buf, dst=camera.sobel_y_abs_buf)
        cv2.addWeighted(camera.sobel_x_abs_buf, 0.5, camera.sobel_y_abs_buf, 0.5, 0, dst=camera.edges_buf)

        camera.edge_density = cv2.countNonZero(camera.edges_buf) / camera.max_pixels

        # --- LOW-EDGE SHADOW ---
        if camera.white_ratio > 0.10 and camera.edge_density < 0.02:
            #camera.auto_tuner.record(MotionDecision(
            #    passed=False,
            #    reason="shadow_low_edge2",
            #    details={"white_ratio": camera.white_ratio, "edge_density": camera.edge_density}
            #))
            camera.motion_boxes_list.clear()
            return True

        return False

    def _find_motion(self, camera: Camera) -> list[tuple[int, int, int, int]]:
        """
        Extract motion contours and bounding boxes.
        This preserves your original behavior:
        - only run if score > motion_threshold
        - apply contour filtering inside _find_motion_boxes()
        """
        camera.motion_boxes_list.clear()

        if camera.score <= camera.profile.motion_threshold:
            return []

        # krs = kept rectangles
        krs, kcs, dsrs, dscs, dars, dacs = self._find_motion_boxes(camera)
        camera.motion_boxes_list.extend(krs)

        # --- TOTAL MOTION AREA CHECK ---
        total_motion_area = sum(
            (x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in camera.motion_boxes_list
        )

        if total_motion_area < camera.profile.min_total_motion_area:
            #camera.auto_tuner.record(MotionDecision(
            #    passed=False,
            #    reason="low_total_area",
            #    details={"total_motion_area": total_motion_area}
            #))
            camera.motion_boxes_list.clear()

        return camera.motion_boxes_list
    
    def _run_yolo_if_needed(
        self,
        camera: Camera,
        frame_bgr: NDArray[np.uint8],
        motion_boxes: list[tuple[int, int, int, int]]
    ) -> Results | None:
        """
        Run YOLO only when:
        - debug mode is enabled, OR
        - there is motion AND motion_confidence > 0.05

        This preserves your original behavior exactly.
        """
        if not (camera.debug or (motion_boxes and camera.motion_confidence > 0.05)):
            return None

        result: Results = camera.model.model.predict(
            frame_bgr,
            conf=camera.profile.yolo_confidence_threshold,
            classes=self.selected_classes if self.selected_classes else None,
            verbose=False,
            imgsz=512,
        )[0]

        return result

    def _filter_yolo_overlaps(
        self,
        camera: Camera,
        result: Results | None,
        motion_boxes: list[tuple[int, int, int, int]]
    ) -> None:
        """
        Filter YOLO detections to only those overlapping motion boxes.
        This preserves your original behavior:
        - inflate motion boxes
        - compute overlaps
        - build keep_mask
        - update camera.has_moving_object
        - record tuner events for YOLO overlap noise
        """
        camera.classes_in_frame_dict.clear()
        camera.has_moving_object = False
        camera.keep_mask = []

        if result is None:
            return

        # Extract YOLO boxes
        yolo_boxes = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            yolo_boxes.append((int(x1), int(y1), int(x2), int(y2)))

        # Inflate motion boxes
        inflated_motion_boxes = [
            self._inflate_box(b, camera.profile.inflate_motion_boxes)
            for b in motion_boxes
        ]

        # Determine which YOLO boxes overlap motion
        moving_yolo_indices: set[int] = set()

        for i, yb in enumerate(yolo_boxes):
            for mb in inflated_motion_boxes:
                if self._boxes_overlap(mb, yb):
                    moving_yolo_indices.add(i)
                    break

        # Build keep mask
        keep_mask = [i in moving_yolo_indices for i in range(len(yolo_boxes))]
        camera.keep_mask = keep_mask
        camera.has_moving_object = any(keep_mask)

        # Tuner: YOLO overlap noise
        # YOLO misses are common and NOT motion noise → ignore completely
        if motion_boxes and not camera.has_moving_object:
            # Do NOT record tuner event
            # Do NOT collapse confidence
            pass

        # Filter YOLO results
        result.boxes = result.boxes[keep_mask]

        # Extract class + color for each kept detection
        for box in result.boxes:
            class_name = self.model.model.names[int(box.cls)]
            roi = self.yolo_box_to_roi(camera.latest_frame, box)
            if roi.size > 0:
                color = self._detect_object_color(roi)
                camera.classes_in_frame_dict[class_name].add(color)

    def _apply_fast_stop_logic(
        self,
        camera: Camera,
        motion_boxes: list[tuple[int, int, int, int]],
        now: float
    ) -> None:
        """
        Apply fast-stop logic to reduce recording stop latency.
        This preserves your improved behavior:
        - hard reset when motion disappears
        - collapse confidence quickly
        - decay confidence when YOLO sees nothing
        """

        # --- HARD RESET WHEN MOTION DISAPPEARS ---
        if not motion_boxes:
            camera.motion_persistence = 0
            camera.persist_score = 0.0
            camera.motion_confidence = min(camera.motion_confidence, 0.05)
            #camera.last_motion_time = now
            return

        # --- ADDITIONAL CONFIDENCE DECAY WHEN YOLO SEES NOTHING ---
        if camera.recording and not camera.has_moving_object:
            camera.motion_confidence *= 0.5
            # apply decay twice if still above STOP_CONF
            stop_conf = camera.profile.motion_confidence_min * 0.5
            if camera.motion_confidence > stop_conf:
                camera.motion_confidence *= 0.5

    def _should_start_recording(
        self,
        camera: Camera,
        motion_boxes: list[tuple[int, int, int, int]],
        now: float
    ) -> bool:
        """
        Determine whether recording should start.
        Preserves original logic:
        - motion_persistence >= min_motion_frames
        - YOLO sees moving objects
        - object area >= min_sum_box_area
        - motion_confidence >= START_CONF
        """

        object_area = sum((x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in motion_boxes)

        motion_is_persistent = (
            camera.motion_persistence >= camera.profile.min_motion_frames
        )

        object_motion = (
            motion_is_persistent and
            camera.has_moving_object
        )

        object_area_ok = (
            object_area >= camera.profile.min_sum_box_area
        )

        START_CONF = camera.profile.motion_confidence_min

        return (
            object_motion and
            object_area_ok and
            camera.motion_confidence >= START_CONF
        )

    def _should_continue_recording(self, camera: Camera) -> bool:
        """
        Continue recording if:
        - confidence >= STOP_CONF (hysteresis), OR
        - motion persistence window is active.

        Post-record timing is handled in _update_recording_state.
        """

        START_CONF = camera.profile.motion_confidence_min
        STOP_CONF  = max(0.10, START_CONF * 0.30)

        if not camera.recording:
            return False

        # 1. Hysteresis
        if camera.motion_confidence >= STOP_CONF:
            return True

        # 2. Persistence window
        if camera.motion_persistence >= camera.profile.min_motion_frames:
            return True

        return False


    def _update_recording_state(
        self,
        camera: Camera,
        motion_boxes: list[tuple[int, int, int, int]],
        now: float
    ) -> None:
        """
        Full recording state machine:
        - start recording
        - continue recording
        - stop recording
        - update last_motion_time only when real motion exists
        - update active object tags
        - record tuner events
        """

        camera.should_start = self._should_start_recording(camera, motion_boxes, now)
        camera.should_continue = self._should_continue_recording(camera)

        camera.should_record = camera.should_start or camera.should_continue

        # Only update last_motion_time when real motion + YOLO objects exist
        if (
            motion_boxes and
            camera.has_moving_object and
            camera.motion_confidence >= camera.profile.motion_confidence_min
        ):
            camera.last_motion_time = now

        # --- TUNER: insufficient persistence ---
        if motion_boxes and camera.motion_persistence < camera.profile.min_motion_frames:
            camera.auto_tuner.record(MotionDecision(
                passed=False,
                reason="short_motion",
                details={"persistence": camera.motion_persistence}
            ))

        # --- TUNER: insufficient confidence ---
        START_CONF = camera.profile.motion_confidence_min
        if motion_boxes and camera.motion_confidence < START_CONF:
            pass
            #camera.auto_tuner.record(MotionDecision(
            #    passed=False,
            #    reason="low_confidence",
            #    details={"confidence": camera.motion_confidence}
            #))

        # --- START RECORDING ---
        if camera.should_start and not camera.recording:
            camera.recording = True
            camera.recording_start_time = now
            camera.active_objects_dict = deepcopy(camera.classes_in_frame_dict)
            camera.last_recording_time = now

            #camera.auto_tuner.record(MotionDecision(
            #    passed=True,
            #    reason="recording_start",
            #    details={"confidence": camera.motion_confidence}
            #))

            log_event(
                message=f"recording start {self._tags_to_str(camera.active_objects_dict)}",
                level="info",
                camera=camera
            )

        # --- CONTINUE RECORDING ---
        if camera.recording:
            # Merge new object colors into active set
            for item, colors in camera.classes_in_frame_dict.items():
                camera.active_objects_dict[item].update(colors)

            #camera.auto_tuner.record(MotionDecision(
            #    passed=True,
            #    reason="recording_continue",
            #    details={"confidence": camera.motion_confidence}
            #))

        # --- STOP RECORDING ---
        if camera.recording and not camera.should_continue:
            if now - camera.last_motion_time > constants.POST_RECORD_DURATION:
                camera.recording = False
                tags = deepcopy(camera.active_objects_dict)

                # Merge segments asynchronously
                self._merge_segments_async(camera, tags, now)

                #camera.auto_tuner.record(MotionDecision(
                #    passed=True,
                #    reason="recording_stop",
                #    details={"confidence": camera.motion_confidence}
                #))

                # Reset state
                camera.classes_in_frame_dict.clear()
                camera.active_objects_dict.clear()
                camera.motion_frames = 0
                camera.no_motion_frames = 0
                camera.motion_persistence = 0

    def _render_debug_ui(
        self,
        camera: Camera,
        frame_bgr: NDArray[np.uint8],
        yolo_result: Results | None
    ) -> None:
        """
        Render debug UI panels if debug mode is enabled.
        This preserves your original behavior:
        - draw motion boxes
        - draw YOLO overlays
        - draw tuner dashboard
        - combine into a single debug mosaic
        """
        if not camera.debug:
            return

        camera.debug_motion_image = self.draw_debug_panels(
            camera,
            frame_bgr,
            yolo_result,
            camera.motion_boxes_list,
            [], [], [], [], []  # placeholders for krs/kcs/dsrs/dscs/dars/dacs
        )

        if camera.recording:
            timestamp_str = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S_%f")
            filename_str = os.path.join(camera.images_dir, f"{timestamp_str}.jpg")
            cv2.imwrite(filename_str, camera.debug_motion_image)


    def _draw_status_text(
        self,
        camera: Camera,
        frame_bgr: NDArray[np.uint8]
    ) -> None:
        """
        Draw status text and object text on the frame.
        Preserves original colors and shadow styling.
        """
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

    def _apply_yolo_overlay(
        self,
        frame_bgr: NDArray[np.uint8],
        yolo_result: Results | None
    ) -> NDArray[np.uint8]:
        """
        Apply YOLO overlay if results exist.
        Preserves original behavior:
        - result.plot(pil=False) returns BGR
        """
        if yolo_result is None:
            return frame_bgr

        return yolo_result.plot(pil=False)

    def _select_debug_frame(
        self,
        camera: Camera,
        frame_bgr: NDArray[np.uint8],
        yolo_result: Results | None
    ) -> NDArray[np.uint8]:
        """
        Choose the final debug frame:
        - if debug panels exist → use them
        - else if YOLO results exist → use YOLO overlay
        - else → use raw frame
        """
        if camera.debug and camera.debug_motion_image is not None:
            return camera.debug_motion_image

        if yolo_result is not None:
            return yolo_result.plot(pil=False)

        return frame_bgr

    def _finalize_output(
        self,
        camera: Camera,
        frame_bgr: NDArray[np.uint8],
        yolo_result: Results | None
    ) -> None:
        """
        Finalize the output frame for the GUI.
        This preserves your original behavior:
        - choose debug mosaic if available
        - else choose YOLO overlay
        - else use raw frame
        - update status text and objects text
        - store final frame in camera.latest_frame
        """
        # Draw status text on the raw frame BEFORE selecting debug/YOLO overlays
        self._draw_status_text(camera, frame_bgr)

        # Select final frame (debug mosaic > YOLO overlay > raw)
        final_frame = self._select_debug_frame(camera, frame_bgr, yolo_result)

        # Update GUI-visible frame
        camera.latest_frame = final_frame

        # Build status text
        parts = [self._make_status(camera)]
        if camera.is_night:
            parts.append("Night")

        parts.append(f"FPS {int(camera.fps.value())}:{camera.drop_rate:.2f}")

        camera.objects_text = self._tags_to_str(camera.active_objects_dict)
        camera.status_text = " | ".join(parts)


    def _select_debug_frame(
        self,
        camera: Camera,
        frame_bgr: NDArray[np.uint8],
        yolo_result: Results | None
    ) -> NDArray[np.uint8]:
        """
        Choose the final debug frame:
        - if debug panels exist → use them
        - else if YOLO results exist → use YOLO overlay
        - else → use raw frame
        """
        if camera.debug and camera.debug_motion_image is not None:
            return camera.debug_motion_image

        if yolo_result is not None:
            return yolo_result.plot(pil=False)

        return frame_bgr


    def _apply_yolo_overlay(
        self,
        frame_bgr: NDArray[np.uint8],
        yolo_result: Results | None
    ) -> NDArray[np.uint8]:
        """
        Apply YOLO overlay if results exist.
        Preserves original behavior:
        - result.plot(pil=False) returns BGR
        """
        if yolo_result is None:
            return frame_bgr

        return yolo_result.plot(pil=False)

    def _process_frames(self, camera: Camera):
        """
        Thread to process frames from the camera queue. Processing is as follows:
        - get the frame from the queue (latest frame processing, some frames are dropped)
        - convert to grayscale for image processing (faster than color)
        - blur the gray (better for motion detection)
        - calculate the difference between this gray frame and the previous one (for motion detection)
        - calculate a threshold image based on the difference and score (count) the white pixels
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

            # --- FRAME ACQUISITION ---
            frame_bgr = camera.get_frame()
            if frame_bgr is None:
                continue

            now = time.time()

            # --- ENVIRONMENT UPDATES ---
            self._update_night_day(camera, frame_bgr, now)
            camera.auto_adjust_if_needed(now)

            # --- MOTION PIPELINE ---
            self._compute_gray_and_blur(camera, frame_bgr)
            self._update_background(camera)
            self._compute_motion_diff(camera)
            if self._apply_shadow_filters(camera):
                continue

            motion_boxes = self._find_motion(camera)

            # --- YOLO PIPELINE ---
            yolo_result = self._run_yolo_if_needed(camera, frame_bgr, motion_boxes)
            self._filter_yolo_overlaps(camera, yolo_result, motion_boxes)

            # --- CONFIDENCE + PERSISTENCE ---
            camera.update_confidence(motion_boxes, now)
            camera.update_persistence(motion_boxes)
            self._apply_fast_stop_logic(camera, motion_boxes, now)

            # --- RECORDING LOGIC ---
            self._update_recording_state(camera, motion_boxes, now)

            # --- DEBUG UI ---
            self._render_debug_ui(camera, frame_bgr, yolo_result)

            # --- FINAL OUTPUT ---
            self._finalize_output(camera, frame_bgr, yolo_result)


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

                #tuner.record(MotionDecision(
                #    passed=False,
                #    reason="small_contour",
                #    details={"area": area, "min_area": min_area}
                #))
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
                ), camera=camera)
                continue

            # --- MINIMUM WIDTH/HEIGHT FILTER ---
            x, y, w0, h0 = cv2.boundingRect(cnt)
            if w0 < min_w or h0 < min_h:
                small_rects.append((x, y, x + w0, y + h0))
                small_contours.append(cnt)

                #tuner.record(MotionDecision(
                #    passed=False,
                #    reason="small_dimensions",
                #    details={"w": w0, "h": h0, "min_w": min_w, "min_h": min_h}
                #), camera=camera)
                continue

            # --- SOBEL EDGE DENSITY FILTER ---
            roi_edges = edges[y:y+h0, x:x+w0]
            edge_count = cv2.countNonZero(roi_edges)
            edge_density = edge_count / max(1, (w0 * h0))

            if edge_density < min_edge_density:
                angular_rects.append((x, y, x + w0, y + h0))
                angular_contours.append(cnt)

                #tuner.record(MotionDecision(
                #    passed=False,
                #    reason="low_edge_density",  # <-- matches tuner rule
                #    details={"edge_density": edge_density, "min_edge_density": min_edge_density}
                #), camera=camera)
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
                ), camera=camera)
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
            ), camera=camera)

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
        x2 = min(self.width - 1, x2 + inflate_px)
        y2 = min(self.height - 1, y2 + inflate_px)
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

        self._draw_status_text(camera, orig_panel)

        TITLE_Y = 40
        cv2.putText(orig_panel, "Original Frame (YOLO)", (10, TITLE_Y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 2. Background model
        bg_panel = cv2.convertScaleAbs(camera.background_buf)
        bg_panel = cv2.cvtColor(bg_panel, cv2.COLOR_GRAY2BGR)
        cv2.putText(bg_panel, "Background Model", (10, TITLE_Y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 3. Diff (filtered)
        diff_panel = cv2.cvtColor(camera.diff_filtered_buf, cv2.COLOR_GRAY2BGR)
        cv2.putText(diff_panel, "Diff (Filtered)", (10, TITLE_Y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 4. Threshold panel
        thresh_panel = cv2.cvtColor(camera.thresh_buf, cv2.COLOR_GRAY2BGR)
        cv2.putText(thresh_panel, "Threshold", (10, TITLE_Y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
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
        yL = 60
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
        dbg(f"should_start={camera.should_start}")
        dbg(f"should_continue={camera.should_continue}")

        dbg(f"has_moving_object={camera.has_moving_object}")
        dbg(f"motion_boxes={len(camera.motion_boxes_list)}")

        dbg(f"score={camera.score} / {camera.profile.motion_threshold}")
        dbg(f"motion_confidence={camera.motion_confidence:.2f} / {camera.profile.motion_confidence_min}")
        dbg(f"STOP_CONF={max(0.10, camera.profile.motion_confidence_min * 0.30):.2f}")

        dbg(f"motion_persistence={camera.motion_persistence} / {camera.profile.min_motion_frames}")
        dbg(f"persist_score={camera.persist_score:.2f}")

        dbg(f"since_last_motion={time.time() - camera.last_motion_time:.2f}s")
        dbg(f"pixel_score={camera.pixel_score:.2f}")
        dbg(f"box_score={camera.box_score:.2f}")

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


