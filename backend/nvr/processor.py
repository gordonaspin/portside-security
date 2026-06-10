import os
import time
from datetime import datetime
from copy import deepcopy
from logging import getLogger
from threading import Thread, Event, current_thread
from typing import Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO
from ultralytics.engine.results import Results

import constants
from logger.logger import log_event
from nvr.debug_panel import draw_status_text, draw_debug_panels
from nvr.camera.camera import Camera
from nvr.camera.motion_tuner import MotionDecision
from reader.rtsp_reader import Reader
from recorder.factory import FrameRecorderFactory
from recorder.recorders import Recorder
from utils.thread_safe import ThreadSafeList
from utils.utils import (
    make_readable_ts,
    tags_to_str,
    boxes_overlap,
    detect_object_color)

logger = getLogger("pynvr.processor")

class FrameProcessor():
    def __init__(
        self,
        camera: Camera,
        reader: Reader,
        recorder_factory: FrameRecorderFactory,
        model_cfg: dict[str, str],
        stop_event: Event,
        recordings: ThreadSafeList
    ):
        self.camera: Camera = camera
        self.reader: Reader = reader
        self.recorder_factory: FrameRecorderFactory = recorder_factory
        self.model: YOLO = YOLO(model_cfg["name"])
        classname_to_classindex: dict = {v: k for k, v in self.model.names.items()}
        self.selected_classes: list[int] = [classname_to_classindex[n] for n in model_cfg["classes"]]

        self.stop_event: Event = stop_event
        self.recordings: ThreadSafeList = recordings
        self.thread: Thread = None
        self.recorder: Recorder = self.recorder_factory.create(self.camera, self.stop_event, self.recordings)
        self.frame_count: int = 0
        self.status_text: str = "Not streaming"
        self.objects_text: str = ""
        self.last_night_time_check: float = time.time()

    def start(self):
        self.thread = Thread(target=self._process_frames, daemon=True)
        self.thread.start()

    def stop(self):
        log_event(message="stopping frame processor", level="info", camera=self.camera)
        self.thread.join()

    def _process_frames(self):
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

        current_thread().name = f"{self.camera.config.name} processor"

        while not self.stop_event.is_set() and not self.reader.process.stdout.closed:

            # --- FRAME ACQUISITION ---
            frame_bgr = self.reader.get_frame()
            if frame_bgr is None:
                continue

            yolo_frame = self.camera.buffers.yolo_frame
            if yolo_frame is None:
                continue

            full_h, full_w = frame_bgr.shape[:2]
            yolo_h, yolo_w = yolo_frame.shape[:2]

            now = time.time()
            self.frame_count += 1

            # --- ENVIRONMENT UPDATES ---
            self._update_night_day(frame_bgr, now)
            self.camera.tuner.maybe_auto_adjust(now)

            # --- MOTION PIPELINE ---
            self._compute_gray_and_blur(frame_bgr)
            self._update_background()
            if time.time() - self.camera.start_time < constants.DELAY_FIRST_RECORDING_SECONDS:
                continue
            
            self._compute_motion_diff()
            if self._apply_shadow_filters():
                continue

            motion_boxes = self._find_motion()

            # --- YOLO PIPELINE (runs on yolo_frame) ---
            yolo_result = self._run_yolo_if_needed(yolo_frame, motion_boxes)

            # --- Scale YOLO boxes to full resolution ---
            self._scale_yolo_boxes_to_full_res(
                yolo_result,
                full_w, full_h,
                yolo_w, yolo_h
            )

            # --- Filter YOLO detections based on motion overlap ---
            self._filter_yolo_overlaps(frame_bgr, yolo_result, motion_boxes, full_w, full_h)

            # --- CONFIDENCE + PERSISTENCE ---
            self.camera.motion.update_confidence(motion_boxes, now)
            self.camera.motion.update_persistence(motion_boxes)
            self._apply_fast_stop_logic(motion_boxes, now)

            if self.recorder.fps.as_int() > 0 and self.frame_count > constants.DELAY_FIRST_RECORDING_SECONDS * self.recorder.fps.as_int():
                # --- RECORDING LOGIC ---
                self._update_recording_state(motion_boxes, now)

            # 1. Build status text strings
            self._update_status_strings()

            # 2. Build debug UI if camera.debug (uses annotated frame)
            self._render_debug_ui(frame_bgr, yolo_result)

            # 3. Finalize output (select debug > YOLO > raw)
            self._finalize_output(frame_bgr, yolo_result)
            
            self.recorder.add_frame(self.camera.latest_frame)

    def _update_status_strings(self):
        """
        creates a string that represents the status (red/green for recording/live)
        """
        idx = int(time.time() * 4) % 4
        record_cycle = ["*", "*", " ", " "]
        pulse = record_cycle[idx] if self.camera.recording_state.recording else ""

        status = f"{pulse}{'REC' if self.camera.recording_state.recording else 'LIVE'}"

        parts = [status]
        if self.camera.is_night:
            parts.append("Night")

        parts.append(f"FPS {int(self.recorder.fps.as_int())}/{int(self.reader.fps.as_int())}")
        parts.append(make_readable_ts(time.time()))

        self.objects_text = tags_to_str(self.camera.motion.active_objects_dict)
        self.status_text = " | ".join(parts)

    def _update_night_day(self, frame_bgr: NDArray[np.uint8], now: float) -> None:
        """
        Periodically check whether the camera is in night mode.
        This preserves the original behavior:
        - check every PERIODIC_CHECK_INTERVAL seconds
        - update camera.is_night
        - apply the correct motion profile (day/night)
        """
        if now - self.last_night_time_check <= constants.PERIODIC_CHECK_INTERVAL:
            return

        self.camera.is_night = self._is_night_time(frame_bgr)
        self.camera.motion.profile = self.camera.motion.night_profile if self.camera.is_night else self.camera.motion.day_profile
        self.last_night_time_check = now

    def _is_night_time(self, frame,
                    luma_threshold=90,
                    ir_chroma_threshold=4.0):
        """
        night detector for NVR use.
        Uses 2 signals:
        - avg luma (Y channel)
        - IR mode detection (RGB channel collapse)
        """

        if frame is None or frame.size == 0:
            return True

        # --- Compute luma ---
        avg_luma = self.camera.buffers.gray_buf.mean()

        # --- Detect IR mode (RGB channels collapse) ---
        b, g, r = cv2.split(frame.astype(np.float32))
        chroma = np.mean(np.abs(r - g)) + np.mean(np.abs(g - b))
        ir_mode_on = chroma < ir_chroma_threshold

        # --- Final decision ---
        if (avg_luma < luma_threshold) or ir_mode_on:
            return True

        return False

    def _compute_gray_and_blur(self, frame_bgr: NDArray[np.uint8]) -> None:
        """
        Convert frame to grayscale and apply Gaussian blur.
        This preserves the original behavior:
        - grayscale is faster for motion detection
        - blur reduces high-frequency noise
        """
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY, dst=self.camera.buffers.gray_buf)
        cv2.GaussianBlur(self.camera.buffers.gray_buf, (21, 21), 0, dst=self.camera.buffers.gray_buf)

    def _update_background(self) -> None:
        """
        Maintain a running background model using accumulateWeighted.
        - Night mode uses a faster alpha (0.12)
        - Day mode uses a slower alpha (0.02)
        """
        if self.camera.buffers.background_buf is None:
            self.camera.buffers.background_buf = self.camera.buffers.gray_buf.astype("float32")
            return

        cv2.accumulateWeighted(
            self.camera.buffers.gray_buf,
            dst=self.camera.buffers.background_buf,
            alpha=0.12 if self.camera.is_night else 0.02
        )

        cv2.convertScaleAbs(self.camera.buffers.background_buf, dst=self.camera.buffers.bg_frame_buf)

    def _compute_motion_diff(self) -> None:
        """
        Compute absolute difference between background and current frame.
        Apply noise-adaptive thresholding and filtering.
        """
        # --- MOTION DIFF ---
        cv2.absdiff(self.camera.buffers.bg_frame_buf, self.camera.buffers.gray_buf, dst=self.camera.buffers.diff_buf)

        # --- NOISE-ADAPTIVE LOW-INTENSITY FILTERING ---
        self.camera.motion.noise = np.std(self.camera.buffers.diff_buf)
        cutoff = max(8, min(20, self.camera.motion.noise * 1.5))

        cv2.threshold(self.camera.buffers.diff_buf, cutoff, 255, cv2.THRESH_BINARY, dst=self.camera.buffers.diff_mask_buf)
        cv2.bitwise_and(self.camera.buffers.diff_buf, self.camera.buffers.diff_mask_buf, dst=self.camera.buffers.diff_filtered_buf)

        # --- BLUR TO REDUCE HIGH-FREQUENCY NOISE ---
        cv2.GaussianBlur(self.camera.buffers.diff_filtered_buf, (7, 7), 0, dst=self.camera.buffers.diff_blur_buf)

        # --- OTSU THRESHOLD ON CLEANED DIFF ---
        cv2.threshold(
            self.camera.buffers.diff_blur_buf, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            dst=self.camera.buffers.thresh_buf
        )

        self.camera.motion.score = cv2.countNonZero(self.camera.buffers.thresh_buf)
        self.camera.recording_state.white_ratio = self.camera.motion.score / self.camera.config.max_pixels

    def _apply_shadow_filters(self) -> bool:
        """
        Apply shadow suppression filters.
        Returns True if motion should be discarded and processing should continue.
        """

        # --- GLOBAL SHADOW SWEEP ---
        if self.camera.recording_state.white_ratio > 0.10 and self.camera.motion.edge_density < self.camera.motion.profile.min_edge_density(self.camera.motion.noise):
            self.camera.motion.motion_boxes_list.clear()
            return True

        # --- SOBEL EDGES ---
        cv2.Sobel(self.camera.buffers.gray_buf, cv2.CV_16S, 1, 0, dst=self.camera.buffers.sobel_x_buf)
        cv2.Sobel(self.camera.buffers.gray_buf, cv2.CV_16S, 0, 1, dst=self.camera.buffers.sobel_y_buf)
        cv2.convertScaleAbs(self.camera.buffers.sobel_x_buf, dst=self.camera.buffers.sobel_x_abs_buf)
        cv2.convertScaleAbs(self.camera.buffers.sobel_y_buf, dst=self.camera.buffers.sobel_y_abs_buf)
        cv2.addWeighted(self.camera.buffers.sobel_x_abs_buf, 0.5, self.camera.buffers.sobel_y_abs_buf, 0.5, 0, dst=self.camera.buffers.edges_buf)

        self.camera.motion.edge_density = cv2.countNonZero(self.camera.buffers.edges_buf) / self.camera.config.max_pixels

        # --- LOW-EDGE SHADOW ---
        if self.camera.recording_state.white_ratio > 0.10 and self.camera.motion.edge_density < 0.02:
            self.camera.motion.motion_boxes_list.clear()
            return True

        return False

    def _find_motion(self) -> list[tuple[int, int, int, int]]:
        """
        Extract motion contours and bounding boxes.
        This preserves your original behavior:
        - only run if score > motion_threshold_pixels
        - apply contour filtering inside _find_motion_boxes()
        """
        self.camera.motion.motion_boxes_list.clear()

        if self.camera.motion.score <= self.camera.motion.profile.motion_threshold_pixels:
            return []

        # krs = kept rectangles
        krs, kcs, dsrs, dscs, dars, dacs = self._find_motion_boxes()
        self.camera.motion.motion_boxes_list.extend(krs)

        # --- TOTAL MOTION AREA CHECK ---
        total_motion_area = sum(
            (x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in self.camera.motion.motion_boxes_list
        )

        if total_motion_area < self.camera.motion.profile.min_total_motion_area:
            self.camera.motion.motion_boxes_list.clear()

        return self.camera.motion.motion_boxes_list

    def _find_motion_boxes(self):
        """
        Find motion boxes using contour analysis with solidity filtering.
        Returns:
            kept_rects, kept_contours,
            small_rects, small_contours,
            angular_rects, angular_contours
        """

        tuner = self.camera.tuner.tuner  # shorthand

        # Profile thresholds
        min_solidity = self.camera.motion.profile.min_contour_solidity
        min_w = self.camera.motion.profile.min_box_width
        min_h = self.camera.motion.profile.min_box_height
        min_edge_density = self.camera.motion.profile.min_edge_density(self.camera.motion.noise)
        max_aspect = self.camera.motion.profile.max_allowed_aspect_ratio
        min_contour_area_ratio = self.camera.motion.profile.min_contour_area_ratio

        thresh = self.camera.buffers.thresh_buf
        edges = self.camera.buffers.edges_buf

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
                ), camera=self.camera)
                continue

            # --- MINIMUM WIDTH/HEIGHT FILTER ---
            x, y, w0, h0 = cv2.boundingRect(cnt)
            if w0 < min_w or h0 < min_h:
                small_rects.append((x, y, x + w0, y + h0))
                small_contours.append(cnt)
                continue

            # --- SOBEL EDGE DENSITY FILTER ---
            roi_edges = edges[y:y+h0, x:x+w0]
            edge_count = cv2.countNonZero(roi_edges)
            edge_density = edge_count / max(1, (w0 * h0))

            if edge_density < min_edge_density:
                angular_rects.append((x, y, x + w0, y + h0))
                angular_contours.append(cnt)
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
                ), camera=self.camera)
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
            ), camera=self.camera)

        return (
            kept_rects, kept_contours,
            small_rects, small_contours,
            angular_rects, angular_contours
        )


    def _run_yolo_if_needed(
        self,
        yolo_frame: NDArray[np.uint8],
        motion_boxes: list[tuple[int, int, int, int]]
    ) -> Results | None:

        if not (self.camera.config.debug or (motion_boxes and self.camera.motion.motion_confidence > 0.05)):
            return None

        result: Results = self.model.predict(
            yolo_frame,
            conf=self.camera.motion.profile.yolo_confidence_threshold.value,
            classes=self.selected_classes if self.selected_classes else None,
            verbose=False,
            imgsz=max(yolo_frame.shape[0], yolo_frame.shape[1]),
        )[0]

        return result

    def _filter_yolo_overlaps(
        self,
        frame_bgr,
        result: Results | None,
        motion_boxes: list[tuple[int, int, int, int]], full_w: int, full_h: int
    ) -> None:
        """
        Filter YOLO detections to only those overlapping motion boxes.
        This preserves your original behavior:
        - inflate motion boxes
        - compute overlaps
        - build keep_mask
        - update camera.motion.has_moving_object
        - record tuner events for YOLO overlap noise
        """
        self.camera.motion.classes_in_frame_dict.clear()
        self.camera.motion.has_moving_object = False

        if result is None:
            return

        # Extract YOLO boxes
        yolo_boxes = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            yolo_boxes.append((int(x1), int(y1), int(x2), int(y2)))

        # Inflate motion boxes
        inflated_motion_boxes = [
            self._inflate_box(b, self.camera.motion.profile.inflate_motion_boxes, frame_w=full_w, frame_h=full_h)
            for b in motion_boxes
        ]

        # Determine which YOLO boxes overlap motion
        moving_yolo_indices: set[int] = set()

        for i, yb in enumerate(yolo_boxes):
            for mb in inflated_motion_boxes:
                if boxes_overlap(mb, yb):
                    moving_yolo_indices.add(i)
                    break

        # Build keep mask
        keep_mask = [i in moving_yolo_indices for i in range(len(yolo_boxes))]
        self.camera.motion.has_moving_object = any(keep_mask)

        # Tuner: YOLO overlap noise
        # YOLO misses are common and NOT motion noise → ignore completely
        if motion_boxes and not self.camera.motion.has_moving_object:
            # Do NOT record tuner event
            # Do NOT collapse confidence
            pass

        # Filter YOLO results
        result.boxes = result.boxes[keep_mask]

        full_h, full_w = frame_bgr.shape[:2]
        # Extract class + color for each kept detection
        for box in result.boxes:
            x1 = max(0, min(full_w - 1, int(box.xyxy[0][0])))
            y1 = max(0, min(full_h - 1, int(box.xyxy[0][1])))
            x2 = max(0, min(full_w - 1, int(box.xyxy[0][2])))
            y2 = max(0, min(full_h - 1, int(box.xyxy[0][3])))

            roi = frame_bgr[y1:y2, x1:x2]
            if roi.size > 0:
                class_name = self.model.names[int(box.cls)]
                color = detect_object_color(roi, self.camera.is_night)
                self.camera.motion.classes_in_frame_dict[class_name].add(color)

    def _inflate_box(self, box, inflate_px, frame_w, frame_h):
        """
        Inflate a bounding box by inflate_px pixels on all sides.
        Accepts either:
            - a YOLO box tensor (xyxy)
            - a tuple/list (x1, y1, x2, y2)
        Returns a tuple (x1, y1, x2, y2) clamped to frame bounds.
        """

        # Normalize input
        if hasattr(box, "xyxy"):   # YOLO Boxes object
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy.tolist()
        else:
            x1, y1, x2, y2 = box

        x1 -= inflate_px
        y1 -= inflate_px
        x2 += inflate_px
        y2 += inflate_px

        # Clamp to frame bounds
        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(0, min(frame_w - 1, x2))
        y2 = max(0, min(frame_h - 1, y2))

        return (x1, y1, x2, y2)


    def _scale_yolo_boxes_to_full_res(
        self,
        result: Results | None,
        full_w: int,
        full_h: int,
        yolo_w: int,
        yolo_h: int
    ) -> None:

        if result is None or result.boxes is None or len(result.boxes) == 0:
            return

        boxes = result.boxes

        # Clone the entire data tensor (N x 6 or N x 7 depending on model)
        data = boxes.data.clone()

        # Extract xyxy (first 4 columns)
        xyxy = data[:, :4]

        scale_x = full_w / yolo_w
        scale_y = full_h / yolo_h

        scale = xyxy.new_tensor([scale_x, scale_y, scale_x, scale_y])

        # Vectorized scaling (no in-place ops)
        xyxy = xyxy * scale

        # Replace the first 4 columns with the scaled coords
        data[:, :4] = xyxy

        # Replace the entire tensor (allowed)
        boxes.data = data

    def _apply_fast_stop_logic(
        self,
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
            self.camera.motion.motion_persistence = 0
            self.camera.motion.persist_score = 0.0
            self.camera.motion.motion_confidence = min(self.camera.motion.motion_confidence, 0.05)
            return

        # --- ADDITIONAL CONFIDENCE DECAY WHEN YOLO SEES NOTHING ---
        if self.camera.recording_state.recording and not self.camera.motion.has_moving_object:
            self.camera.motion.motion_confidence *= 0.5
            # apply decay twice if still above STOP_CONF
            stop_conf = self.camera.motion.profile.min_motion_confidence.value * 0.5
            if self.camera.motion.motion_confidence > stop_conf:
                self.camera.motion.motion_confidence *= 0.5

    def _update_recording_state(
        self,
        motion_boxes: list[tuple[int, int, int, int]],
        now: float
    ) -> None:
        """
        Clean, final recording state machine:
        - decide start
        - decide continue
        - stop after POST_RECORD_DURATION with no real motion
        - update active object tags
        """

        # --- DECIDE START / CONTINUE ---
        self.camera.recording_state.should_record = self._should_start_recording(motion_boxes, now)
        self.camera.recording_state.should_continue = self._should_continue_recording()

        # --- UPDATE last_motion_time WHEN REAL MOTION EXISTS ---
        if (
            motion_boxes and
            self.camera.motion.has_moving_object and
            self.camera.motion.motion_confidence >= self.camera.motion.profile.min_motion_confidence.value
        ):
            self.camera.motion.last_motion_time = now

        # --- START RECORDING ---
        if self.camera.recording_state.should_record and not self.camera.recording_state.recording:
            self.camera.recording_state.recording = True
            self.camera.recording_state.recording_start_time = now
            self.camera.motion.active_objects_dict = deepcopy(self.camera.motion.classes_in_frame_dict)
            self.recorder.start_recording()

            log_event(
                message=f"recording start {tags_to_str(self.camera.motion.active_objects_dict)}",
                level="info",
                camera=self.camera
            )

        # --- CONTINUE RECORDING (merge object colors) ---
        if self.camera.recording_state.recording:
            for item, colors in self.camera.motion.classes_in_frame_dict.items():
                self.camera.motion.active_objects_dict[item].update(colors)

        # --- STOP RECORDING ---
        if self.camera.recording_state.recording and not self.camera.recording_state.should_continue:
            if now - self.camera.motion.last_motion_time > constants.POST_RECORD_DURATION:
                self.recorder.stop_recording()
                # Reset state
                self.recorder = self.recorder_factory.create(self.camera, self.stop_event, self.recordings)
                self.camera.recording_state.recording = False
                self.camera.motion.classes_in_frame_dict.clear()
                self.camera.motion.active_objects_dict.clear()
                self.camera.motion.motion_persistence = 0

    def _should_start_recording(
        self,
        motion_boxes: list[tuple[int, int, int, int]],
        now: float
    ) -> bool:
        """
        Determine whether recording should start.
        Preserves original logic:
        - motion_persistence >= min_motion_frames
        - YOLO sees moving objects
        - object area >= min_sum_box_area_pixels
        - motion_confidence >= START_CONF
        """

        object_area = sum((x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in motion_boxes)

        motion_is_persistent = (
            self.camera.motion.motion_persistence >= self.camera.motion.profile.min_motion_frames.value
        )

        object_motion = (
            motion_is_persistent and
            self.camera.motion.has_moving_object
        )

        object_area_ok = (
            object_area >= self.camera.motion.profile.min_sum_box_area_pixels
        )

        START_CONF = self.camera.motion.profile.min_motion_confidence.value

        return (
            object_motion and
            object_area_ok and
            self.camera.motion.motion_confidence >= START_CONF
        )

    def _should_continue_recording(self) -> bool:
        """
        Continue recording if:
        - confidence >= STOP_CONF (hysteresis), OR
        - motion persistence window is active.

        Post-record timing is handled in _update_recording_state.
        """

        START_CONF = self.camera.motion.profile.min_motion_confidence.value
        STOP_CONF  = max(0.10, START_CONF * 0.30)

        if not self.camera.recording_state.recording:
            return False

        # 1. Hysteresis
        if self.camera.motion.motion_confidence >= STOP_CONF:
            return True

        # 2. Persistence window
        if self.camera.motion.motion_persistence >= self.camera.motion.profile.min_motion_frames.value:
            return True

        return False


    def _render_debug_ui(
        self,
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
        if not self.camera.config.debug:
            return

        self.camera.debug_motion_image = draw_debug_panels(
            self.camera,
            self.model.names,
            self.frame_count,
            frame_bgr,
            yolo_result,
            self.status_text,
            self.objects_text,
            self.camera.recording_state.recording,
            self.camera.motion.motion_boxes_list,
            [], [], [], [], []  # placeholders for krs/kcs/dsrs/dscs/dars/dacs
        )

        if self.camera.recording_state.recording:
            timestamp_str = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S_%f")
            filename_str = os.path.join(self.camera.config.images_dir, f"{timestamp_str}.jpg")
            cv2.imwrite(filename_str, self.camera.debug_motion_image)


    def _select_frame(
        self,
        frame_bgr: NDArray[np.uint8],
        yolo_result: Results | None
    ) -> Tuple[NDArray[np.uint8], bool]:
        """
        Choose the final debug frame:
        - if debug panels exist → use them
        - else if YOLO results exist → use YOLO overlay
        - else → use raw frame
        """
        if self.camera.config.debug and self.camera.debug_motion_image is not None:
            return self.camera.debug_motion_image, True

        return self._apply_yolo_overlay(frame_bgr, yolo_result), False


    def _finalize_output(
        self,
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

        # Select final frame (debug mosaic > YOLO overlay > raw)
        final_frame, is_debug_frame = self._select_frame(frame_bgr, yolo_result)
        # Draw status text on the ORIGINAL frame

        #if not is_debug_frame:
        #    draw_status_text(
        #        final_frame,
        #        self.status_text,
        #        self.objects_text,
        #        self.camera.recording_state.recording,
        #    )
        # Update GUI-visible frame
        self.camera.latest_frame = final_frame

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

        #return yolo_result.plot(pil=False)
        return self.draw_yolo_boxes_fullres(
            frame_bgr,
            yolo_result,
            self.model.names
            )
    
    import cv2

    def draw_yolo_boxes_fullres(
        self,
        frame,
        result,
        class_names=None,
        min_conf=0.25,
        box_thickness=2,
        font_scale=0.5,
    ):
        """
        Draws YOLO boxes on a full-resolution frame.
        Assumes boxes have already been scaled to full resolution.

        frame: np.ndarray (H, W, 3) BGR
        result: Ultralytics Results object (with scaled boxes)
        class_names: model.names (list or dict)
        min_conf: minimum confidence to draw
        """

        if result is None or result.boxes is None or len(result.boxes) == 0:
            return frame

        H, W = frame.shape[:2]

        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        def yolo_color(cid: int):
            """
            Deterministic, distinct, visually balanced color for each class.
            Matches Ultralytics-style palette behavior.
            """
            # Hash the class ID into a stable 24-bit value
            h = (cid * 2654435761) & 0xFFFFFFFF

            # Extract RGB components
            r = (h >> 16) & 255
            g = (h >> 8) & 255
            b = h & 255

            # Boost brightness so colors are not too dark
            return (int(b * 0.7 + 75), int(g * 0.7 + 75), int(r * 0.7 + 75))

        for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, cls_ids):
            if conf < min_conf:
                continue

            # clamp
            x1 = int(max(0, min(W - 1, x1)))
            y1 = int(max(0, min(H - 1, y1)))
            x2 = int(max(0, min(W - 1, x2)))
            y2 = int(max(0, min(H - 1, y2)))

            if x2 <= x1 or y2 <= y1:
                continue

            color = yolo_color(cid)

            # draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

            # label text
            label = str(cid)
            if class_names is not None:
                if isinstance(class_names, dict):
                    label = class_names.get(cid, str(cid))
                else:
                    if 0 <= cid < len(class_names):
                        label = class_names[cid]

            label = f"{label} {conf:.2f}"

            # text size
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )

            # text background box
            tx1, ty1 = x1, max(0, y1 - th - baseline - 2)
            tx2, ty2 = x1 + tw + 4, y1

            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), color, -1)

            # text
            cv2.putText(
                frame,
                label,
                (tx1 + 2, ty2 - baseline - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return frame
