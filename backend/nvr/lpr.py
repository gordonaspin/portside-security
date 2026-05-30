import cv2
import numpy as np
import re
import yaml
from logging import getLogger
from typing import Tuple, Optional

from paddleocr import PaddleOCR
from ultralytics import YOLO
from shapely.geometry import box as shapely_box

from nvr.camera.camera import Camera

logger = getLogger("pynvr")

class LicensePlateRecognition:
    def __init__(self, model: str, license_patterns_file: str = "backend/model/license_plate_patterns.yaml"):
        self.license_plate_patterns, self.replace_pattern = self.load_license_plate_patterns(license_patterns_file)
        self.model = YOLO(model)
        self.model.fuse()
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")

    def get_ocr_result(self, roi: np.ndarray) -> list:
        # PaddleOCR >= 2.7: predict() is the recommended API
        # Returns a list of dicts: [{'text': str, 'confidence': float, 'bbox': [...]}, ...]
        return self.ocr.predict(roi)

    def load_license_plate_patterns(self, file_path: str) -> Tuple[dict, str]:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
            return data.get("license_plate_patterns", {}), r"[^A-Za-z0-9]"

    def detect_state_and_plate(self, plate_text: str) -> Tuple[Optional[str], Optional[str]]:
        for state, pattern in self.license_plate_patterns.items():
            if re.match(pattern, plate_text):
                return state, plate_text
        return None, None

    def normalize_ocr(self, text: str) -> str:
        # Normalize common OCR confusions
        text = text.upper()
        text = text.replace("O", "0").replace("I", "1").replace("S", "5").replace("B", "8")
        return text

    def format_plate(self, text: str) -> str:
        text = re.sub(r"[-\s]+", "", text)
        text = text.upper()
        text = re.sub(r"[^A-Z0-9]", "", text)
        return text


class VideoProcessor:
    def __init__(self, license_plate_recognition: LicensePlateRecognition):
        self.lpr = license_plate_recognition

    def process_frame(self, camera: Camera, frame: np.ndarray):
        self.lpr.model.fp16 = True
        # Use a more realistic confidence threshold
        result = self.lpr.model.predict(frame, conf=0.25, verbose=False)[0]
        frame, detected_texts = self.validate_and_annotate(camera, frame, result)
        result.plot(pil=False)
        return frame, detected_texts

    def preprocess_frame(self, camera: Camera, frame: np.ndarray) -> np.ndarray:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=camera.lpr.gray_buf)
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0, dst=camera.lpr.gray_buf)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized_frame = clahe.apply(gray_frame, dst=camera.lpr.equalized_buf)
        preprocessed_frame = cv2.cvtColor(equalized_frame, cv2.COLOR_GRAY2BGR, dst=camera.lpr.preprocessed_buf)
        return preprocessed_frame

    def validate_and_annotate(self, camera: Camera, frame: np.ndarray, result) -> Tuple[np.ndarray, list]:
        detected_texts = []
        processed_boxes = []
        h, w = frame.shape[:2]

        if len(result.boxes) == 0:
            logger.debug("LPR YOLO: no result boxes")

        logger.debug(f"LPR YOLO: {len(result.boxes)} result boxes")
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            current_box = shapely_box(x1, y1, x2, y2)

            if any(self.calculate_iou(current_box, b) > 0.5 for b in processed_boxes):
                continue

            # Optional: aspect ratio filter (plates are wide)
            bw, bh = x2 - x1, y2 - y1
            if bh <= 0 or bw <= 0:
                logger.debug(f"LPR YOLO: discarding box bh or bw <= 0")
                continue
            ratio = bw / bh
            if ratio < 2.0 or ratio > 6.0:
                logger.debug(f"LPR YOLO: discarding aspect ratio {ratio}")
                continue

            # Add padding around ROI for better OCR
            pad = 10
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(w, x2 + pad)
            y2p = min(h, y2 + pad)
            if x2p <= x1p or y2p <= y1p:
                logger.debug(f"LPR YOLO: discarding after padding")
                continue

            roi = frame[y1p:y2p, x1p:x2p]
            preprocessed_roi = self.preprocess_frame(camera, roi)

            state, detected_text = self.extract_text_from_roi(preprocessed_roi)

            if detected_text and state:
                detected_texts.append((detected_text, state, (x1p, y1p, x2p, y2p)))

            processed_boxes.append(current_box)

        frame = self.annotate_frame(frame, detected_texts)
        return frame, detected_texts

    def calculate_iou(self, box1, box2) -> float:
        intersection = box1.intersection(box2).area
        union = box1.union(box2).area
        return intersection / union if union > 0 else 0

    def extract_text_from_roi(self, roi: np.ndarray) -> Tuple[Optional[str], Optional[str]]:
        ocr_result = self.lpr.get_ocr_result(roi)
        if not ocr_result:
            return None, None

        for item in ocr_result:
            # PaddleOCR predict() item: {'text': str, 'confidence': float, 'bbox': [...]}
            text = item.get("text", "")
            confidence = float(item.get("confidence", 0.0))

            if confidence < 0.5:
                continue

            formatted = self.lpr.format_plate(text)
            formatted = self.lpr.normalize_ocr(formatted)

            if not formatted:
                continue

            state, plate = self.lpr.detect_state_and_plate(formatted)
            if plate:
                return state, plate

        return None, None

    def annotate_frame(self, frame: np.ndarray, detected_texts: list) -> np.ndarray:
        for detected_text, state, (x1, y1, x2, y2) in detected_texts:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                frame,
                f"{detected_text} ({state})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
        return frame
