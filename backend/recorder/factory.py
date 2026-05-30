from logging import getLogger

from constants import PRE_RECORD_DURATION
from nvr.camera.camera import Camera
from recorder.recorders import OpenCVFrameRecorder, FfmpegFrameRecorder, FfmpegSegmentRecorder

logger = getLogger("pynvr.recorder")

class FrameRecorderFactory:
    @staticmethod
    def create(camera: Camera, factory_name: str):
        if factory_name == "OpenCVFrameRecorderFactory":
            return OpenCVFrameRecorderFactory
        elif factory_name == "FfmpegFrameRecorderFactory":
            return FfmpegFrameRecorderFactory
        elif factory_name == "FfmpegSegmentRecorderFactory":
            return FfmpegSegmentRecorderFactory
        else:
            logger.warning(f"Unknown recorder factory '{factory_name}' for camera {camera.config.name}. Defaulting to FfmpegFrameRecorderFactory.")
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
