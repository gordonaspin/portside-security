from collections.abc import Callable
from logging import getLogger
from threading import Event

from constants import PRE_RECORD_DURATION
from nvr.camera.camera import Camera
from recorder.recorders import OpenCVFrameRecorder, AVFFmpegFrameRecorder, FFmpegFrameRecorder, FFmpegSegmentRecorder

logger = getLogger("pynvr.recorder")

class OpenCVFrameRecorderFactory:
    @staticmethod
    def create(camera: Camera, stop_event: Event, recordings: Callable, pre_record_duration=PRE_RECORD_DURATION):
        return OpenCVFrameRecorder(camera=camera, stop_event=stop_event, recordings=recordings, pre_record_duration=pre_record_duration)

class AVFFmpegFrameRecorderFactory:
    @staticmethod
    def create(camera: Camera, stop_event: Event, recordings: Callable, pre_record_duration=PRE_RECORD_DURATION):
        return AVFFmpegFrameRecorder(camera=camera, stop_event=stop_event, recordings=recordings, pre_record_duration=pre_record_duration)

class FFmpegFrameRecorderFactory:
    @staticmethod
    def create(camera: Camera, stop_event: Event, recordings: Callable, pre_record_duration=PRE_RECORD_DURATION):
        return FFmpegFrameRecorder(camera=camera, stop_event=stop_event, recordings=recordings, pre_record_duration=pre_record_duration)

class FFmpegSegmentRecorderFactory:
    @staticmethod
    def create(camera: Camera, stop_event: Event, recordings: Callable, pre_record_duration=PRE_RECORD_DURATION):
        return FFmpegSegmentRecorder(camera=camera, stop_event=stop_event, recordings=recordings, pre_record_duration=pre_record_duration)

class FrameRecorderFactory:
    factory_mapping = {
        "OpenCVFrame": OpenCVFrameRecorderFactory,
        "AVFFmpegFrame": AVFFmpegFrameRecorderFactory,
        "FFmpegFrame": FFmpegFrameRecorderFactory,
        "FFmpegSegment": FFmpegSegmentRecorderFactory,
    }
    @staticmethod
    def create(camera: Camera, factory_name: str):
        if factory_name not in FrameRecorderFactory.factory_mapping:
            logger.warning(f"Unknown recorder factory '{factory_name}', defaulting to AVFFmpegFrameRecorderFactory")
        else:
            logger.info(f"Using {factory_name} recorder factory for camera {camera.config.name}")
        return FrameRecorderFactory.factory_mapping.get(factory_name, AVFFmpegFrameRecorderFactory)

