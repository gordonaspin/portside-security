"""Constants"""
from enum import Enum, auto

# =========================
# SETTINGS
# =========================
MAX_LOG_LINES = 1000
TS_FILE_RING_SECONDS = 120


class ExitCode(Enum):
    """
    ExitCode definitions
    """
    EXIT_NORMAL: int = 0
    EXIT_FAILED_ALREADY_RUNNING: int = auto()
    EXIT_FAILED_CLICK_EXCEPTION: int = auto()
    EXIT_FAILED_CLICK_USAGE: int = auto()
    EXIT_FAILED_NOT_A_DIRECTORY: int = auto()
    EXIT_FAILED_MISSING_COMMAND: int = auto()


class StreamingState(Enum):
    """
    StreamingState definitions
    """
    STREAMING_INIT: int = 0
    STREAMING_NORMAL: int = auto()
    STREAMING_STOPPED: int = auto()
    STREAMING_FAILED: int = auto()