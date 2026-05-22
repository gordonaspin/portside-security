"""Constants"""
from enum import Enum, auto
import numpy as np # For retrying connection after timeouts and errors

# =========================
# SETTINGS
# =========================
MAX_LOG_LINES = 1000

PRE_RECORD_DURATION = 4
POST_RECORD_DURATION = 1
RECORDING_FRAME_COUNT_MINIMUM = PRE_RECORD_DURATION + POST_RECORD_DURATION - 2

TS_FILE_RING_SECONDS = 120

PERIODIC_CHECK_INTERVAL = 300 # seconds
NIGHT_TIME_THRESHOLD = 100


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

