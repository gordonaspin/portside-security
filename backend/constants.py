"""Constants"""
from enum import Enum, auto

# =========================
# SETTINGS
# =========================
MAX_LOG_LINES = 1000

PRE_RECORD_DURATION = 3
POST_RECORD_DURATION = 3

TS_FILE_RING_SECONDS = 120
DELAY_FIRST_RECORDING_SECONDS = 15

PERIODIC_CHECK_INTERVAL = 60 # seconds
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

