import time
from datetime import timedelta
from logging import getLogger
from pathlib import Path
from threading import Event, Thread, current_thread

from utils.thread_safe import ThreadSafeSet
from utils.utils import make_readable_ts

logger = getLogger("pynvr.cleaner")

class CleanerConfig:
    def __init__(self, folder: str, filespec: str, age_seconds: int, period: int):
        self.folder = folder
        self.filespec = filespec
        self.age_seconds = age_seconds
        self.period_seconds = period
        self.last_cleanup_time = time.time()

class FileCleaner():
    do_not_delete_set: ThreadSafeSet = ThreadSafeSet()
    min_sleep_seconds = 60
    cleaner_config: list[CleanerConfig] = []
    stop_event: Event = None
    thread: Thread = None

    @staticmethod
    def add(folder: str, filespec: str, age: timedelta, period: timedelta):
        config = CleanerConfig(folder, filespec, age.total_seconds(), period.total_seconds())
        FileCleaner.cleaner_config.append(config)
        FileCleaner.min_sleep_seconds = min(FileCleaner.min_sleep_seconds, period.total_seconds())
        if FileCleaner.thread is None:
            logger.debug("Starting file cleaner thread")
            FileCleaner.start()

    @staticmethod
    def start():
        FileCleaner.thread = Thread(target=FileCleaner._cleanup, daemon=True)
        FileCleaner.thread.start()

    @staticmethod
    def _cleanup():
        """
        Thread that periodically deletes old files
        """
        current_thread().name = "cleanup_segments"

        while True:
            if FileCleaner.stop_event is not None and FileCleaner.stop_event.is_set():
                break

            now = time.time()
            for config in FileCleaner.cleaner_config:
                if now - config.last_cleanup_time >= config.period_seconds:
                    path = Path(config.folder)
                    cutoff = now - config.age_seconds
                    for file in path.rglob(config.filespec):
                        if file not in FileCleaner.do_not_delete_set and file.is_file():
                            # .stat().st_mtime gets the modification time
                            stat_entry = file.stat()
                            if stat_entry.st_mtime < cutoff:
                                file.unlink()
                                logger.debug(f"FileCleaner deleted: {file} dated {make_readable_ts(stat_entry.st_mtime)}")
                    config.last_cleanup_time = now
            time.sleep(FileCleaner.min_sleep_seconds)