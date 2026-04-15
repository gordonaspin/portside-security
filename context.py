""" Context manager for the application. """
import dataclasses
from camera import Camera

@dataclasses.dataclass
class Context:
    """Context manager for the application."""
    directory: str
    username: str
    password: str
    logging_config: str
    motion_threshold: list[int, int]
    confidence_threshold: float
    motion_detect_frame_count: int
    resolution: list[int, int]
    cameras: dict[Camera]
    model: str
    classes: list[str]
    debug: bool
