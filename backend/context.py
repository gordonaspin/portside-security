""" Context manager for the application. """
import dataclasses

@dataclasses.dataclass
class Context:
    """Context manager for the application."""
    directory: str
    log_directory: str
    username: str
    password: str
    gui_username: str
    gui_password: str
    camera_config: dict
    bind_address: str
    resolution: list[int, int]
    yolo_config: dict
    debug: bool
    debug_files: bool = False
