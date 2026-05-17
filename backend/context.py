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
    motion_threshold: float
    confidence_threshold: float
    resolution: list[int, int]
    model: str
    lpr_model: str
    classes: list[str]
    debug: bool
    debug_files: bool = False
