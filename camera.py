import dataclasses
from subprocess import Popen

@dataclasses.dataclass
class Camera:
    name: str
    url: str
    enabled: bool
    recordings_dir: str
    segments_dir: str
    images_dir: str
    process: Popen = None
    hd: bool = False
    last_event_time: float = 0.0
    active_objects: dict = dataclasses.field(default_factory=dict)
    active_events: dict = dataclasses.field(default_factory=dict)