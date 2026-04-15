import dataclasses
from subprocess import Popen

@dataclasses.dataclass
class Camera:
    name: str
    url: str
    enabled: bool
    process: Popen = None
    hd: bool = False