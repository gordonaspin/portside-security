from dataclasses import dataclass

@dataclass
class RecordingState:
    recording: bool = False
    recording_start_time: float = 0.0
    should_record: bool = False
    should_continue: bool = False

    # Used only by shadow filters, not recording logic
    white_ratio: float = 0.0
