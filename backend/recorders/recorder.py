from constants import PRE_RECORD_DURATION
from collections import defaultdict
from datetime import datetime

from camera.camera import Camera

class Recorder:
    def __init__(self, camera: Camera, pre_record_duration=PRE_RECORD_DURATION):
        self.camera = camera
        self.pre_record_duration = pre_record_duration
        self.filename = None

    def create_metadata(self, tags: defaultdict, media_filename, metadata_filename, start_time, end_time):
        # Convert to a standard dict and sets to lists
        serializable_tags = {k: list(v) for k, v in tags.items()}
        profile = self.camera.profile_to_dict()
        stats = self.camera.auto_tuner.summarize()
        recs = self.camera.auto_tuner.recommend_adjustments()

        json_data = {
            "camera": self.camera.name,
            "tags": serializable_tags,
            "media_filename": media_filename,
            "start_time": start_time,
            "end_time": end_time,
            "start_fmt": datetime.fromtimestamp(start_time).strftime("%Y/%m/%d %H:%M:%S"),
            "end_fmt": datetime.fromtimestamp(end_time).strftime("%Y/%m/%d/ %H:%M:%S"),
            "metadata_filename": metadata_filename,
            "profile": profile,
            "tuner_stats": stats,
            "recommendations": recs,
        }
        return json_data

    def _tags_to_str(self, tags: defaultdict[set]):
        if not tags:
            return ""

        parts = []
        for obj, colors in tags.items():
            object_str = obj
            color_str = ":".join(colors)
            parts.append(f"{object_str}({color_str})")
        return ",".join(parts)