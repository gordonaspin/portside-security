import numpy as np

from .kalman_filter import KalmanFilter
from .matching import iou_distance, linear_assignment
from .utils import tlbr_to_tlwh, tlwh_to_xyah


class Track:
    def __init__(self, tlbr, score, cls, track_id, kf):
        self.tlbr = np.array(tlbr, dtype=np.float32)
        self.score = float(score)
        self.cls = int(cls)
        self.track_id = track_id

        self.kf = kf
        self.mean, self.cov = kf.initiate(tlwh_to_xyah(tlbr_to_tlwh(self.tlbr)))

        self.age = 1
        self.active = True

    def predict(self):
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        self.age += 1

    def update(self, det):
        self.tlbr = det.tlbr
        self.score = det.score
        self.cls = det.cls
        self.mean, self.cov = self.kf.update(
            self.mean, self.cov, tlwh_to_xyah(tlbr_to_tlwh(self.tlbr))
        )
        self.active = True


class Detection:
    def __init__(self, tlbr, score, cls):
        self.tlbr = np.array(tlbr, dtype=np.float32)
        self.score = float(score)
        self.cls = int(cls)


class BYTETracker:
    def __init__(self, track_thresh=0.5, match_thresh=0.8, track_buffer=30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer

        self.kf = KalmanFilter()
        self.tracks = []
        self.next_id = 1

    def update(self, dets, img_info, img_size):
        """
        dets: Nx6 array [x1,y1,x2,y2,score,cls]
        """
        # Convert detections
        detections = [
            Detection(d[:4], d[4], d[5])
            for d in dets
            if d[4] >= self.track_thresh
        ]

        # Predict existing tracks
        for t in self.tracks:
            t.predict()

        # Match tracks to detections
        cost = iou_distance(self.tracks, detections)
        matches, u_tracks, u_dets = linear_assignment(cost, 1 - self.match_thresh)

        # Update matched tracks
        for ti, di in matches:
            self.tracks[ti].update(detections[di])

        # Create new tracks
        for di in u_dets:
            det = detections[di]
            self.tracks.append(
                Track(det.tlbr, det.score, det.cls, self.next_id, self.kf)
            )
            self.next_id += 1

        # Remove stale tracks
        self.tracks = [
            t for t in self.tracks if t.age <= self.track_buffer
        ]

        return self.tracks
