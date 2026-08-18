from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import TRACK_DISTANCE_PX, TRACK_MAX_MISSED_FRAMES


@dataclass
class Track:
    track_id: int
    center_x: float
    center_y: float
    area: float

    person_id: Optional[int] = None
    person_name: str = "Unknown"
    confidence: float = 0.0

    embedding: Optional[np.ndarray] = None

    last_seen: int = 0
    missed: int = 0

    # Unknown-face learning
    unknown_confirmations: int = 0
    registered_unknown: bool = False
    unknown_id: Optional[int] = None
    unknown_samples: int = 0
    last_unknown_capture: int = 0

    # Lip / speaker
    lip_open: float = 0.0
    speaking: bool = False
    speech_frames: int = 0
    silent_frames: int = 0


class CentroidTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    @staticmethod
    def _distance(a, b):
        return ((a.center_x - b[0]) ** 2 + (a.center_y - b[1]) ** 2) ** 0.5

    def update(self, detections, frame_number):
        """
        detections: list of dicts containing bbox=(x0,y0,x1,y1)
        Returns list of (detection, Track).
        """
        if not detections:
            for track in self.tracks.values():
                track.missed += 1
            self._remove_old()
            return []

        unmatched_tracks = set(self.tracks.keys())
        assignments = []

        # Greedy nearest-neighbour matching.
        for det in detections:
            x0, y0, x1, y1 = det["bbox"]
            center = ((x0 + x1) / 2, (y0 + y1) / 2)
            area = max(1, (x1 - x0) * (y1 - y0))

            best_id = None
            best_distance = float("inf")

            for track_id in list(unmatched_tracks):
                track = self.tracks[track_id]
                d = self._distance(track, center)
                area_ratio = abs(area - track.area) / max(track.area, 1)

                if d <= TRACK_DISTANCE_PX and area_ratio <= 1.0 and d < best_distance:
                    best_distance = d
                    best_id = track_id

            if best_id is None:
                track = Track(
                    track_id=self.next_id,
                    center_x=center[0],
                    center_y=center[1],
                    area=area,
                    last_seen=frame_number,
                )
                self.tracks[self.next_id] = track
                self.next_id += 1
            else:
                track = self.tracks[best_id]
                unmatched_tracks.remove(best_id)
                track.center_x = center[0]
                track.center_y = center[1]
                track.area = area
                track.last_seen = frame_number
                track.missed = 0

            assignments.append((det, track))

        for track_id in unmatched_tracks:
            self.tracks[track_id].missed += 1

        self._remove_old()
        return assignments

    def _remove_old(self):
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id].missed > TRACK_MAX_MISSED_FRAMES:
                del self.tracks[track_id]
