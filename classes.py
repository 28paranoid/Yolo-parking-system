# classes.py
import os
import json
import numpy as np
from scipy.spatial import distance
from collections import defaultdict, deque
from shapely.geometry import Polygon


def load_manual_spots(path):
    """Load manually-marked parking spots (from mark_spots.py / parking_spots.json)
    into the same spot structure the pipeline expects."""
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        data = json.load(f)
    spots = []
    for item in data.get('spots', []):
        pts = item.get('points', [])
        if len(pts) < 3:
            continue
        poly = np.array(pts, dtype=np.int32)
        x_min, y_min = poly.min(axis=0)
        x_max, y_max = poly.max(axis=0)
        try:
            area = float(Polygon([tuple(p) for p in pts]).area)
        except Exception:
            area = 0.0
        if area <= 0:
            area = float((x_max - x_min) * (y_max - y_min))
        spots.append({
            'polygon': poly,
            'center': (int((x_min + x_max) // 2), int((y_min + y_max) // 2)),
            'area': area,
        })
    return spots


def merge_nearby_spots(spots, merge_distance):
    """Cluster spots whose centers are within `merge_distance` and return merged
    axis-aligned boxes. Shared by both the vehicle-based and custom-space calibration."""
    if not spots:
        return []
    merged, used = [], set()
    for i, s1 in enumerate(spots):
        if i in used:
            continue
        cluster = [s1]
        used.add(i)
        for j, s2 in enumerate(spots[i + 1:], i + 1):
            if j not in used and distance.euclidean(s1['center'], s2['center']) < merge_distance:
                cluster.append(s2)
                used.add(j)
        all_pts = np.vstack([s['polygon'] for s in cluster])
        x_min, y_min = all_pts.min(axis=0)
        x_max, y_max = all_pts.max(axis=0)
        merged.append({
            'polygon': np.array([[x_min, y_min], [x_max, y_min],
                                 [x_max, y_max], [x_min, y_max]], np.int32),
            'center': ((x_min + x_max) // 2, (y_min + y_max) // 2),
            'area': (x_max - x_min) * (y_max - y_min)
        })
    return merged


class SpaceDetectionCalibrator:
    """Builds a stable set of parking spots from a CUSTOM model's 'parking_space'
    detections aggregated over many sampled frames.

    Note: a spot that is occupied in a given frame may not be detected (the car
    covers it), so we aggregate across several scan points to catch each spot while
    it is visible at least once."""

    def __init__(self, merge_distance, min_area):
        self.merge_distance = merge_distance
        self.min_area = min_area
        self.space_boxes = []  # list of (x1, y1, x2, y2)

    def add_space(self, box):
        x1, y1, x2, y2 = [int(v) for v in box]
        if (x2 - x1) * (y2 - y1) >= self.min_area:
            self.space_boxes.append((x1, y1, x2, y2))

    def build_spots(self):
        spots = []
        for (x1, y1, x2, y2) in self.space_boxes:
            poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
            spots.append({
                'polygon': poly,
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'area': (x2 - x1) * (y2 - y1)
            })
        # Merge overlapping / duplicate detections of the same physical spot.
        return merge_nearby_spots(spots, self.merge_distance)


class ParkingSpotDetector:
    """Fallback calibrator: infers spots from where vehicles stay stationary.
    Used only when the custom parking-space model is unavailable."""

    def __init__(self, min_stationary_time, fps, merge_distance, min_area):
        self.min_stationary_frames = int(min_stationary_time * fps)
        self.merge_distance = merge_distance
        self.min_area = min_area
        self.vehicle_positions = defaultdict(list)

    def add_detection(self, frame_num, track_id, box):
        x1, y1, x2, y2 = box
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.vehicle_positions[track_id].append((frame_num, box, center))

    def calculate_spots(self):
        potential_spots = []
        for track_id, positions in self.vehicle_positions.items():
            if len(positions) < self.min_stationary_frames:
                continue
            centers = [p[2] for p in positions]
            movements = [distance.euclidean(centers[i], centers[i + 1]) for i in range(len(centers) - 1)]
            if np.mean(movements) < 10:
                boxes = [p[1] for p in positions[len(positions) // 3:]]
                avg_box = np.median(boxes, axis=0).astype(int)
                x1, y1, x2, y2 = avg_box
                padding = 15
                poly = np.array([[x1 - padding, y1 - padding], [x2 + padding, y1 - padding],
                                 [x2 + padding, y2 + padding], [x1 - padding, y2 + padding]], np.int32)
                area = (x2 - x1) * (y2 - y1)
                if area >= self.min_area:
                    potential_spots.append({'polygon': poly, 'center': ((x1 + x2) // 2, (y1 + y2) // 2), 'area': area})
        return merge_nearby_spots(potential_spots, self.merge_distance)

    def _merge_nearby_spots(self, spots):
        # Kept for backward compatibility.
        return merge_nearby_spots(spots, self.merge_distance)


class SmartVehicleTracker:
    def __init__(self, memory_frames):
        self.memory_frames = memory_frames
        self.lost_vehicles = {}
        # Store recent movements to calculate velocity (now includes centers)
        self.vehicle_history = defaultdict(lambda: deque(maxlen=10))
        self.track_to_spot = {}

    def update_vehicle(self, track_id, box, frame_num):
        center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        self.vehicle_history[track_id].append((frame_num, box, center))
        if track_id in self.lost_vehicles:
            del self.lost_vehicles[track_id]

    def is_stationary(self, track_id, max_movement=10):
        """Returns True if vehicle moved less than threshold in recent frames"""
        if track_id not in self.vehicle_history or len(self.vehicle_history[track_id]) < 5:
            return True  # Assume stationary if new

        centers = [h[2] for h in self.vehicle_history[track_id]]
        distances = [distance.euclidean(centers[i], centers[i + 1]) for i in range(len(centers) - 1)]
        return np.mean(distances) < max_movement

    def get_smoothed_box(self, track_id):
        if track_id not in self.vehicle_history:
            return None
        return np.mean([b for _, b, _ in self.vehicle_history[track_id]], axis=0).astype(int)

    def mark_lost(self, track_id, last_box, spot_idx, frame_num):
        self.lost_vehicles[track_id] = {
            'last_seen': frame_num,
            'last_box': last_box,
            'spot': spot_idx,
            'is_parked': True
        }
        self.track_to_spot[track_id] = spot_idx

    def get_lost_vehicles(self, current_frame):
        active = {}
        for tid, info in list(self.lost_vehicles.items()):
            limit = self.memory_frames * 2 if info.get('is_parked') else self.memory_frames
            if current_frame - info['last_seen'] <= limit:
                active[tid] = info
            else:
                del self.lost_vehicles[tid]
                if tid in self.track_to_spot:
                    del self.track_to_spot[tid]
        return active
