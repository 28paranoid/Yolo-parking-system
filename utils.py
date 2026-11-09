# utils.py
import cv2
import csv
import os
import datetime
import numpy as np


def initialize_csv(filepath):
    """Creates the CSV file with headers."""
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Event', 'Spot_ID', 'Vehicle_ID', 'Video_Timestamp', 'Real_Time', 'Image_Path'])


def log_event(csv_path, event_type, spot_id, vehicle_id, video_time, image_name):
    """Writes an event (Entry/Exit) to the CSV."""
    real_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([event_type, spot_id, vehicle_id, video_time, real_time, image_name])


def nms_merge(dets, iou_thresh=0.6):
    """Non-max suppression to merge duplicate boxes coming from the two models.
    dets: list of [x1, y1, x2, y2, conf, cls]. Returns the kept detections."""
    if not dets:
        return []
    arr = np.array(dets, dtype=np.float32)
    x1, y1, x2, y2, scores = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]
    return [dets[i] for i in keep]


def crop_vehicle_safe(frame, box):
    """Safely crop vehicle from bounding box with boundary checks."""
    h, w, _ = frame.shape
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def save_event_image(frame, folder, event_type, spot_id, vehicle_id, cropped_vehicle=None):
    """Saves a snapshot when an event occurs. Uses cropped vehicle if provided."""
    filename = f"{event_type}_Spot{spot_id}_ID{vehicle_id}.jpg"
    filepath = os.path.join(folder, filename)

    if cropped_vehicle is not None:
        img_to_save = cropped_vehicle.copy()
    else:
        img_to_save = frame.copy()

    cv2.putText(img_to_save, f"{event_type.upper()} - Spot {spot_id}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imwrite(filepath, img_to_save)
    return filename
