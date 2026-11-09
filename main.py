# main.py (Hybrid: pretrained COCO vehicles + custom parking-space model)
import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from boxmot import create_tracker
from shapely.geometry import Polygon, Point, box as shapely_box
import gc

# Custom modules
import config
import utils
from classes import (
    SmartVehicleTracker,
    load_manual_spots,
)

# --- SETUP ---
config.setup_directories()
utils.initialize_csv(config.CSV_FILE)

# ============================================================
# MODELS
# ============================================================
print(f"Loading models on device: {config.DEVICE_INDEX} ...")

# Model A: pretrained COCO -> vehicles
vehicle_model = YOLO(config.PRETRAINED_MODEL_PATH).to(config.YOLO_DEVICE)

# Model B: your custom-trained model -> parking spaces (+ optional car class)
space_model = None
if os.path.exists(config.CUSTOM_MODEL_PATH):
    space_model = YOLO(config.CUSTOM_MODEL_PATH).to(config.YOLO_DEVICE)
    print(f"Custom model loaded: {config.CUSTOM_MODEL_PATH} | classes: {space_model.names}")
else:
    print(f"⚠️   Custom model '{config.CUSTOM_MODEL_PATH}' not found.")
    print("    Falling back to stationary-vehicle calibration and pretrained vehicles only.")
    print("    (Train best.pt on your Label-Studio dataset to enable the full hybrid pipeline.)")


def initialize_tracker():
    return create_tracker(
        tracker_type=config.TRACKER_TYPE,
        tracker_config=None,
        reid_weights=config.REID_WEIGHTS,
        device=config.DEVICE_INDEX,
        half=True if torch.cuda.is_available() else False
    )


# ============================================================
# HYBRID DETECTION HELPERS
# ============================================================
def detect_pretrained_vehicles(frame):
    """Vehicles from the pretrained COCO model -> [x1,y1,x2,y2,conf,cls]."""
    dets = []
    results = vehicle_model(frame, conf=config.CONFIDENCE_THRESHOLD, verbose=False)
    for box in results[0].boxes:
        name = vehicle_model.names[int(box.cls[0])]
        if name in config.VEHICLE_CLASSES:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            dets.append([float(x1), float(y1), float(x2), float(y2), float(box.conf[0]), 0])
    return dets


def run_custom_model(frame):
    """Single pass of the custom model. Returns (vehicle_dets, space_boxes)."""
    veh, spaces = [], []
    if space_model is None:
        return veh, spaces
    conf_floor = min(config.SPACE_CONFIDENCE, config.CONFIDENCE_THRESHOLD)
    results = space_model(frame, conf=conf_floor, verbose=False)
    for box in results[0].boxes:
        name = space_model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        if name == config.SPACE_CLASS_NAME and conf >= config.SPACE_CONFIDENCE:
            spaces.append((x1, y1, x2, y2))
        elif name in config.CUSTOM_CAR_CLASSES and conf >= config.CONFIDENCE_THRESHOLD:
            veh.append([float(x1), float(y1), float(x2), float(y2), conf, 0])
    return veh, spaces


def get_fused_vehicle_detections(frame):
    """Hybrid vehicle detections: pretrained COCO (+ custom car class), merged via NMS."""
    dets = detect_pretrained_vehicles(frame)
    if config.FUSE_CUSTOM_VEHICLES and space_model is not None:
        cust_veh, _ = run_custom_model(frame)
        dets.extend(cust_veh)
    return utils.nms_merge(dets, config.NMS_IOU)


# --- VIDEO SETUP ---
cap = cv2.VideoCapture(config.VIDEO_FILENAME)
if not cap.isOpened():
    print(f"❌ Error: Could not open video '{config.VIDEO_FILENAME}'.")
    raise SystemExit(1)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

PROCESS_EVERY_N_FRAMES = max(1, int(fps / 5))  # ~5 FPS processing

print(f"\nVideo Configuration:")
print(f"   Original FPS: {fps}")
print(f"   Processing Rate: ~5 FPS (skipping every {PROCESS_EVERY_N_FRAMES} frames)")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(config.OUTPUT_FILENAME, fourcc, fps, (width, height))  # source FPS -> smooth playback

# ============================================================
# PARKING SPOTS  -> loaded from MANUAL marking (parking_spots.json)
# ============================================================
parking_spots = load_manual_spots(config.MANUAL_SPOTS_FILE)
if not parking_spots:
    print(f"\n No manually-marked spots found in '{config.MANUAL_SPOTS_FILE}'.")
    print("    Run 'python mark_spots.py' first to draw your parking spots, then run main.py again.")
    cap.release()
    out_writer.release()
    raise SystemExit(1)
print(f"\n Loaded {len(parking_spots)} manually-marked parking spots.")

# Validate that spot coordinates match the current video resolution
import json as _json
with open(config.MANUAL_SPOTS_FILE) as _f:
    _spots_meta = _json.load(_f)
_expected = _spots_meta.get('frame_size', [])
if _expected and (_expected[0] != width or _expected[1] != height):
    print(f"⚠️  Warning: parking_spots.json was marked on {_expected[0]}×{_expected[1]} "
          f"but video is {width}×{height}. Spot polygons may be misaligned!")

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
gc.collect()

# Precompute Shapely polygons (static — no need to rebuild every frame)
for spot_data in parking_spots:
    spot_data['shapely_poly'] = Polygon(spot_data['polygon'])

# ============================================================
# MAIN MONITORING
# ============================================================
spot_status = {i: {
    'occupied': False,
    'vehicle_id': None,
    'entry_time': None,
    'confidence_counter': 0,
    'miss_counter': 0,
    'last_valid_crop': None
} for i in range(len(parking_spots))}

tracker_memory = SmartVehicleTracker(config.MEMORY_FRAMES)
main_tracker = initialize_tracker()

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_count = 0
processed_count = 0

# Confirmations are counted once per PROCESSED frame (~5 FPS), so scale to that
# rate rather than the raw video FPS (the old `2.5 * fps` needed ~15s of
# unbroken overlap before a spot ever flipped to occupied).
PROCESS_FPS = fps / PROCESS_EVERY_N_FRAMES if PROCESS_EVERY_N_FRAMES else fps
ENTRY_CONFIRMATION = max(1, int(round(config.ENTRY_CONFIRM_SECONDS * PROCESS_FPS)))
EXIT_CONFIRMATION = max(1, int(round(config.EXIT_CONFIRM_SECONDS * PROCESS_FPS)))
print(f"   Processed FPS: ~{PROCESS_FPS:.1f} | ENTRY after {ENTRY_CONFIRMATION} hits | EXIT after {EXIT_CONFIRMATION} misses")

print(f"\n  STARTING LIVE MONITORING...")
print("Press 'q' to stop." if config.SHOW_PREVIEW else "Preview off - rendering video only...")

last_vehicle_boxes = []   # boxes from the most recent processed frame (redrawn on skipped frames)
time_str = "00:00"

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        current_time = frame_count / fps
        time_str = f"{int(current_time // 60):02d}:{int(current_time % 60):02d}"

        # ======== HEAVY WORK: detection + tracking + logic, only every Nth frame ========
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            processed_count += 1

            # 1. Hybrid detection
            detections = get_fused_vehicle_detections(frame)

            # 2. Tracking
            active_vehicles = {}
            if detections:
                tracks = main_tracker.update(np.array(detections, dtype=np.float32), frame)
                for track in tracks:
                    track_id = int(track[4])
                    box = track[0:4].astype(int)
                    tracker_memory.update_vehicle(track_id, box, frame_count)
                    smoothed = tracker_memory.get_smoothed_box(track_id)
                    if smoothed is not None:
                        active_vehicles[track_id] = {
                            'box': smoothed,
                            'poly': shapely_box(smoothed[0], smoothed[1], smoothed[2], smoothed[3]),
                            'center': ((smoothed[0] + smoothed[2]) // 2, (smoothed[1] + smoothed[3]) // 2),
                            'ground': (int((smoothed[0] + smoothed[2]) // 2), int(smoothed[3]))
                        }

            lost_vehicles = tracker_memory.get_lost_vehicles(frame_count)

            # 3. IOU-based parking logic (updates spot_status only; drawing happens below)
            for idx, spot_data in enumerate(parking_spots):
                spot_poly = spot_data['shapely_poly']
                spot_area = spot_data['area']

                # Hysteresis: a NEW spot needs IOU_THRESHOLD to start; one already in
                # progress (or occupied) only needs the lower IOU_KEEP_THRESHOLD to keep
                # counting -> a marginal/flickering overlap no longer stalls the entry.
                in_progress = spot_status[idx]['occupied'] or spot_status[idx]['confidence_counter'] > 0
                iou_gate = config.IOU_KEEP_THRESHOLD if in_progress else config.IOU_THRESHOLD

                best_vehicle = None
                max_ov = 0
                for tid, veh in active_vehicles.items():
                    # Overlap coefficient (size-independent): fraction of the smaller box.
                    inter = spot_poly.intersection(veh['poly']).area
                    denom = min(spot_area, veh['poly'].area)
                    ov = inter / denom if denom > 0 else 0.0
                    if ov <= iou_gate:
                        continue
                    # Reject passing cars in the lane. A real parked car satisfies at
                    # least one of: center inside spot, ground point inside spot, or it
                    # covers a big fraction of the spot. A lane car satisfies none.
                    cover = inter / spot_area if spot_area > 0 else 0.0
                    if config.REQUIRE_CENTER_IN_SPOT and not (
                        spot_poly.contains(Point(veh['center'])) or
                        spot_poly.contains(Point(veh['ground'])) or
                        cover >= config.SPOT_COVER_MIN):
                        continue
                    # Reject brand-new/fleeting tracks and moving cars.
                    if tracker_memory.age(tid) < config.MIN_TRACK_AGE:
                        continue
                    if not tracker_memory.is_stationary(tid, config.MAX_MOVEMENT_FOR_PARKED):
                        continue
                    if ov > max_ov:
                        max_ov = ov
                        best_vehicle = {'id': tid, **veh}

                if not best_vehicle:
                    for tid, info in lost_vehicles.items():
                        if info['spot'] == idx:
                            lb = info['last_box']
                            lost_poly = shapely_box(lb[0], lb[1], lb[2], lb[3])
                            inter = spot_poly.intersection(lost_poly).area
                            denom = min(spot_area, lost_poly.area)
                            iou = inter / denom if denom > 0 else 0.0
                            if iou > iou_gate:
                                best_vehicle = {'id': tid, 'box': lb, 'poly': lost_poly}
                                break

                if best_vehicle:
                    if spot_status[idx]['last_valid_crop'] is None:
                        crop = utils.crop_vehicle_safe(frame, best_vehicle['box'])
                        if crop is not None:
                            spot_status[idx]['last_valid_crop'] = crop

                    if not spot_status[idx]['occupied']:
                        spot_status[idx]['miss_counter'] = 0
                        spot_status[idx]['confidence_counter'] += 1
                        if spot_status[idx]['confidence_counter'] >= ENTRY_CONFIRMATION:
                            spot_status[idx].update({
                                'occupied': True,
                                'vehicle_id': best_vehicle['id'],
                                'entry_time': time_str
                            })
                            print(f"[ENTRY] Spot {idx + 1} | ID: {best_vehicle['id']} | Time: {time_str}")
                            save_img = spot_status[idx]['last_valid_crop'] if spot_status[idx]['last_valid_crop'] is not None else frame
                            img_name = utils.save_event_image(frame, config.CAPTURES_DIR, "ENTRY", idx + 1, best_vehicle['id'], save_img)
                            utils.log_event(config.CSV_FILE, "ENTRY", idx + 1, best_vehicle['id'], time_str, img_name)
                    else:
                        spot_status[idx]['confidence_counter'] = 0
                else:
                    if spot_status[idx]['occupied']:
                        spot_status[idx]['confidence_counter'] -= 1
                        if spot_status[idx]['confidence_counter'] <= -EXIT_CONFIRMATION:
                            vid = spot_status[idx]['vehicle_id']
                            print(f"[EXIT]  Spot {idx + 1} | ID: {vid} | Time: {time_str}")
                            save_img = spot_status[idx]['last_valid_crop'] if spot_status[idx]['last_valid_crop'] is not None else frame
                            img_name = utils.save_event_image(frame, config.CAPTURES_DIR, "EXIT", idx + 1, vid, save_img)
                            utils.log_event(config.CSV_FILE, "EXIT", idx + 1, vid, time_str, img_name)
                            spot_status[idx].update({
                                'occupied': False, 'vehicle_id': None, 'entry_time': None,
                                'confidence_counter': 0, 'miss_counter': 0, 'last_valid_crop': None
                            })
                    else:
                        # Empty spot: tolerate a few missed/flickering frames (grace),
                        # then reset so a car that merely passed through doesn't slowly
                        # accumulate into a false entry.
                        spot_status[idx]['miss_counter'] += 1
                        if spot_status[idx]['miss_counter'] > config.ENTRY_MISS_GRACE:
                            spot_status[idx]['confidence_counter'] = 0

            # cache vehicle boxes so skipped frames can redraw them
            last_vehicle_boxes = [tuple(int(c) for c in v['box']) for v in active_vehicles.values()]

        # ======== LIGHT WORK: draw + show + save on EVERY frame (smooth video) ========
        # Vehicles -> GREEN
        for (x1, y1, x2, y2) in last_vehicle_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), config.CAR_COLOR, 2)

        # Parking spots -> YELLOW when empty, RED when occupied (+ labels)
        for idx, spot_data in enumerate(parking_spots):
            spot_color = config.SPOT_OCCUPIED_COLOR if spot_status[idx]['occupied'] else config.SPOT_FREE_COLOR
            cv2.polylines(frame, [spot_data['polygon']], True, spot_color, 2)
            if spot_status[idx]['occupied']:
                cv2.putText(frame, f"ID:{spot_status[idx]['vehicle_id']}",
                            (spot_data['center'][0] - 20, spot_data['center'][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                cv2.putText(frame, f"P{idx + 1}", (spot_data['polygon'][0][0], spot_data['polygon'][0][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, spot_color, 2)

        # HUD
        occupied_count = sum(1 for status in spot_status.values() if status['occupied'])
        cv2.rectangle(frame, (10, 10), (350, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Occupied: {occupied_count}/{len(parking_spots)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {time_str}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        out_writer.write(frame)

        if config.SHOW_PREVIEW:
            # Scale to fit the window while preserving aspect ratio (never upscale),
            # so portrait videos aren't stretched into a landscape box.
            scale = min(1.0, config.PREVIEW_MAX_SIZE / max(width, height))
            display_frame = cv2.resize(frame, (int(width * scale), int(height * scale))) if scale < 1.0 else frame
            cv2.imshow('Smart Parking System (Hybrid)', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()

print(f"\n✅ Done. Video saved to {config.OUTPUT_FILENAME}")
