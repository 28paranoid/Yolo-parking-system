# config.py
import os
import shutil
import torch

# --- DEVICE ---
DEVICE_INDEX = 0 if torch.cuda.is_available() else 'cpu'
YOLO_DEVICE = f"cuda:{DEVICE_INDEX}" if DEVICE_INDEX != 'cpu' else 'cpu'

# --- VIDEO ---
VIDEO_FILENAME = 'parking.mp4'
OUTPUT_FILENAME = 'parking_result_video.mp4'

# ============================================================
# HYBRID DETECTION (two models)
# ============================================================
# 1) Pretrained COCO model -> robust VEHICLE detection.
PRETRAINED_MODEL_PATH = 'yolov8n.pt'
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle']   # COCO class names treated as vehicles

# 2) YOUR custom-trained model -> PARKING-SPACE detection (and optionally its own car class).
#    Train this on your Label-Studio-annotated Phuentsholing dataset.
CUSTOM_MODEL_PATH = 'best.pt'
SPACE_CLASS_NAME = 'parking_space'   # the class name in YOUR model that marks a parking spot
CUSTOM_CAR_CLASSES = ['car']         # class name(s) in YOUR model that are vehicles (for fusion)

# Hybrid switches
USE_CUSTOM_FOR_SPACES = True   # parking spots come from your custom model's space detections
FUSE_CUSTOM_VEHICLES = True    # also merge your custom 'car' detections with the COCO vehicles
# If best.pt is missing, the code automatically falls back to the old
# "spots = where vehicles stay stationary" calibration so it still runs.

# --- Detection confidence ---
CONFIDENCE_THRESHOLD = 0.35    # vehicles (pretrained)
SPACE_CONFIDENCE = 0.35        # parking spaces (custom model)
NMS_IOU = 0.6                  # merge duplicate vehicle boxes coming from the two models

# --- Occupancy / spot geometry ---
MIN_STATIONARY_TIME = 5.0
SPOT_MERGE_DISTANCE = 30
MEMORY_FRAMES = 60
MIN_SPOT_AREA = 1500
IOU_THRESHOLD = 0.15           # a vehicle must cover >= this fraction of a spot to count
MAX_MOVEMENT_FOR_PARKED = 10   # px movement still considered "parked"

# --- Occupancy robustness (hysteresis + flicker grace) ---
IOU_KEEP_THRESHOLD = 0.07      # lower overlap bar to KEEP counting once entry is in progress
ENTRY_MISS_GRACE = 4           # processed frames without overlap tolerated before progress resets
REQUIRE_CENTER_IN_SPOT = True  # vehicle's ground point must sit INSIDE the spot (rejects passing cars beside it)
MIN_TRACK_AGE = 3              # frames a vehicle must be tracked before it can occupy a spot

DEBUG_OCCUPANCY = False         # True -> prints per-spot overlap each processed frame (diagnostics)

# --- Occupancy confirmation (in SECONDS of real video time) ---
# Counted at the ~5 FPS processing rate, NOT the raw video FPS.
ENTRY_CONFIRM_SECONDS = 2.0   # car must sit in a spot this long -> ENTRY
EXIT_CONFIRM_SECONDS = 1.5    # spot must stay empty this long   -> EXIT

# --- Calibration (sampling positions across the video) ---
CALIBRATION_SCAN_POINTS = [0.0, 0.25, 0.50, 0.75]
FRAMES_PER_SCAN = 400

# --- Output ---
OUTPUT_FPS = 1
# Live preview window. True = watch it run (press 'q' to stop).
# False = no window, render the output video only (noticeably faster).
SHOW_PREVIEW = True
# Longest side (px) of the live preview window; keeps the video's aspect ratio
# (portrait stays portrait). The saved output video is always full original size.
PREVIEW_MAX_SIZE = 900

# --- Manual parking-spot marking ---
MANUAL_SPOTS_FILE = 'parking_spots.json'   # produced by mark_spots.py
SPOT_FREE_COLOR     = (0, 255, 255)   # YELLOW (BGR) - empty parking spot
SPOT_OCCUPIED_COLOR = (0, 0, 255)     # RED    (BGR) - full / occupied parking spot
CAR_COLOR           = (0, 255, 0)     # GREEN  (BGR) - detected / tracked vehicles

# --- Tracker ---
TRACKER_TYPE = "deepocsort"
REID_WEIGHTS = 'osnet_x0_25_msmt17.pt'

# --- Paths ---
LOG_DIR = 'parking_logs'
CAPTURES_DIR = os.path.join(LOG_DIR, 'captures')
CSV_FILE = os.path.join(LOG_DIR, 'parking_history.csv')


def setup_directories():
    # Reset directories for a clean run
    if os.path.exists(LOG_DIR):
        try:
            shutil.rmtree(LOG_DIR)
        except PermissionError:
            pass

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CAPTURES_DIR, exist_ok=True)
    print(f"📂 Created directories: {LOG_DIR}, {CAPTURES_DIR}")
