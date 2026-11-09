# Smart Parking System

A computer-vision system that detects parking-spot occupancy from video. It uses
a hybrid YOLOv8 pipeline — a pretrained COCO model for robust vehicle detection
plus a custom-trained model (`best.pt`) for the parking spaces — with a
DeepOCSORT tracker. Parking spots are **marked manually** before the model runs,
and the system logs every vehicle ENTRY / EXIT to a CSV with snapshots.

- Empty spot → **yellow**  ·  Occupied spot → **red**  ·  Tracked car → **green**

---

## 1. Requirements

- **Python 3.11** (recommended)
- A CUDA GPU is optional — it runs on CPU, just slower
- The large asset files (not in this repo, get them from the shared Drive folder):
  - `parking.mp4` — the input video
  - `best.pt` — custom-trained parking-space model
  - `yolov8n.pt` — pretrained vehicle model (also auto-downloaded by setup)
  - `osnet_x0_25_msmt17.pt` — tracker re-ID weights

Place all four files in the project folder, next to `main.py`.

## 2. Setup

Install the dependencies. Either run the setup script:

```bash
python setup.py
```

or install directly:

```bash
pip install -r requirements.txt
```

`setup.py` also downloads `yolov8n.pt` automatically if it's missing.
> On Windows you may need the **Microsoft C++ Build Tools** for some packages.

## 3. Mark the parking spots (do this first, once per video)

```bash
python mark_spots.py
```

This opens the first frame of `parking.mp4`. Draw each spot:

| Action | Key / Mouse |
|---|---|
| Add a corner point to the current spot | **Left-click** |
| Finish the current spot (needs ≥ 3 points) | **Right-click** |
| Undo the last point | **u** |
| Delete the last finished spot | **z** |
| Clear everything | **c** |
| Save and quit | **s** |
| Quit without saving | **q** / **Esc** |

Spots are saved to `parking_spots.json`. Re-running the tool loads your existing
spots so you can tweak them.

## 4. Run the detector

```bash
python main.py
```

It loads your marked spots, then processes the video — drawing spots and cars,
and printing `[ENTRY]` / `[EXIT]` events. Press **q** to stop (when the preview
is on). Output:

- `parking_result_video.mp4` — annotated video
- `parking_logs/parking_history.csv` — ENTRY / EXIT event log
- `parking_logs/captures/` — snapshot image per event

---

## 5. Live dashboard (optional)

A Streamlit dashboard reads the same files `main.py` writes and refreshes
automatically, so it updates live while the model runs.

```bash
streamlit run streamlit_app.py
```

Run it from the project folder (so it finds `parking_logs/` and
`parking_spots.json`). Start `main.py` in one terminal and the dashboard in
another — as events are logged, the dashboard updates every few seconds.

It shows: occupancy KPIs, a lot map colored by occupancy, occupancy-over-time,
a per-spot timeline, the event log with Bhutan parking-rate fees, and recent
snapshots. If no run has happened yet, it shows sample data so you can demo it.

---

## Configuration (`config.py`)

| Setting | Meaning |
|---|---|
| `VIDEO_FILENAME` | Input video file |
| `SHOW_PREVIEW` | `True` = live window; `False` = render video only (faster) |
| `IOU_THRESHOLD` | Overlap needed to **start** counting a spot as occupied |
| `IOU_KEEP_THRESHOLD` | Lower overlap needed to **keep** counting (hysteresis) |
| `ENTRY_CONFIRM_SECONDS` / `EXIT_CONFIRM_SECONDS` | How long a car must stay / be gone before an event is logged |
| `ENTRY_MISS_GRACE` | Frames of no-overlap tolerated before entry progress resets |
| `SPOT_FREE_COLOR` / `SPOT_OCCUPIED_COLOR` / `CAR_COLOR` | Overlay colors (BGR) |
| `DEBUG_OCCUPANCY` | `True` = print per-spot overlap/counter each frame (diagnostics) |

## Project structure

```
main.py            Main pipeline: load spots, track, decide occupancy, render
mark_spots.py      Interactive tool to draw parking spots on the first frame
config.py          All settings
classes.py         Spot loader, vehicle tracker/memory
utils.py           CSV logging, NMS box merge, crop & snapshot helpers
setup.py           Installs dependencies, downloads yolov8n.pt
parking_spots.json Your manually-marked spots (created by mark_spots.py)
parking_logs/      CSV event log + event snapshots (created at runtime)
```

## Troubleshooting

- **"No manually-marked spots found"** → run `python mark_spots.py` first.
- **A spot is slow to register** → its box may be drawn loosely; redraw it tighter,
  or set `DEBUG_OCCUPANCY = True` to see its overlap values.
- **Runs slowly** → set `SHOW_PREVIEW = False`, or use a CUDA GPU.
