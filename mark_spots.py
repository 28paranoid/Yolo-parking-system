# mark_spots.py
# Interactive tool to MANUALLY mark parking spots on the first frame of the
# video, BEFORE running main.py. Spots are saved to parking_spots.json and
# loaded by main.py. Parking spots are drawn in RED.
import os
import json
import cv2
import numpy as np
import config

WINDOW = "Mark Parking Spots"
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)


def main():
    video = config.VIDEO_FILENAME
    if not os.path.exists(video):
        print(f"Video '{video}' not found. Put it next to this script.")
        return

    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Could not read the first frame of the video.")
        return

    h, w = frame.shape[:2]

    # Load existing spots so you can continue editing a previous marking.
    spots = []
    if os.path.exists(config.MANUAL_SPOTS_FILE):
        try:
            with open(config.MANUAL_SPOTS_FILE) as f:
                data = json.load(f)
            spots = [[tuple(p) for p in s["points"]] for s in data.get("spots", [])]
            print(f"Loaded {len(spots)} existing spots (press 'c' to clear all).")
        except Exception:
            spots = []

    current = []          # points of the polygon being drawn
    mouse = [0, 0]

    def on_mouse(event, x, y, flags, param):
        mouse[0], mouse[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(current) >= 3:
                spots.append(current.copy())
                current.clear()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, min(1280, w), min(720, h))
    cv2.setMouseCallback(WINDOW, on_mouse)

    print("================= MARK PARKING SPOTS =================")
    print(" Left click   : add a corner point to the current spot")
    print(" Right click  : finish the current spot (needs >= 3 points)")
    print(" u            : undo the last point")
    print(" z            : delete the last finished spot")
    print(" c            : clear everything")
    print(" s            : save and quit")
    print(" q / ESC      : quit without saving")
    print("=====================================================")

    while True:
        disp = frame.copy()

        # finished spots -> RED
        for i, poly in enumerate(spots):
            pts = np.array(poly, np.int32)
            cv2.polylines(disp, [pts], True, RED, 2)
            cx = int(np.mean([p[0] for p in poly]))
            cy = int(np.mean([p[1] for p in poly]))
            cv2.putText(disp, f"P{i + 1}", (cx - 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)

        # polygon in progress -> YELLOW
        if current:
            for p in current:
                cv2.circle(disp, p, 4, YELLOW, -1)
            if len(current) >= 2:
                cv2.polylines(disp, [np.array(current, np.int32)], False, YELLOW, 2)
            cv2.line(disp, current[-1], (mouse[0], mouse[1]), YELLOW, 1)

        cv2.rectangle(disp, (0, 0), (disp.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(disp, f"Spots: {len(spots)}  |  L-click=add  R-click=finish  s=save  q=quit",
                    (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow(WINDOW, disp)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('q'), 27):
            print("Quit without saving.")
            break
        elif key == ord('u'):
            if current:
                current.pop()
        elif key == ord('z'):
            if spots:
                spots.pop()
        elif key == ord('c'):
            spots.clear()
            current.clear()
        elif key == ord('s'):
            if len(current) >= 3:
                spots.append(current.copy())
                current.clear()
            data = {
                "video": video,
                "frame_size": [w, h],
                "spots": [{"points": [[int(x), int(y)] for (x, y) in poly]} for poly in spots],
            }
            with open(config.MANUAL_SPOTS_FILE, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(spots)} spots to {config.MANUAL_SPOTS_FILE}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
