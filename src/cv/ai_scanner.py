# --------------------------------------------------
#  src/cv/ai_scanner.py   –  YOLO‑v8n + ZBar listener
# --------------------------------------------------
"""
Locate barcodes with a tiny YOLO‑v8n detector, then try to decode them
with ZBar.  Every decoded string is pushed into a Queue that the
Streamlit sidebar polls in real‑time.
"""

from __future__ import annotations
from pathlib import Path
from queue import Queue

import cv2
from pyzbar.pyzbar import decode, ZBarSymbol
from ultralytics import YOLO

# ── local weights -----------------------------------------------------
WEIGHTS = Path(__file__).parent / "weights" / "yolov8n.pt"
if not WEIGHTS.exists():
    raise FileNotFoundError(
        f"{WEIGHTS} missing.\n"
        "Download once (≈8 MB):\n"
        "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt\n"
        "and place it in  src/cv/weights/"
    )

_model = YOLO(str(WEIGHTS))
_model.fuse()                                    # fuse layers in‑place

# ── continuous webcam listener ------------------------------------------------
def listen(
    cam_index: int,
    out_q: Queue,
    conf: float = 0.15,  # Changed the Confidence filter from 0.25 to current value
    window: str = "AI Scanner (ESC = quit / s = show)") -> None:

    """
    Grab frames → YOLO detect plausible 1‑D barcodes → smart pre‑process →
    ZBar decode → push result strings into *out_q*.
    """
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)     # To adjusted to 1080p
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)     # Changed to 1820x1080

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        # --- YOLO detection ---------------------------------------------
        for box in _model(frame, conf=conf, verbose=False)[0].boxes.xyxy.cpu():
            x1, y1, x2, y2 = box.int().tolist()
            w, h = x2 - x1, y2 - y1

            # keep only wide, flat rectangles (typical 1‑D barcode shape)
            if w < 1.4 * h or w * h < 7_000:    #lower filter from 10_000 to current value
                continue

            roi = frame[y1:y2, x1:x2]

            # --- PRE‑PROCESS for ZBar ----------------------------------
            gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Adjusted upscale from fx=2, fy=2
            up    = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            blur  = cv2.GaussianBlur(up, (0, 0), 1.5)
            sharp = cv2.addWeighted(up, 1.8, blur, -0.8, 0)

            # 1) global Otsu
            _, th1 = cv2.threshold(sharp, 0, 255, cv2.THRESH_OTSU)
            # 2) adaptive thresh (normal + inverted)
            th2 = cv2.adaptiveThreshold(
                sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 31, 7
            )
            th3 = cv2.bitwise_not(th2)

            # keep a reference so we can preview if user presses “s”
            cand_imgs = (th1, th2, th3)

            # try all three candidates
            for img in cand_imgs:
                for obj in decode(
                        img,
                        symbols=[
                            ZBarSymbol.EAN13,  # UPC‑A = EAN‑13 starting with ‘0’
                            ZBarSymbol.EAN8,
                            ZBarSymbol.UPCE,
                            ZBarSymbol.CODE128,
                        ],
                ):
                    code = obj.data.decode()  # define variable
                    print(f"[ZBAR] decoded → {code}")  # debug print
                    out_q.put(code)  # enqueue once

            # draw green rectangle for user feedback
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow(window, frame)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:  # ESC quits
            break
        if k == ord("s"):  # press “s” → show the three candidates
            cv2.imshow("cand‑1 (Otsu)", cand_imgs[0])
            cv2.imshow("cand‑2 (adapt)", cand_imgs[1])
            cv2.imshow("cand‑3 (invert)", cand_imgs[2])

    cap.release()
    cv2.destroyAllWindows()
