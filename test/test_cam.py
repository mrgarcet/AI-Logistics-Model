import cv2, itertools

backend = cv2.CAP_DSHOW         # try CAP_MSMF if needed
for i in itertools.count(0):
    cap = cv2.VideoCapture(i, backend)
    if not cap.isOpened():
        print(f"index {i} failed (no more cameras)"); break
    ok, _ = cap.read()
    cap.release()
    print(f"index {i} → {'OK' if ok else 'read failed'}")
