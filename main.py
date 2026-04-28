import cv2
import time

from detection.detector import PersonDetector
from tracking.tracker import PersonTracker
from behavior.behavior import BehaviorAnalyzer
from utils.utils import log_event, save_screenshot

detector = PersonDetector()
tracker = PersonTracker()
behavior = BehaviorAnalyzer()

cap = cv2.VideoCapture(0)

prev_time = 0

if not cap.isOpened():
    print("Camera not opening ❌")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    detections = detector.detect(frame)
    tracked = tracker.update(detections, frame)
    alerts = behavior.check_loitering(tracked)

    for obj in tracked:
        x1, y1, x2, y2 = obj["bbox"]
        track_id = obj["id"]

        color = (0, 255, 0)

        if track_id in alerts:
            color = (0, 0, 255)

            cv2.putText(frame, "ALERT: Loitering", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            log_event(track_id, "Loitering")
            save_screenshot(frame, track_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # FPS Display
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Title
    cv2.putText(frame, "Smart Surveillance System", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # Add label on screen
    cv2.putText(frame, "GREEN: Normal  RED: Alert", (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("Final System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()