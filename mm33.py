from ultralytics import YOLO
import cv2
import math
import numpy as np

def get_centroid(x1, y1, x2, y2):
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("yolo-Weights/yolov8n.pt")
model.classes = [0]  # Only detect persons

people_last_seen = {}
all_person_ids = set()
person_id_count = 0
max_distance = 100
frame_threshold = 3  # Person must be detected in at least 3 frames to count as new
person_detected_frames = {}  # Tracks how many frames a potential new person has been seen

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    current_frame_people = {}

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            centroid = get_centroid(x1, y1, x2, y2)
            confidence = math.ceil(box.conf[0] * 100) / 100

            if confidence > 0.82:
                found = False
                for pid, (last_centroid, _) in people_last_seen.items():
                    distance = np.linalg.norm(np.array(centroid) - np.array(last_centroid))
                    if distance < max_distance:
                        current_frame_people[pid] = (centroid, box)
                        found = True
                        person_detected_frames[pid] = frame_threshold  # Reset the frame count
                        break

                if not found:
                    new_id = person_id_count + 1
                    if new_id in person_detected_frames:
                        if person_detected_frames[new_id] > 0:
                            person_detected_frames[new_id] -= 1
                        if person_detected_frames[new_id] == 0:
                            person_id_count += 1
                            current_frame_people[person_id_count] = (centroid, box)
                            all_person_ids.add(person_id_count)
                    else:
                        person_detected_frames[new_id] = frame_threshold - 1

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(img, f"ID {person_id_count}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    people_last_seen = current_frame_people
    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) == ord('q'):
        print(f"Total unique persons seen: {len(all_person_ids)}")
        break

cap.release()
cv2.destroyAllWindows()


