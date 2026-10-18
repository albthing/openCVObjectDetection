import cv2
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

camera = cv2.VideoCapture(1)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    raise RuntimeError('Camera not found')

while True:
    success, frame = camera.read()

    if not success:
        break
    
    results = model.predict(frame, conf=0.45, verbose=False)
    result = results[0]

    bestTarget = None

    for box in result.boxes:
        classId = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        label = model.names[classId]

        cv2.rectangle(
            frame, (x1, y1), (x2, y2), (0, 255, 0), 2
        )

        cv2.putText(
            frame, f"{label} {confidence: .2f} ({x1 - x2 //2}, {y1 + y2 // 2})",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 255, 0), 2
        )

        if (
            bestTarget is None
            or confidence > bestTarget["confidence"]
        ):
            bestTarget = {
                "label": label,
                "confidence": confidence,
                "x": (x1 + x2) // 2,
                "y": (y1 + y2) // 2,
                "w": x2 - x1,
                "h": y1 - y2,
            }

    height, width = frame.shape[:2]
    cv2.drawMarker(
        frame, (width // 2, height // 2),
        (255, 255, 0), cv2.MARKER_CROSS, 20, 2
    )

    if bestTarget is not None:
        print(
            f"{bestTarget['label']} "
            f"at ({bestTarget['x']}, {bestTarget['y']})"
        )

    cv2.imshow("Robot Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()