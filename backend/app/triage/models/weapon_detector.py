import cv2
import base64
from ultralytics import YOLO


class WeaponDetector:
    _instance = None

    def __init__(self):
        self.model = YOLO("./app/triage/models/YOLO/weapons.pt")

    def predict(self, image, annotate=False) -> dict:
        results = self.model.predict(
            image,
            conf=0.25,
            verbose=False
        )

        detections = []

        for box in results[0].boxes:
            detections.append({
                "class": self.model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()[0],
            })

        response = {
            "detected": len(detections) > 0,
            "detections": detections,
        }

        if annotate and detections:
            annotated = results[0].plot()

            success, buffer = cv2.imencode(".jpg", annotated)

            if success:
                response["annotated_image"] = base64.b64encode(
                    buffer
                ).decode("utf-8")

        return response