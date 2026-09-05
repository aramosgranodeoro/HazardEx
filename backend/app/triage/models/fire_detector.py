from ultralytics import YOLO
import cv2
import base64


class FireSmokeDetector:
    _instance = None

    def __init__(self):
        self.model = YOLO("./app/triage/models/YOLO/fire.pt")

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

        # Generar imagen con bounding boxes solo cuando se solicite
        if annotate and detections:
            annotated_image = results[0].plot()

            success, buffer = cv2.imencode(".jpg", annotated_image)

            if success:
                response["annotated_image"] = base64.b64encode(
                    buffer
                ).decode("utf-8")

        return response