from ultralytics import YOLO

class WeaponDetector:
    _instance = None

    def __init__(self):
        self.model = YOLO("./app/triage/models/weapons.pt")

    def predict(self, image) -> dict:
        results = self.model.predict(image, conf=0.25, verbose=False)
        
        detections = []
        for box in results[0].boxes:
            detections.append({
                "class": self.model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()[0],
            })
        
        return {
            "detected": len(detections) > 0,
            "detections": detections,
        }