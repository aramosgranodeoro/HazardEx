from ultralytics import YOLO


def main():
    # Cargar modelo preentrenado
    model = YOLO("yolo11s.pt")

    # Entrenamiento
    results = model.train(
        data="data.yaml",
        epochs=150,
        imgsz=768,
        batch=16,
        device=0,
        patience=25,
        project="runs/detect",
        name="weapon_yolo11s",
    )

    # Validación sobre el conjunto de test
    metrics = model.val(
        data="data.yaml",
        split="test",
    )
    print("mAP50-95:", metrics.box.map)
    print("mAP50:", metrics.box.map50)


if __name__ == "__main__":
    main()