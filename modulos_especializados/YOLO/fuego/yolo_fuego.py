from ultralytics import YOLO


def main():
    # yolov8s: buen equilibrio precision/velocidad para 12GB VRAM.
    # Si el dataset es dificil de aprender, prueba yolov8m (mas VRAM/tiempo).
    modelo = YOLO("yolo11s.pt")

    modelo.train(
        data="data.yaml",
        epochs=150,
        imgsz=768,
        batch=16,
        device=0,
        patience=25, # early stopping si no mejora el mAP50 en 25 epochs
        project="runs_v2",
        name="smoke_fire"
    )

    # Evaluacion del modelo entrenado sobre el conjunto de test
    metricas = modelo.val(data="data.yaml", split="test")
    print(metricas.box.map)      # mAP50-95
    print(metricas.box.map50)    # mAP50


if __name__ == "__main__":
    main()