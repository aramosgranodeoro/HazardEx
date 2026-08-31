"""
Evalua un modelo entrenado y desglosa precision, recall y mAP50 por clase
(fire / smoke / smoke_and_fire). Sirve para comparar el experimento
baseline vs el experimento con oversampling en la memoria del TFG.

Uso:
    python eval_por_clase.py --weights runs_fire_smoke/yolov8s_baseline/weights/best.pt
    python eval_por_clase.py --weights runs_fire_smoke/yolov8s_oversampled/weights/best.pt --split test
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Ruta al best.pt")
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    modelo = YOLO(args.weights)
    metricas = modelo.val(data=args.data, split=args.split)

    nombres_clases = metricas.names  # dict {id: nombre}

    print(f"\n=== Resultados globales ({args.split}) ===")
    print(f"mAP50-95: {metricas.box.map:.4f}")
    print(f"mAP50:    {metricas.box.map50:.4f}")

    print(f"\n=== Desglose por clase ===")
    print(f"{'Clase':<16}{'Precision':>10}{'Recall':>10}{'mAP50':>10}{'mAP50-95':>10}")

    # ap_class_index indica el orden de las clases en los arrays p, r, ap50, ap
    for idx, clase_id in enumerate(metricas.box.ap_class_index):
        nombre = nombres_clases[int(clase_id)]
        precision = metricas.box.p[idx]
        recall = metricas.box.r[idx]
        map50 = metricas.box.ap50[idx]
        map5095 = metricas.box.ap[idx]
        print(f"{nombre:<16}{precision:>10.4f}{recall:>10.4f}{map50:>10.4f}{map5095:>10.4f}")

    print("\nCriterio orientativo: si el recall de 'fire' es notablemente "
          "inferior (p. ej. >10-15 puntos) al de 'smoke', se justifica "
          "un segundo experimento con oversampling.")


if __name__ == "__main__":
    main()