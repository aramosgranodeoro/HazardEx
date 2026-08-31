"""
balance_dataset_armas.py

Analiza el balance de clases del dataset Sohas (formato YOLO: un .txt por
imagen, cada línea "clase cx cy w h").

Cuenta, por split (train/test):
  - nº de instancias (bounding boxes) por clase
  - nº de imágenes en las que aparece cada clase (una imagen puede tener
    varias clases)
  - imágenes sin ninguna anotación (label vacío o inexistente)

Uso:
  python balance_dataset_armas.py --dataset_root "D:/OD-WeaponDetection-master/OD-WeaponDetection-master/Weapons and similar handled objects/Sohas_weapon-Detection-YOLOv5/obj_train_data"
"""
#D:/OD-WeaponDetection-master/OD-WeaponDetection-master/Weapons and similar handled objects/Sohas_weapon-Detection-YOLOv5/obj_train_data
import argparse
from collections import Counter
from pathlib import Path

CLASES = ["pistol","knife"]


def analizar_split(labels_dir: Path, images_dir: Path) -> dict:
    """Devuelve estadísticas de instancias/imágenes por clase para un split."""
    instancias_por_clase = Counter()
    imagenes_por_clase = Counter()
    n_imagenes_sin_anotacion = 0
    n_labels_con_clase_invalida = 0

    extensiones_img = {".jpg", ".jpeg", ".png"}
    imagenes = [p for p in images_dir.iterdir() if p.suffix.lower() in extensiones_img]

    for img_path in imagenes:
        label_path = labels_dir / f"{img_path.stem}.txt"

        if not label_path.exists() or label_path.stat().st_size == 0:
            n_imagenes_sin_anotacion += 1
            continue

        clases_en_imagen = set()
        with open(label_path, "r", encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea:
                    continue
                partes = linea.split()
                try:
                    idx_clase = int(partes[0])
                except (ValueError, IndexError):
                    n_labels_con_clase_invalida += 1
                    continue

                if idx_clase < 0 or idx_clase >= len(CLASES):
                    n_labels_con_clase_invalida += 1
                    continue

                instancias_por_clase[idx_clase] += 1
                clases_en_imagen.add(idx_clase)

        for idx_clase in clases_en_imagen:
            imagenes_por_clase[idx_clase] += 1

    return {
        "n_imagenes_total": len(imagenes),
        "n_imagenes_sin_anotacion": n_imagenes_sin_anotacion,
        "n_labels_con_clase_invalida": n_labels_con_clase_invalida,
        "instancias_por_clase": instancias_por_clase,
        "imagenes_por_clase": imagenes_por_clase,
    }


def imprimir_tabla(nombre_split: str, stats: dict) -> None:
    total_instancias = sum(stats["instancias_por_clase"].values())

    print(f"\n=== Split: {nombre_split} ===")
    print(f"Imágenes totales:        {stats['n_imagenes_total']}")
    print(f"Imágenes sin anotación:  {stats['n_imagenes_sin_anotacion']}")
    if stats["n_labels_con_clase_invalida"]:
        print(f"Líneas con clase inválida (ignoradas): {stats['n_labels_con_clase_invalida']}")
    print(f"Instancias totales (bboxes): {total_instancias}\n")

    print(f"{'Clase':<12}{'Instancias':>12}{'% instancias':>15}{'Imágenes':>12}{'% imágenes':>14}")
    print("-" * 65)
    for idx, nombre_clase in enumerate(CLASES):
        n_inst = stats["instancias_por_clase"].get(idx, 0)
        n_img = stats["imagenes_por_clase"].get(idx, 0)
        pct_inst = (n_inst / total_instancias * 100) if total_instancias else 0.0
        pct_img = (n_img / stats["n_imagenes_total"] * 100) if stats["n_imagenes_total"] else 0.0
        print(f"{nombre_clase:<12}{n_inst:>12}{pct_inst:>14.1f}%{n_img:>12}{pct_img:>13.1f}%")


def calcular_ratio_desbalance(stats: dict) -> None:
    conteos = [stats["instancias_por_clase"].get(i, 0) for i in range(len(CLASES))]
    conteos_no_cero = [c for c in conteos if c > 0]
    if not conteos_no_cero:
        return
    ratio = max(conteos_no_cero) / min(conteos_no_cero)
    clase_max = CLASES[conteos.index(max(conteos))]
    clase_min = CLASES[conteos.index(min(c for c in conteos if c > 0))]
    print(f"\nRatio de desbalance (clase mayoritaria / minoritaria): {ratio:.1f}x")
    print(f"  Mayoritaria: {clase_max} | Minoritaria: {clase_min}")
    if ratio >= 3:
        print("  -> Desbalance notable: considera class weights, oversampling de la clase minoritaria")
        print("     o ajustar el muestreo estratificado como ya haces en tus scripts de evaluación.")


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Balance de clases del dataset Sohas (YOLO)")
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Ruta a Sohas_weapon-Detection-YOLOv5 (contiene obj_train_data/)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Splits a analizar (por defecto: train val test)",
    )
    return parser.parse_args()


def main() -> None:
    args = parsear_argumentos()
    dataset_root = Path(args.dataset_root)

    stats_global = Counter()

    for split in args.splits:
        images_dir = dataset_root / "images" / split
        labels_dir = dataset_root / "labels" / split

        if not images_dir.exists() or not labels_dir.exists():
            print(f"AVISO: no existe images/{split} o labels/{split}, se omite.")
            continue

        stats = analizar_split(labels_dir, images_dir)
        imprimir_tabla(split, stats)
        calcular_ratio_desbalance(stats)

        stats_global.update(stats["instancias_por_clase"])

    if len(args.splits) > 1:
        print("\n=== TOTAL (todos los splits analizados) ===")
        total = sum(stats_global.values())
        print(f"{'Clase':<12}{'Instancias':>12}{'% instancias':>15}")
        print("-" * 40)
        for idx, nombre_clase in enumerate(CLASES):
            n_inst = stats_global.get(idx, 0)
            pct = (n_inst / total * 100) if total else 0.0
            print(f"{nombre_clase:<12}{n_inst:>12}{pct:>14.1f}%")


if __name__ == "__main__":
    main()