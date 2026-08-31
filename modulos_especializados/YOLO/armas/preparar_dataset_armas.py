from pathlib import Path
import random
import shutil
from collections import defaultdict

# ==========================
# CONFIGURACIÓN
# ==========================
ROOT = Path(r"D:/OD-WeaponDetection-master/OD-WeaponDetection-master/Weapons and similar handled objects/Sohas_weapon-Detection-YOLOv5/obj_train_data")
TRAIN_IMAGES = ROOT / "images" / "train"
TRAIN_LABELS = ROOT / "labels" / "train"
VAL_IMAGES = ROOT / "images" / "val"
VAL_LABELS = ROOT / "labels" / "val"
VAL_IMAGES.mkdir(parents=True, exist_ok=True)
VAL_LABELS.mkdir(parents=True, exist_ok=True)

VAL_PERCENT = 0.20
SEED = 42

# Clases originales del dataset (orden fijo del dataset.yaml de Sohas):
# 0 pistol, 1 smartphone, 2 knife, 3 monedero, 4 billete, 5 tarjeta
# Nos quedamos solo con pistol y knife, renumerando: pistol->0, knife->1
MAPEO_CLASES = {0: 0, 2: 1}
CLASES_FINALES = ["pistol", "knife"]

# Si True, borra también las imágenes que tras el filtrado se quedan sin
# ninguna caja (ni pistol ni knife). Si False (recomendado), se mantienen
# como negativos/background, lo cual ayuda a YOLO a reducir falsos positivos.
ELIMINAR_IMAGENES_SIN_OBJETIVO = False

random.seed(SEED)


# ==========================
# PASO 1: filtrar labels a solo pistol/knife
# ==========================
def filtrar_labels():
    n_lineas_originales = 0
    n_lineas_conservadas = 0
    imagenes_sin_objetivo = []

    label_files = list(TRAIN_LABELS.glob("*.txt"))
    for label_path in label_files:
        lineas_finales = []
        with open(label_path, "r", encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea:
                    continue
                n_lineas_originales += 1
                partes = linea.split()
                idx_original = int(partes[0])
                if idx_original in MAPEO_CLASES:
                    nuevo_idx = MAPEO_CLASES[idx_original]
                    lineas_finales.append(" ".join([str(nuevo_idx)] + partes[1:]))
                    n_lineas_conservadas += 1

        if lineas_finales:
            label_path.write_text("\n".join(lineas_finales) + "\n", encoding="utf-8")
        else:
            label_path.write_text("", encoding="utf-8")
            imagenes_sin_objetivo.append(label_path.stem)

    print("=== Filtrado de labels (solo pistol/knife) ===")
    print(f"Líneas originales:   {n_lineas_originales}")
    print(f"Líneas conservadas:  {n_lineas_conservadas}")
    print(f"Imágenes sin pistol/knife tras filtrar: {len(imagenes_sin_objetivo)}")

    if ELIMINAR_IMAGENES_SIN_OBJETIVO:
        for stem in imagenes_sin_objetivo:
            label_path = TRAIN_LABELS / f"{stem}.txt"
            label_path.unlink(missing_ok=True)
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                img_path = TRAIN_IMAGES / f"{stem}{ext}"
                if img_path.exists():
                    img_path.unlink()
                    break
        print(f"Eliminadas {len(imagenes_sin_objetivo)} imágenes sin pistol/knife.")
    print()


# ==========================
# PASO 2: split train/val balanceado por clase
# ==========================
def clasificar_imagenes():
    """Agrupa imágenes según qué clases contienen: solo pistol, solo knife, ambas, ninguna."""
    grupos = defaultdict(list)

    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        images.extend(TRAIN_IMAGES.glob(ext))

    for img in images:
        label_path = TRAIN_LABELS / f"{img.stem}.txt"
        clases_presentes = set()
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                for linea in f:
                    linea = linea.strip()
                    if linea:
                        clases_presentes.add(int(linea.split()[0]))

        if clases_presentes == {0}:
            grupo = "solo_pistol"
        elif clases_presentes == {1}:
            grupo = "solo_knife"
        elif clases_presentes == {0, 1}:
            grupo = "ambas"
        else:
            grupo = "ninguna"

        grupos[grupo].append(img)

    return grupos


def split_balanceado():
    grupos = clasificar_imagenes()
    total = sum(len(v) for v in grupos.values())

    print("=== Distribución antes del split (por grupo) ===")
    for grupo, imgs in grupos.items():
        print(f"  {grupo:<12}: {len(imgs)}")

    # Se toma el 20% DENTRO de cada grupo -> mantiene la proporción de
    # pistol/knife/ambas/negativos igual en train y en val.
    val_images = []
    for imgs in grupos.values():
        imgs_copia = imgs[:]
        random.shuffle(imgs_copia)
        n_val = round(len(imgs_copia) * VAL_PERCENT)
        val_images.extend(imgs_copia[:n_val])

    print(f"\nTotal imágenes : {total}")
    print(f"Validación     : {len(val_images)} ({len(val_images) / total * 100:.1f}%)")

    for img in val_images:
        label = TRAIN_LABELS / f"{img.stem}.txt"
        shutil.move(str(img), str(VAL_IMAGES / img.name))
        if label.exists():
            shutil.move(str(label), str(VAL_LABELS / label.name))

    print("Conjunto de validación creado de forma balanceada.")


if __name__ == "__main__":
    filtrar_labels()
    split_balanceado()