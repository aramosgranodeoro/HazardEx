from pathlib import Path

# ==========================
# CONFIGURACIÓN
# ==========================
ROOT = Path(r"D:/OD-WeaponDetection-master/OD-WeaponDetection-master/Weapons and similar handled objects/Sohas_weapon-Detection-YOLOv5/obj_train_data")
TEST_IMAGES = ROOT / "images" / "test"
TEST_LABELS = ROOT / "labels" / "test"

# Clases originales del dataset (orden fijo del dataset.yaml de Sohas):
# 0 pistol, 1 smartphone, 2 knife, 3 monedero, 4 billete, 5 tarjeta
# Nos quedamos solo con pistol y knife, renumerando: pistol->0, knife->1
MAPEO_CLASES = {0: 0, 2: 1}
CLASES_FINALES = ["pistol", "knife"]

# Si True, borra también las imágenes que tras el filtrado se quedan sin
# ninguna caja (ni pistol ni knife). Si False (recomendado), se mantienen
# como negativos/background.
ELIMINAR_IMAGENES_SIN_OBJETIVO = False


def filtrar_labels_test():
    n_lineas_originales = 0
    n_lineas_conservadas = 0
    imagenes_sin_objetivo = []

    label_files = list(TEST_LABELS.glob("*.txt"))
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

    print("=== Filtrado de labels del split TEST (solo pistol/knife) ===")
    print(f"Líneas originales:   {n_lineas_originales}")
    print(f"Líneas conservadas:  {n_lineas_conservadas}")
    print(f"Imágenes sin pistol/knife tras filtrar: {len(imagenes_sin_objetivo)}")

    if ELIMINAR_IMAGENES_SIN_OBJETIVO:
        for stem in imagenes_sin_objetivo:
            label_path = TEST_LABELS / f"{stem}.txt"
            label_path.unlink(missing_ok=True)
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                img_path = TEST_IMAGES / f"{stem}{ext}"
                if img_path.exists():
                    img_path.unlink()
                    break
        print(f"Eliminadas {len(imagenes_sin_objetivo)} imágenes sin pistol/knife.")


if __name__ == "__main__":
    filtrar_labels_test()