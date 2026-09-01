import base64
import csv
import cv2
import io
import json
import math
import os
import re
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText

# ── Configuración ─────────────────────────────────────────────────────────────

MODEL_ID   = "BSC-LT/Salamandra-VL-7B-2512"
OUTPUT_CSV = "resultados_salamandra_vl_7b.csv"

CARPETAS = {
    "violencia": r"D:\DatasetTriaje\Violencia",
    "armas":     r"D:\DatasetTriaje\Armas",
    "fuego":     r"D:\DatasetTriaje\Fuego",
    "accidente": r"D:\DatasetTriaje\Coches",
    "normal":    r"D:\DatasetTriaje\Normal",
}

EXTENSIONES_IMG   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXTENSIONES_VIDEO = {".mp4", ".avi", ".mov", ".mkv"}
MAX_CHARTS        = 250

PROMPT = """Eres un analista visual de seguridad. Analiza esta imagen o secuencia de vídeo y clasifícala en UNA de estas categorías:

- violencia:  pelea física, agresión o violencia entre personas
- armas:      cuchillo, arma de fuego u objeto peligroso visible
- fuego:      llamas, humo, objetos o edificios en llamas
- accidente:  colisión de tráfico, accidente de coche, vehículos dañados o fuera de la carretera
- grafico:    cualquier gráfico, tabla, infografía o artículo con datos visuales
- normal:     contenido cotidiano seguro que no encaja en ninguna categoría anterior

Reglas:
1. Si ves CUALQUIER gráfico, diagrama o visualización de datos → elige siempre "grafico"
2. Escribe ÚNICAMENTE una de estas palabras exactas: violencia, armas, fuego, accidente, grafico, normal
3. No se aceptan otros valores

Responde ÚNICAMENTE con JSON válido:
{
    "categoria_predicha": "escribe aquí solo una palabra de la lista",
    "confianza": 0.00,
    "descripcion": "una frase explicando el razonamiento"
}"""

# ── Carga del modelo (singleton) ──────────────────────────────────────────────

from transformers import BitsAndBytesConfig

_processor = None
_model     = None

def cargar_modelo():
    global _processor, _model
    if _model is not None:
        return
    print(f"Cargando {MODEL_ID} en 4-bit ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    _model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="cuda",
    )
    _model.eval()
    print("Modelo cargado ✓")


# ── Utilidades de imagen/vídeo ────────────────────────────────────────────────

def extraer_frames_video(video_path, n_frames=9):
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []
    n_ini   = max(1, int(n_frames * 0.3))
    n_fin   = n_frames - n_ini
    idx_ini = [int(total * i / (n_ini * 3)) for i in range(n_ini)]
    idx_fin = [int(total * (0.5 + 0.5 * i / n_fin)) for i in range(n_fin)]
    indices = sorted(set(idx_ini + idx_fin))
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((224, 168))
            )
    cap.release()
    return frames


def frames_a_grid(frames, cols=3):
    n    = len(frames)
    rows = math.ceil(n / cols)
    w, h = frames[0].size
    grid = Image.new("RGB", (w * cols, h * rows), color=(0, 0, 0))
    for i, frame in enumerate(frames):
        frame = frame.copy()
        draw  = ImageDraw.Draw(frame)
        draw.rectangle([(0, 0), (50, 18)], fill=(0, 0, 0))
        draw.text((3, 3), f"t={i+1}/{n}", fill=(255, 255, 0))
        row, col = divmod(i, cols)
        grid.paste(frame, (col * w, row * h))
    return grid


def cargar_como_imagen(file_path):
    ext = Path(file_path).suffix.lower()
    if ext in EXTENSIONES_IMG:
        return Image.open(file_path).convert("RGB")
    elif ext in EXTENSIONES_VIDEO:
        frames = extraer_frames_video(file_path)
        if not frames:
            return None
        return frames_a_grid(frames) if len(frames) > 1 else frames[0]
    return None


# ── Inferencia ────────────────────────────────────────────────────────────────

def analizar_imagen(pil_image):
    cargar_modelo()

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT},
        ]
    }]

    # Normaliza tamaño para consistencia con grids de vídeo
    pil_image = pil_image.convert("RGB").resize((672, 672))

    text   = _processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = _processor(
        text=text,
        images=[pil_image],
        return_tensors="pt"
    ).to(_model.device, torch.float16)

    t0 = time.time()
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=1.0,   # do_sample=False ignora temperature, pero evita warning
        )
    elapsed_ms = (time.time() - t0) * 1000

    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    raw     = _processor.decode(new_ids, skip_special_tokens=True)

    prompt_tokens = int(inputs["input_ids"].shape[-1])
    gen_tokens    = int(new_ids.shape[0])

    metricas = {
        "total_ms":         round(elapsed_ms, 1),
        "prompt_ms":        0,
        "eval_ms":          round(elapsed_ms, 1),
        "prompt_tokens":    prompt_tokens,
        "generated_tokens": gen_tokens,
        "tokens_per_sec":   round(gen_tokens / max(elapsed_ms / 1000, 1e-9), 1),
    }
    return raw.strip(), metricas


# ── Parser ────────────────────────────────────────────────────────────────────

CATEGORIAS_VALIDAS = {"violencia", "armas", "fuego", "accidente", "grafico", "normal"}

# Palabras clave para inferir categoría cuando el modelo no devuelve JSON válido
PALABRAS_CLAVE = {
    "armas":     ["arma", "cuchillo", "pistola", "navaja", "rifle"],
    "violencia": ["violencia", "pelea", "agresión", "agresion", "golpe"],
    "fuego":     ["fuego", "llama", "incendio", "humo"],
    "accidente": ["accidente", "colisión", "colision", "choque", "crash"],
    "grafico":   ["gráfico", "grafico", "tabla", "infografía", "infografia"],
    "normal":    ["normal"],
}

def _inferir_por_palabras_clave(texto):
    texto_low = texto.lower()
    for cat, palabras in PALABRAS_CLAVE.items():
        if any(p in texto_low for p in palabras):
            return cat
    return "ERROR"


def parsear_respuesta(raw):
    # 1) Intento JSON directo
    try:
        clean = raw.strip()
        if "```" in clean:
            clean = clean.split("```")[1].removeprefix("json").strip()
        data  = json.loads(clean)
        pred  = str(data.get("categoria_predicha", "ERROR")).lower().strip()
        # normaliza variantes y espacios
        pred  = re.sub(r"\s+", " ", pred).strip()
        pred  = pred.replace("gráfico", "grafico").replace("accidente de tráfico", "accidente")
        pred  = pred.replace("pelea", "violencia").replace("humo", "fuego")
        pred  = pred.replace("arma", "armas").replace("crash", "accidente")
        if pred not in CATEGORIAS_VALIDAS:
            pred = _inferir_por_palabras_clave(raw)
        return pred, data.get("confianza", 0), data.get("descripcion", "")
    except json.JSONDecodeError:
        pass

    # 2) Intento regex sobre el campo categoria_predicha
    match = re.search(r'"categoria_predicha"\s*:\s*"?\s*([a-záéíóú]+)', raw, re.IGNORECASE)
    if match:
        pred = match.group(1).lower().strip()
        if pred in CATEGORIAS_VALIDAS:
            return pred, 0, raw

    # 3) Fallback: inferir por palabras clave en el texto libre
    pred = _inferir_por_palabras_clave(raw)
    return pred, 0, raw


# ── Fuentes de datos ──────────────────────────────────────────────────────────

def cargar_archivos_carpeta(label, carpeta):
    items = []
    for f in Path(carpeta).rglob("*"):
        if f.suffix.lower() in EXTENSIONES_IMG | EXTENSIONES_VIDEO:
            items.append({"label": label, "path": f, "source": "local"})
    return items


def cargar_chartqa(max_items=MAX_CHARTS):
    print(f"  Cargando ChartQA (máx {max_items} imágenes)...")
    dataset = load_dataset("HuggingFaceM4/ChartQA", split="test", streaming=True)
    vistos, items = set(), []
    for ejemplo in dataset:
        img_id = ejemplo.get("imgname", str(len(items)))
        if img_id in vistos:
            continue
        vistos.add(img_id)
        items.append({"label": "grafico", "image": ejemplo["image"], "source": "chartqa"})
        if len(items) >= max_items:
            break
    print(f"  ChartQA: {len(items)} imágenes únicas cargadas")
    return items


# ── Pipeline principal ────────────────────────────────────────────────────────

def analizar_todo():
    todos = []

    print("Cargando datos locales...")
    for label, carpeta in CARPETAS.items():
        if not Path(carpeta).exists():
            print(f"  ⚠️  Carpeta no encontrada: {carpeta}")
            continue
        items = cargar_archivos_carpeta(label, carpeta)
        print(f"  {label}: {len(items)} archivos")
        todos.extend(items)

    print("Cargando ChartQA...")
    todos.extend(cargar_chartqa())

    total = len(todos)
    print(f"\nTotal a procesar: {total}")
    print(f"Modelo: {MODEL_ID}\n")

    campos = [
        "indice", "label_real", "prediccion", "confianza", "descripcion", "correcto", "source",
        "total_ms", "prompt_ms", "eval_ms", "prompt_tokens", "generated_tokens", "tokens_per_sec"
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()

        for i, item in enumerate(todos):
            label_real = item["label"]
            source     = item["source"]
            nombre     = str(item.get("path", f"chartqa_{i}"))
            print(f"[{i+1}/{total}] {Path(nombre).name} | Label: {label_real}")

            try:
                pil_image = (
                    item["image"].convert("RGB")
                    if source == "chartqa"
                    else cargar_como_imagen(item["path"])
                )
                if pil_image is None:
                    raise ValueError("No se pudo cargar la imagen")

                raw, metricas     = analizar_imagen(pil_image)
                pred, conf, desc  = parsear_respuesta(raw)
                correcto          = 1 if pred == label_real else 0

            except Exception as e:
                print(f"  ⚠️  Error: {e}")
                pred, conf, desc, correcto = "ERROR", 0, str(e), 0
                metricas = {k: 0 for k in ["total_ms","prompt_ms","eval_ms",
                                            "prompt_tokens","generated_tokens","tokens_per_sec"]}

            print(f"  → {pred} | correcto: {correcto}")

            writer.writerow({
                "indice":           i + 1,
                "label_real":       label_real,
                "prediccion":       pred,
                "confianza":        conf,
                "descripcion":      desc,
                "correcto":         correcto,
                "source":           source,
                "total_ms":         metricas["total_ms"],
                "prompt_ms":        metricas["prompt_ms"],
                "eval_ms":          metricas["eval_ms"],
                "prompt_tokens":    metricas["prompt_tokens"],
                "generated_tokens": metricas["generated_tokens"],
                "tokens_per_sec":   metricas["tokens_per_sec"],
            })
            f.flush()

    # ── Métricas finales ──────────────────────────────────────────────────────
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix

    df       = pd.read_csv(OUTPUT_CSV)
    df_valid = df[df["prediccion"] != "ERROR"]

    print(f"\n{'='*55}")
    print(f"Modelo  : {MODEL_ID}")
    print(f"Total   : {total} | Errores: {len(df) - len(df_valid)}")
    print(f"Accuracy: {df_valid['correcto'].mean():.2%}")
    print(f"\n{classification_report(df_valid['label_real'], df_valid['prediccion'])}")

    print("Matriz de confusión:")
    labels = sorted(df_valid["label_real"].unique())
    cm     = confusion_matrix(df_valid["label_real"], df_valid["prediccion"], labels=labels)
    cm_df  = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)
    print(f"\nCSV guardado en: {OUTPUT_CSV}")


if __name__ == "__main__":
    analizar_todo()