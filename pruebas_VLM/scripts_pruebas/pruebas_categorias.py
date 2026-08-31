import ollama
import base64
import csv
import cv2
import io
import json
import math
import os
import re
import requests
from pathlib import Path
from PIL import Image, ImageDraw
from datasets import load_dataset

# ── Configuración ─────────────────────────────────────────────────────────────

MODEL = "qwen3.5:latest"  

CARPETAS = {
    "violence":  r"D:\DatasetTriaje\Violencia",
    "weapons":   r"D:\DatasetTriaje\Armas",
    "fire":      r"D:\DatasetTriaje\Fuego",
    "accident":  r"D:\DatasetTriaje\Coches",
    "normal":    r"D:\DatasetTriaje\Normal",
}

EXTENSIONES_IMG   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXTENSIONES_VIDEO = {".mp4", ".avi", ".mov", ".mkv"}
MAX_CHARTS        = 250


# ── Utilidades ────────────────────────────────────────────────────────────────

def pil_to_base64(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def extraer_frames_video(video_path, n_frames=9):
    """Extrae n frames con más peso en la segunda mitad."""
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

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((224, 168)))
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
    """Devuelve una PIL Image independientemente de si es imagen o vídeo."""
    ext = Path(file_path).suffix.lower()
    if ext in EXTENSIONES_IMG:
        return Image.open(file_path).convert("RGB")
    elif ext in EXTENSIONES_VIDEO:
        frames = extraer_frames_video(file_path)
        if not frames:
            return None
        return frames_a_grid(frames) if len(frames) > 1 else frames[0]
    return None


# ── Modelo ────────────────────────────────────────────────────────────────────

def analizar_imagen(pil_image):
    response = ollama.chat(
        model=MODEL,
        options={"temperature": 0.1, "num_predict": 4096, "num_ctx": 4096},
        messages=[{
            "role": "user",
            "content": """
                You are a visual safety analyst. Analyze this image or video sequence and estimate the probability of each safety category.

                Possible categories:

                - violence: physical fight, assault, aggression between people
                - weapons: visible knife, gun, firearm, or dangerous weapon
                - fire: flames, smoke, burning objects or buildings
                - accident: traffic collision, car crash, road accident, cars too close, damaged vehicles, cars off the road
                - chart: ANY graph, chart, table, infographic, or news article with data
                - normal: none of the above, everyday safe content

                Rules:
                1. Evaluate ALL categories and assign a confidence score between 0.00 and 1.00 to each one.
                2. The sum of all confidence scores does NOT need to be exactly 1.00, since multiple categories can be present at the same time.
                3. A single image or video can contain multiple safety categories (for example: accident + fire + weapons).
                4. If no safety category is detected, "normal" should have the highest confidence score.
                5. If you see ANY chart, graph, table, infographic, or data visualization, assign a high confidence score to "chart".
                6. Sort categories from highest to lowest confidence.
                7. Do not include explanations outside the JSON.

                Respond ONLY with valid JSON:

                {
                    "predicted_categories": [
                        {
                            "category": "fire",
                            "confidence": 0.92
                        },
                        {
                            "category": "violence",
                            "confidence": 0.10
                        },
                        {
                            "category": "weapons",
                            "confidence": 0.05
                        },
                        {
                            "category": "accident",
                            "confidence": 0.20
                        },
                        {
                            "category": "chart",
                            "confidence": 0.00
                        },
                        {
                            "category": "normal",
                            "confidence": 0.03
                        }
                    ],
                    "description": "One sentence explaining the reasoning behind the prediction."
                }

                """,
            "images": [pil_to_base64(pil_image)]
        }]
    )
    raw = response["message"]["content"]
    raw = re.sub(r"<tool_call>.*?</tool_call>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<thinking>.*?</thinking>", "", raw, flags=re.DOTALL)

    metricas = {
        "total_ms":        round(response.get("total_duration", 0) / 1e6, 1),
        "load_ms":         round(response.get("load_duration", 0) / 1e6, 1),
        "prompt_ms":       round(response.get("prompt_eval_duration", 0) / 1e6, 1),
        "eval_ms":         round(response.get("eval_duration", 0) / 1e6, 1),
        "prompt_tokens":   response.get("prompt_eval_count", 0),
        "generated_tokens":response.get("eval_count", 0),
        "tokens_per_sec":  round(
            response.get("eval_count", 0) / max(response.get("eval_duration", 1) / 1e9, 1e-9), 1
        ),
    }

    return raw.strip(), metricas

# ── Parser ────────────────────────────────────────────────────────────────────

CATEGORIAS_VALIDAS = {"violence", "weapons", "fire", "accident", "chart", "normal"}

def parsear_respuesta(raw):
    """
    Parsea la respuesta del modelo con el nuevo esquema 'predicted_categories'
    (lista de {category, confidence} ordenada de mayor a menor).
    Devuelve: (prediccion_top, confianza_top, descripcion)
    """
    try:
        clean = raw.strip()
        if "```" in clean:
            partes = clean.split("```")
            clean = partes[1].removeprefix("json").strip() if len(partes) > 1 else clean

        data = json.loads(clean)
        categorias = data.get("predicted_categories", [])

        if not categorias:
            return "ERROR", 0, data.get("description", raw)

        # Normaliza cada categoría y coge la de mayor confianza
        normalizadas = []
        for c in categorias:
            cat = str(c.get("category", "")).lower().strip()
            cat = cat.replace("no_crash", "normal").replace("crash", "accident")
            cat = cat.replace("fight", "violence").replace("smoke", "fire")
            conf = float(c.get("confidence", 0) or 0)
            if cat in CATEGORIAS_VALIDAS:
                normalizadas.append((cat, conf))

        if not normalizadas:
            return "ERROR", 0, data.get("description", raw)

        pred, confianza = max(normalizadas, key=lambda x: x[1])
        return pred, confianza, data.get("description", "")

    except (json.JSONDecodeError, ValueError, AttributeError, IndexError):
        # Fallback: intenta extraer el bloque JSON con balanceo de llaves
        try:
            inicio = raw.index("{")
            profundidad = 0
            for i, ch in enumerate(raw[inicio:], start=inicio):
                if ch == "{":
                    profundidad += 1
                elif ch == "}":
                    profundidad -= 1
                    if profundidad == 0:
                        bloque = raw[inicio:i + 1]
                        return parsear_respuesta(bloque)
        except (ValueError, RecursionError):
            pass

        # Último recurso: regex para pillar al menos la categoría con mayor confianza mencionada
        matches = re.findall(
            r'"category"\s*:\s*"([^"]+)"\s*,\s*"confidence"\s*:\s*([0-9.]+)',
            raw, re.IGNORECASE
        )
        if matches:
            normalizadas = []
            for cat, conf in matches:
                cat = cat.lower().strip()
                cat = cat.replace("no_crash", "normal").replace("crash", "accident")
                cat = cat.replace("fight", "violence").replace("smoke", "fire")
                if cat in CATEGORIAS_VALIDAS:
                    normalizadas.append((cat, float(conf)))
            if normalizadas:
                pred, confianza = max(normalizadas, key=lambda x: x[1])
                return pred, confianza, raw

        return "ERROR", 0, raw


# ── Fuentes de datos ──────────────────────────────────────────────────────────

def cargar_archivos_carpeta(label, carpeta):
    """Carga todos los archivos multimedia de una carpeta."""
    items = []
    for f in Path(carpeta).rglob("*"):
        if f.suffix.lower() in EXTENSIONES_IMG | EXTENSIONES_VIDEO:
            items.append({"label": label, "path": f, "source": "local"})
    return items


def cargar_chartqa(max_items=MAX_CHARTS):
    """Carga imágenes de ChartQA desde HuggingFace."""
    print(f"  Cargando ChartQA (máx {max_items} imágenes)...")
    dataset = load_dataset("HuggingFaceM4/ChartQA", split="test", streaming=True)
    
    vistos   = set()
    items    = []
    for ejemplo in dataset:
        img_id = ejemplo.get("imgname", str(len(items)))
        if img_id in vistos:
            continue
        vistos.add(img_id)
        items.append({"label": "chart", "image": ejemplo["image"], "source": "chartqa"})
        if len(items) >= max_items:
            break
    print(f"  ChartQA: {len(items)} imágenes únicas cargadas")
    return items


# ── Pipeline principal ────────────────────────────────────────────────────────

def analizar_todo(output_csv=None):
    if output_csv is None:
        safe_model = MODEL.replace(":", "_").replace("/", "_")
        output_csv = f"resultados_2_{safe_model}.csv"

    # Recopila todos los items
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
    todos.extend(cargar_chartqa(max_items=MAX_CHARTS))

    total = len(todos)
    print(f"\nTotal a procesar: {total}")

    campos = [
    "indice", "label_real", "prediccion", "confianza", "descripcion", "correcto", "source",
    "total_ms", "prompt_ms", "eval_ms", "prompt_tokens", "generated_tokens", "tokens_per_sec"
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()

        for i, item in enumerate(todos):
            label_real = item["label"]
            source     = item["source"]
            nombre     = str(item.get("path", f"chartqa_{i}"))
            print(f"\n[{i+1}/{total}] {Path(nombre).name} | Label: {label_real}")

            try:
                if source == "chartqa":
                    pil_image = item["image"].convert("RGB")
                else:
                    pil_image = cargar_como_imagen(item["path"])

                if pil_image is None:
                    raise ValueError("No se pudo cargar la imagen")

                raw, metricas = analizar_imagen(pil_image)
                prediccion, confianza, descripcion = parsear_respuesta(raw)
                correcto = 1 if prediccion == label_real else 0

            except Exception as e:
                print(f"  ⚠️  Error: {e}")
                prediccion, confianza, descripcion, correcto = "ERROR", 0, "", 0

            print(f"  Pred: {prediccion} | Correcto: {correcto}")

            writer.writerow({
                "indice":      i + 1,
                "label_real":  label_real,
                "prediccion":  prediccion,
                "confianza":   confianza,
                "descripcion": descripcion,
                "correcto":    correcto,
                "source":      source,
                "total_ms":    metricas.get("total_ms", 0),
                "prompt_ms":   metricas.get("prompt_ms", 0),
                "eval_ms":     metricas.get("eval_ms", 0),
                "prompt_tokens":   metricas.get("prompt_tokens", 0),
                "generated_tokens":metricas.get("generated_tokens", 0),
                "tokens_per_sec":  metricas.get("tokens_per_sec", 0)
            })
            f.flush()

    # ── Métricas finales ──────────────────────────────────────────────────────
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix

    df = pd.read_csv(output_csv)
    df_valid = df[df["prediccion"] != "ERROR"]

    print(f"\n{'='*50}")
    print(f"Modelo : {MODEL}")
    print(f"Total  : {total} | Errores: {len(df) - len(df_valid)}")
    print(f"Accuracy: {df_valid['correcto'].mean():.2%}")
    print(f"\n{classification_report(df_valid['label_real'], df_valid['prediccion'])}")

    print("\nMatriz de confusión:")
    labels = sorted(df_valid["label_real"].unique())
    cm     = confusion_matrix(df_valid["label_real"], df_valid["prediccion"], labels=labels)
    cm_df  = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)


if __name__ == "__main__":
    analizar_todo()