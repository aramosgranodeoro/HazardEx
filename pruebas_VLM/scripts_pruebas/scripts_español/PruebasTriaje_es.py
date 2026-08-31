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

MODELS = [ "internlm/interns1:mini-q8_0", "qwen3.5:latest"]

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

def analizar_imagen(pil_image, model):
    response = ollama.chat(
        model=model,
        options={"temperature": 0.1, "num_predict": 4096, "num_ctx": 4096},
        messages=[{
            "role": "user",
            "content": """
            Eres un analista visual de seguridad. Analiza esta imagen o secuencia de vídeo y clasifícala en UNA de estas categorías:

            - violencia:  pelea física, agresión, violencia entre personas
            - armas:      cuchillo, pistola, arma de fuego o arma peligrosa visible
            - fuego:      llamas, humo, objetos o edificios en llamas
            - accidente:  colisión de tráfico, accidente de coche, accidente de carretera, coches demasiado cerca, vehículos dañados, coches fuera de la vía
            - gráfico:    CUALQUIER gráfico, tabla, infografía o artículo de noticias con datos
            - normal:     ninguna de las anteriores, contenido cotidiano y seguro

            Reglas:
            1. Si ves CUALQUIER gráfico, diagrama o visualización de datos → elige siempre "gráfico"
            2. Escribe ÚNICAMENTE una de estas palabras exactas: violencia, armas, fuego, accidente, gráfico, normal
            3. No se aceptan otros valores

            Responde ÚNICAMENTE con JSON válido:
            {
                "predicted_category": "ESCRIBE_AQUI_LA_CATEGORIA",
                "confidence": 0.00,
                "description": "una frase explicando el razonamiento detrás de la predicción en español"
            }
            """,
            "images": [pil_to_base64(pil_image)]
        }]
    )
    raw = response["message"]["content"]
    raw = re.sub(r"<tool_call>.*?</tool_call>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<thinking>.*?</thinking>", "", raw, flags=re.DOTALL)
    # Elimina etiquetas </think> huérfanas (Qwen, InternLM)
    raw = re.sub(r"</think>", "", raw)

    metricas = {
        "total_ms":         round(response.get("total_duration", 0) / 1e6, 1),
        "load_ms":          round(response.get("load_duration", 0) / 1e6, 1),
        "prompt_ms":        round(response.get("prompt_eval_duration", 0) / 1e6, 1),
        "eval_ms":          round(response.get("eval_duration", 0) / 1e6, 1),
        "prompt_tokens":    response.get("prompt_eval_count", 0),
        "generated_tokens": response.get("eval_count", 0),
        "tokens_per_sec":   round(
            response.get("eval_count", 0) / max(response.get("eval_duration", 1) / 1e9, 1e-9), 1
        ),
    }

    return raw.strip(), metricas


# ── Parser ────────────────────────────────────────────────────────────────────

CATEGORIAS_VALIDAS = {"violencia", "armas", "fuego", "accidente", "gráfico", "normal"}

def parsear_respuesta(raw):
    try:
        clean = raw.strip()
        if "```" in clean:
            clean = clean.split("```")[1].removeprefix("json").strip()
        data = json.loads(clean)
        pred = str(data.get("predicted_category", "ERROR")).lower().strip()
        pred = pred.replace("no_crash", "normal").replace("crash", "accidente")
        pred = pred.replace("fight", "violencia").replace("smoke", "fuego")
        pred = pred.replace("fire", "fuego").replace("weapon", "armas")
        pred = pred.replace("chart", "gráfico").replace("graph", "gráfico")
        if pred not in CATEGORIAS_VALIDAS:
            pred = "ERROR"
        return pred, data.get("confidence", 0), data.get("description", "")
    except json.JSONDecodeError:
        match = re.search(r'"predicted_category"\s*:\s*"([^"]+)"', raw, re.IGNORECASE)
        if match:
            return match.group(1).lower().strip(), 0, raw
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

    vistos = set()
    items  = []
    for ejemplo in dataset:
        img_id = ejemplo.get("imgname", str(len(items)))
        if img_id in vistos:
            continue
        vistos.add(img_id)
        items.append({"label": "gráfico", "image": ejemplo["image"], "source": "chartqa"})
        if len(items) >= max_items:
            break
    print(f"  ChartQA: {len(items)} imágenes únicas cargadas")
    return items


# ── Pipeline por modelo ───────────────────────────────────────────────────────

def analizar_con_modelo(model, todos):
    safe_model = model.replace(":", "_").replace("/", "_")
    output_csv = f"resultados_{safe_model}.csv"

    print(f"\n{'='*60}")
    print(f"  Modelo: {model}")
    print(f"  CSV de salida: {output_csv}")
    print(f"{'='*60}")

    total  = len(todos)
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

                raw, metricas = analizar_imagen(pil_image, model)
                prediccion, confianza, descripcion = parsear_respuesta(raw)
                correcto = 1 if prediccion == label_real else 0

            except Exception as e:
                print(f"  ⚠️  Error: {e}")
                prediccion, confianza, descripcion, correcto = "ERROR", 0, str(e), 0
                metricas = {k: 0 for k in ["total_ms", "prompt_ms", "eval_ms",
                                            "prompt_tokens", "generated_tokens", "tokens_per_sec"]}

            print(f"  Pred: {prediccion} | Correcto: {correcto}")

            writer.writerow({
                "indice":           i + 1,
                "label_real":       label_real,
                "prediccion":       prediccion,
                "confianza":        confianza,
                "descripcion":      descripcion,
                "correcto":         correcto,
                "source":           source,
                "total_ms":         metricas.get("total_ms", 0),
                "prompt_ms":        metricas.get("prompt_ms", 0),
                "eval_ms":          metricas.get("eval_ms", 0),
                "prompt_tokens":    metricas.get("prompt_tokens", 0),
                "generated_tokens": metricas.get("generated_tokens", 0),
                "tokens_per_sec":   metricas.get("tokens_per_sec", 0),
            })
            f.flush()

    return output_csv


# ── Resumen por modelo ────────────────────────────────────────────────────────

def imprimir_resumen(model, output_csv):
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix

    df       = pd.read_csv(output_csv)
    df_valid = df[df["prediccion"] != "ERROR"]
    total    = len(df)
    errores  = total - len(df_valid)

    print(f"\n{'='*60}")
    print(f"  RESUMEN — {model}")
    print(f"{'='*60}")
    print(f"  Total  : {total} | Errores de parseo: {errores}")
    print(f"  Accuracy: {df_valid['correcto'].mean():.2%}")
    print(f"\n{classification_report(df_valid['label_real'], df_valid['prediccion'])}")

    print("  Matriz de confusión:")
    labels = sorted(df_valid["label_real"].unique())
    cm     = confusion_matrix(df_valid["label_real"], df_valid["prediccion"], labels=labels)
    cm_df  = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)


# ── Pipeline principal ────────────────────────────────────────────────────────

def analizar_todo():
    # Carga de datos — una sola vez para todos los modelos
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

    print(f"\nTotal de muestras: {len(todos)}")
    print(f"Modelos a evaluar: {MODELS}\n")

    csvs_generados = []

    for model in MODELS:
        output_csv = analizar_con_modelo(model, todos)
        csvs_generados.append((model, output_csv))
        imprimir_resumen(model, output_csv)

    # ── Tabla comparativa final ───────────────────────────────────────────────
    import pandas as pd

    print(f"\n{'='*60}")
    print("  COMPARATIVA FINAL")
    print(f"{'='*60}")
    print(f"  {'Modelo':<35} {'Accuracy':>10} {'Errores':>10}")
    print(f"  {'-'*55}")
    for model, csv_path in csvs_generados:
        df       = pd.read_csv(csv_path)
        df_valid = df[df["prediccion"] != "ERROR"]
        acc      = df_valid["correcto"].mean()
        errores  = len(df) - len(df_valid)
        print(f"  {model:<35} {acc:>10.2%} {errores:>10}")
    print()


if __name__ == "__main__":
    analizar_todo()