import os
import cv2
import math
import time
import csv
import json
import re
import base64
import io
import random
from pathlib import Path
from PIL import Image, ImageDraw
import ollama

# ── Configuración ────────────────────────────────────────────────────────────

CARPETA_NORMAL = r"D:\Normal-002"
CARPETA_CRASH  = r"D:\videos-20260603T082436Z-3-001\videos\Crash-1500"
MAX_POR_CATEGORIA = 1000

# ── Utilidades ───────────────────────────────────────────────────────────────

def pil_to_base64(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


# ── Extracción de frames desde vídeo ─────────────────────────────────────────

def extraer_frames_video(video_path, n=9):
    """
    Abre un vídeo MP4 y extrae n frames con más peso en el último 50%
    (donde suelen ocurrir los accidentes).
    Devuelve lista de PIL Images redimensionadas a 224x168.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el vídeo: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Vídeo sin frames detectables: {video_path}")

    # Calcular índices con más peso en la segunda mitad
    n_ini = max(1, int(n * 0.3))
    n_fin = n - n_ini
    idx_ini = [int(total_frames * i / (n_ini * 3)) for i in range(n_ini)]
    idx_fin = [int(total_frames * (0.5 + 0.5 * i / n_fin)) for i in range(n_fin)]
    indices = sorted(set(idx_ini + idx_fin))
    indices = [min(idx, total_frames - 1) for idx in indices]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img   = Image.fromarray(frame_rgb).resize((224, 168))
            frames.append(pil_img)

    cap.release()

    if not frames:
        raise ValueError(f"No se extrajeron frames del vídeo: {video_path}")

    return frames


# ── Grid de frames ────────────────────────────────────────────────────────────

def frames_a_grid(frames, cols=3):
    n    = len(frames)
    rows = math.ceil(n / cols)
    w, h = frames[0].size
    grid = Image.new("RGB", (w * cols, h * rows), color=(0, 0, 0))
    for i, frame in enumerate(frames):
        frame = frame.copy()
        draw  = ImageDraw.Draw(frame)
        draw.rectangle([(0, 0), (55, 20)], fill=(0, 0, 0))
        draw.text((4, 4), f"t={i+1}/{n}", fill=(255, 255, 0))
        row, col = divmod(i, cols)
        grid.paste(frame, (col * w, row * h))
    return grid


# ── Prompt e inferencia ───────────────────────────────────────────────────────

def analizar_grid_accidente(grid_image, model):
    response = ollama.chat(
        model=model,
        options={"temperature": 0.1, "num_predict": 8096, "num_ctx": 16384, "think": False},
        messages=[{
            "role": "user",
            "content": """
            # ROLE
            You are a traffic safety analyst reviewing dashcam footage.

            # CONTEXT
            You are looking at a grid of frames in chronological order (t=1 earliest).
            Analyze the PROGRESSION of events to detect a traffic accident.

            # SIGNS OF A CRASH
            - Sudden impact or collision between vehicles
            - Vehicle leaving the road or rolling over
            - Airbag deployment or broken glass
            - Abrupt stop or erratic movement between frames
            - Debris or damage visible in later frames
            - Vehicles too close to each other suddenly appearing in the same frame

            # OUTPUT FORMAT
            Respond ONLY with a valid JSON object:
            {
                "predicted_category": "crash | no_crash",
                "confidence": 0.00,
                "description": "Short, technical explanation of the physics/event observed."
            }
            """,
            "images": [pil_to_base64(grid_image)]
        }]
    )
    print(f"  Respuesta cruda: {response['message']['content'][:120]}...")

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
    return response["message"]["content"], metricas


# ── Parser ────────────────────────────────────────────────────────────────────

def parsear_respuesta(raw):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<thinking>.*?</thinking>", "", raw, flags=re.DOTALL)
    raw = raw.strip()
    try:
        clean = raw
        if "```" in clean:
            clean = clean.split("```")[1].removeprefix("json").strip()
        data = json.loads(clean)
        return (
            str(data.get("predicted_category", "ERROR")).lower().strip(),
            data.get("confidence", 0),
            data.get("description", "")
        )
    except json.JSONDecodeError:
        match = re.search(r'"predicted_category"\s*:\s*"([^"]+)"', raw, re.IGNORECASE)
        if match:
            return match.group(1).lower().strip(), 0, raw
        return "ERROR", 0, raw


# ── Recolección de vídeos con tope ────────────────────────────────────────────

def recoger_videos(carpeta_normal, carpeta_crash, max_por_cat):
    """
    Devuelve lista de tuplas (video_path, label_real) balanceada y aleatoria.
    """
    def listar_mp4(carpeta):
        return sorted(Path(carpeta).rglob("*.mp4"))

    videos_normal = listar_mp4(carpeta_normal)
    videos_crash  = listar_mp4(carpeta_crash)

    random.seed(42)
    random.shuffle(videos_normal)
    random.shuffle(videos_crash)

    videos_normal = videos_normal[:max_por_cat]
    videos_crash  = videos_crash[:max_por_cat]

    print(f"Vídeos normal:  {len(videos_normal)} (de {len(listar_mp4(carpeta_normal))} disponibles)")
    print(f"Vídeos crash:   {len(videos_crash)}  (de {len(listar_mp4(carpeta_crash))} disponibles)")

    dataset = [(p, "no_crash") for p in videos_normal] + [(p, "crash") for p in videos_crash]
    random.shuffle(dataset)
    return dataset


# ── Pipeline principal ────────────────────────────────────────────────────────

def analizar_ccd_videos(
    carpeta_normal=CARPETA_NORMAL,
    carpeta_crash=CARPETA_CRASH,
    max_por_cat=MAX_POR_CATEGORIA,
    output_csv="resultados_ccd_videos.csv",
    model="llava:7b",
    n_frames=9,
    sleep_entre_videos=2,
):
    dataset = recoger_videos(carpeta_normal, carpeta_crash, max_por_cat)
    total   = len(dataset)
    print(f"\nTotal vídeos a analizar: {total}  |  Modelo: {model}\n")

    campos = [
        "indice", "video_id", "label_real", "prediccion", "confianza", "descripcion",
        "correcto", "total_ms", "prompt_ms", "eval_ms",
        "prompt_tokens", "generated_tokens", "tokens_per_sec"
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()

        for i, (video_path, label_real) in enumerate(dataset):
            video_id = Path(video_path).stem
            print(f"[{i+1}/{total}] {video_id} | Label: {label_real}")

            prediccion, confianza, descripcion, correcto = "ERROR", 0, "", 0
            metricas = {k: 0 for k in ["total_ms", "prompt_ms", "eval_ms",
                                        "prompt_tokens", "generated_tokens", "tokens_per_sec"]}
            try:
                frames                             = extraer_frames_video(video_path, n=n_frames)
                grid                               = frames_a_grid(frames, cols=3)
                raw, metricas                      = analizar_grid_accidente(grid, model=model)
                prediccion, confianza, descripcion = parsear_respuesta(raw)
                correcto                           = 1 if prediccion == label_real else 0
                time.sleep(sleep_entre_videos)

            except Exception as e:
                print(f"  ⚠️  Error: {e}")

            print(f"  → Predicción: {prediccion} | Correcto: {bool(correcto)}")

            writer.writerow({
                "indice":           i + 1,
                "video_id":         video_id,
                "label_real":       label_real,
                "prediccion":       prediccion,
                "confianza":        confianza,
                "descripcion":      descripcion,
                "correcto":         correcto,
                "total_ms":         metricas["total_ms"],
                "prompt_ms":        metricas["prompt_ms"],
                "eval_ms":          metricas["eval_ms"],
                "prompt_tokens":    metricas["prompt_tokens"],
                "generated_tokens": metricas["generated_tokens"],
                "tokens_per_sec":   metricas["tokens_per_sec"],
            })
            f.flush()

    print(f"\n✅ Resultados guardados en: {output_csv}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    modelos = [
        #"llava:7b",
        #"internlm/interns1:mini-q8_0",
        "qwen3.5:latest"
    ]
    for model in modelos:
        model_name_safe = model.replace(":", "_").replace("/", "_").replace("\\", "_")
        analizar_ccd_videos(
            output_csv=f"resultados_ccd_{model_name_safe}.csv",
            model=model,
        )