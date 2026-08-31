import os

import ollama
import base64
import csv
import cv2
import io
import json
import math
import re
import tempfile
from datasets import load_dataset, Video
from PIL import Image, ImageDraw
import pandas as pd
import requests
from sklearn.metrics import classification_report 
from pathlib import Path

HF_TOKEN = "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"
# ── Utilidades ────────────────────────────────────────────────────────────────

def pil_to_base64(pil_image):
    """Convierte una imagen PIL a una cadena base64 para enviar al modelo."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


import av

def extraer_frames(video_path, n_frames=6):
    container = av.open(video_path)
    stream    = container.streams.video[0]
    total     = stream.frames
    
    if total == 0:
        # si no conoce el total, lee todos y samplea
        frames_raw = [f.to_image() for f in container.decode(video=0)]
    else:
        indices = set(int(total * i / n_frames) for i in range(n_frames))
        frames_raw = []
        for idx, frame in enumerate(container.decode(video=0)):
            if idx in indices:
                frames_raw.append(frame.to_image())
            if len(frames_raw) >= n_frames:
                break

    container.close()

    # Samplea n_frames uniformemente si hay más
    if len(frames_raw) > n_frames:
        step = len(frames_raw) / n_frames
        frames_raw = [frames_raw[int(i * step)] for i in range(n_frames)]

    return [f.resize((320, 240)) for f in frames_raw]


def frames_a_grid(frames, cols=3):
    """Une los frames en una cuadrícula con etiqueta de tiempo."""
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


# ── Modelo ────────────────────────────────────────────────────────────────────

def analizar_grid(grid_image, model="llava:7b"):
    response = ollama.chat(
        model=model,
        options={"temperature": 0.1, "num_predict": 4096, "think": False}, 
        messages=[{
            "role": "user",
            "content": """
            # ROLE
            You are a security analyst reviewing surveillance footage.
         
            # CONTEXT
            You are looking at a grid of frames from a video in chronological
            order (t=1 is earliest). Analyze the PROGRESSION of events across
            frames, not just a single moment.

            # SIGNS OF VIOLENCE
            - People hitting, pushing, or grabbing each other
            - Aggressive or erratic movements between frames
            - People falling or being knocked down
            - Crowd suddenly gathering around a conflict

            # OUTPUT FORMAT
            Respond ONLY with valid JSON:
            {
                "predicted_category": "fight | non-fight",
                "confidence": 0.0,
                "description": "one sentence describing what happens across frames"
            }
            """,
            "images": [pil_to_base64(grid_image)]
        }]
    )

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
    return response["message"]["content"], metricas


# ── Parser ────────────────────────────────────────────────────────────────────

def parsear_respuesta(raw):
    print(f"Respuesta cruda del modelo:\n{raw}\n")
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    try:
        clean = raw.strip()
        if "```" in clean:
            clean = clean.split("```")[1].removeprefix("json").strip()
        data = json.loads(clean)
        return (
            str(data.get("predicted_category", "ERROR")).lower().strip(),
            data.get("confidence", 0),
            data.get("description", "")
        )
    except json.JSONDecodeError:
        # intenta extraer predicted_category con regex como fallback
        match = re.search(r'"predicted_category"\s*:\s*"([^"]+)"', raw, re.IGNORECASE)
        if match:
            return match.group(1).lower().strip(), 0, raw
        return "ERROR", 0, raw

from huggingface_hub import snapshot_download

def descargar_rwf2000(destino="C:/Users/adaxi/OneDrive/Escritorio/dataset/rwf2000"):
    snapshot_download(
        repo_id="DanJoshua/RWF-2000",
        repo_type="dataset",
        local_dir=destino,
    )
    print(f"✅ Dataset descargado en {destino}")


def analizar_rwf2000_local(carpeta="C:/Users/adaxi/OneDrive/Escritorio\dataset/rwf2000/RWF-2000/RWF-2000/train", n_frames=6, output_csv="resultados_rwf_llava.csv", model="llava:7b"):
    # RWF2000 organiza los vídeos en carpetas fight/non-fight
    videos_validos = []
    for label, nombre_carpeta in [("fight", "Fight"), ("non-fight", "NonFight")]:
        for video in Path(carpeta).rglob(f"{nombre_carpeta}/*.avi"):
            videos_validos.append((video, label))
        # por si usa mp4
        for video in Path(carpeta).rglob(f"{nombre_carpeta}/*.mp4"):
            videos_validos.append((video, label))

    print(f"Total vídeos encontrados: {len(videos_validos)}")

    campos = ["indice", "archivo", "label_real", "prediccion", "confianza", "descripcion", "correcto", "total_ms", "prompt_ms", "eval_ms", "prompt_tokens", "generated_tokens", "tokens_per_sec"]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()

        for i, (video_path, label_real) in enumerate(videos_validos):
            print(f"\n[{i+1}/{len(videos_validos)}] {video_path.name} | Label: {label_real}")

            try:
                frames = extraer_frames(str(video_path), n_frames=n_frames)
                if not frames:
                    raise ValueError("No se pudieron extraer frames")

                grid                               = frames_a_grid(frames)
                raw, metricas                      = analizar_grid(grid, model=model)
                prediccion, confianza, descripcion = parsear_respuesta(raw)
                correcto                           = 1 if prediccion == label_real else 0

            except Exception as e:
                print(f"  ⚠️  Error: {e}")
                prediccion, confianza, descripcion, correcto = "ERROR", 0, "", 0

            print(f"  Predicción: {prediccion} | Correcto: {correcto}")

            writer.writerow({
                "indice":      i + 1,
                "archivo":     video_path.name,
                "label_real":  label_real,
                "prediccion":  prediccion,
                "confianza":   confianza,
                "descripcion": descripcion,
                "correcto":    correcto,
                "total_ms":           metricas.get("total_ms", 0),
                "prompt_ms":          metricas.get("prompt_ms", 0),
                "eval_ms":            metricas.get("eval_ms", 0),
                "prompt_tokens":      metricas.get("prompt_tokens", 0),
                "generated_tokens":   metricas.get("generated_tokens", 0),
                "tokens_per_sec":     metricas.get("tokens_per_sec", 0)
            })
            f.flush()

    import pandas as pd
    from sklearn.metrics import classification_report
    df = pd.read_csv(output_csv)
    print(f"\n✅ Accuracy: {df['correcto'].mean():.2%}")
    print(classification_report(df["label_real"], df["prediccion"]))

    
if __name__ == "__main__":
    modelos = [
        "llava:7b",
        "internlm/interns1:mini-q8_0",
        "qwen3.5:latest"
    ]

    for model in modelos:
        model_name_safe = model.replace(":", "_").replace("/", "_").replace("\\", "_")
        analizar_rwf2000_local(n_frames=30, output_csv=f"resultados_rwf_{model_name_safe
        }.csv", model=model)
    