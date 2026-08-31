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

# ── Configuración ─────────────────────────────────────────────────────────────

CARPETA_NORMAL = r"D:\Normal-002"
CARPETA_CRASH  = r"D:\videos-20260603T082436Z-3-001\videos\Crash-1500"
MAX_POR_CATEGORIA = 1000

VALID_CATEGORIES = {"accidente", "no_accidente"}

# ── Utilidades ────────────────────────────────────────────────────────────────

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

    # Más peso en la segunda mitad (donde ocurren los accidentes)
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
        options={"temperature": 0.1, "num_predict": 1024, "num_ctx": 16384, "think": False},
        messages=[{
            "role": "user",
            "content": """
            # ROL
            Eres un analista de seguridad vial revisando grabaciones de dashcam.

            # CONTEXTO
            Estás viendo una cuadrícula de fotogramas en orden cronológico (t=1 es el más antiguo).
            Analiza la PROGRESIÓN de los eventos para detectar un accidente de tráfico.

            # SEÑALES DE UN ACCIDENTE
            - Impacto o colisión repentina entre vehículos
            - Vehículo que sale de la carretera o vuelca
            - Despliegue de airbag o cristales rotos
            - Parada brusca o movimiento errático entre fotogramas
            - Escombros o daños visibles en fotogramas posteriores
            - Vehículos demasiado cerca que aparecen repentinamente en el mismo fotograma

            # FORMATO DE SALIDA
            Responde ÚNICAMENTE con un objeto JSON válido, sin texto adicional, sin markdown:
            {
                "categoria_predicha": "escribe aquí solo una palabra de la lista: accidente, no_accidente",
                "confianza": 0.00,
                "descripcion": "Explicación técnica y breve de la física/evento observado."
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


# ── Parser robusto ────────────────────────────────────────────────────────────

def extraer_json_balanceado(texto):
    """Devuelve el primer objeto JSON con llaves balanceadas, o None si no cierra (truncado)."""
    inicio = texto.find("{")
    if inicio == -1:
        return None
    profundidad = 0
    for i in range(inicio, len(texto)):
        if texto[i] == "{":
            profundidad += 1
        elif texto[i] == "}":
            profundidad -= 1
            if profundidad == 0:
                return texto[inicio:i + 1]
    return None  # nunca cerró → probablemente truncado por num_predict


def _normalizar_categoria(texto: str) -> str:
    """Mapea variaciones comunes a una categoría válida."""
    texto = texto.lower()
    # Inglés → español
    if "crash" in texto or "accident" in texto or "collision" in texto:
        return "accidente"
    if "no_crash" in texto or "no crash" in texto or "no accident" in texto:
        return "no_accidente"
    # Español
    if "no_accidente" in texto or "no accidente" in texto or "sin accidente" in texto:
        return "no_accidente"
    if "accidente" in texto or "colisión" in texto or "choque" in texto:
        return "accidente"
    return "unknown"


def _inferir_desde_texto(texto: str) -> str | None:
    """Intenta inferir la categoría leyendo el texto libre del modelo."""
    t = texto.lower()
    tiene_accidente = any(w in t for w in [
        "accidente", "colisión", "choque", "impacto", "volcamiento",
        "crash", "collision", "accident", "impact"
    ])
    sin_accidente = any(w in t for w in [
        "no accidente", "no hay accidente", "no se detecta", "tráfico normal",
        "no crash", "no accident", "normal traffic", "no collision"
    ])

    if sin_accidente and not tiene_accidente:
        return "no_accidente"
    if tiene_accidente:
        return "accidente"
    return None


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parsear_respuesta(raw):
    """
    Extrae categoria_predicha, confianza y descripcion de la respuesta del modelo.
    Estrategias en orden:
      1. Eliminar bloques <think>...</think> y <thinking>...</thinking>
      2. Manejar </think> huérfano (modelos tipo Qwen/Intern)
      3. Extraer JSON de fences ```...```
      4. Extraer JSON con llaves balanceadas (soporta texto alrededor)
      5. Fallback campo a campo por regex (JSON truncado)
      6. Inferir categoría desde texto libre
      7. ERROR con preview del texto crudo
    """
    print(f"  Respuesta cruda del modelo:\n  {raw}\n")

    # 1. Eliminar bloques <think> y <thinking> completos
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<thinking>.*?</thinking>", "", raw, flags=re.DOTALL)

    # 2. </think> huérfano → descartar razonamiento anterior
    if "</think>" in raw:
        raw = raw.split("</think>")[-1]

    raw = raw.strip()

    # 3. Fences ```...```
    if "```" in raw:
        for parte in raw.split("```"):
            if "categoria_predicha" in parte:
                raw = parte.removeprefix("json").strip()
                break

    # 4. JSON con llaves balanceadas
    json_str = extraer_json_balanceado(raw)
    if json_str:
        try:
            data = json.loads(json_str)
            categoria = str(data.get("categoria_predicha", "unknown")).lower().strip()
            if categoria not in VALID_CATEGORIES:
                categoria = _normalizar_categoria(categoria)
            confianza   = _safe_float(data.get("confianza", 0.0))
            descripcion = str(data.get("descripcion", "")).replace("\n", " ").strip()
            return categoria, confianza, descripcion
        except json.JSONDecodeError:
            pass

    # 5. Fallback campo a campo (JSON truncado)
    cat_match  = re.search(r'"categoria_predicha"\s*:\s*"([^"]+)"', raw, re.IGNORECASE)
    conf_match = re.search(r'"confianza"\s*:\s*([0-9.]+)', raw, re.IGNORECASE)
    desc_match = re.search(r'"descripcion"\s*:\s*"([^"]*)', raw, re.IGNORECASE)  # sin exigir comilla de cierre

    if cat_match:
        categoria   = _normalizar_categoria(cat_match.group(1).lower().strip())
        confianza   = _safe_float(conf_match.group(1)) if conf_match else 0.0
        descripcion = desc_match.group(1).strip() if desc_match else "(truncado)"
        return categoria, confianza, descripcion

    # 6. Inferir desde texto libre
    inferida = _inferir_desde_texto(raw)
    if inferida:
        return inferida, 0.0, raw.replace("\n", " ")[:300]

    # 7. No hay nada usable
    return "ERROR", 0.0, raw.replace("\n", " ")[:300]


# ── Recolección de vídeos con tope ────────────────────────────────────────────

def recoger_videos(carpeta_normal, carpeta_crash, max_por_cat):
    """Devuelve lista de tuplas (video_path, label_real) balanceada y aleatoria."""
    def listar_mp4(carpeta):
        return sorted(Path(carpeta).rglob("*.mp4"))

    videos_normal = listar_mp4(carpeta_normal)
    videos_crash  = listar_mp4(carpeta_crash)

    random.seed(42)
    random.shuffle(videos_normal)
    random.shuffle(videos_crash)

    videos_normal = videos_normal[:max_por_cat]
    videos_crash  = videos_crash[:max_por_cat]

    print(f"Vídeos normal  : {len(videos_normal)} (de {len(listar_mp4(carpeta_normal))} disponibles)")
    print(f"Vídeos accidente: {len(videos_crash)}  (de {len(listar_mp4(carpeta_crash))} disponibles)")

    dataset = [(p, "no_accidente") for p in videos_normal] + [(p, "accidente") for p in videos_crash]
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

        stats = {"ok": 0, "unknown": 0, "error": 0}

        for i, (video_path, label_real) in enumerate(dataset):
            video_id = Path(video_path).stem
            print(f"[{i+1}/{total}] {video_id} | Label: {label_real}")

            prediccion, confianza, descripcion, correcto = "ERROR", 0.0, "", 0
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
                stats["error"] += 1

            else:
                if prediccion in ("ERROR", "unknown"):
                    stats["unknown"] += 1
                else:
                    stats["ok"] += 1

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

    # Resumen final
    total_stats = sum(stats.values())
    print(f"\n{'='*50}")
    print(f"  RESUMEN FINAL")
    print(f"{'='*50}")
    print(f"  Total procesados : {total_stats}")
    print(f"  Correctos        : {stats['ok']}  ({stats['ok']/max(total_stats,1)*100:.1f}%)")
    print(f"  Unknown/Error    : {stats['unknown'] + stats['error']}  "
          f"({(stats['unknown'] + stats['error'])/max(total_stats,1)*100:.1f}%)")
    print(f"\n  Resultados guardados en: {output_csv}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    modelos = [
        "llava:7b",
        "internlm/interns1:mini-q8_0",
        "qwen3.5:latest",
    ]
    for model in modelos:
        model_name_safe = model.replace(":", "_").replace("/", "_").replace("\\", "_")
        analizar_ccd_videos(
            output_csv=f"resultados_ccd_{model_name_safe}.csv",
            model=model,
        )