import os
import cv2
import math
import time
import csv
import json
import re
import random
import torch
from pathlib import Path
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# ── Configuración ─────────────────────────────────────────────────────────────

MODEL_ID   = "BSC-LT/Salamandra-VL-7B-2512"
OUTPUT_CSV = "resultados_ccd_salamandra_es.csv"

HF_TOKEN = os.getenv("HF_TOKEN")

CARPETA_NORMAL    = r"D:\Normal-002"
CARPETA_CRASH     = r"D:\videos-20260603T082436Z-3-001\videos\Crash-1500"
MAX_POR_CATEGORIA = 1000

VALID_CATEGORIES = {"accidente", "no_accidente"}

PROMPT = """Eres un analista de seguridad vial revisando grabaciones de dashcam.

Estás viendo una cuadrícula de fotogramas en orden cronológico (t=1 es el más antiguo). Analiza la PROGRESIÓN de los eventos para detectar un accidente de tráfico.

Señales de un accidente:
1. Impacto o colisión repentina entre vehículos
2. Vehículo que sale de la carretera o vuelca
3. Despliegue de airbag o cristales rotos
4. Parada brusca o movimiento errático entre fotogramas
5. Escombros o daños visibles en fotogramas posteriores
6. Vehículos demasiado cerca que aparecen repentinamente en el mismo fotograma

Reglas:
1. Escribe ÚNICAMENTE una de estas palabras exactas: accidente, no_accidente
2. No se aceptan otros valores

Responde ÚNICAMENTE con JSON válido, sin texto adicional, sin markdown:
{
    "categoria_predicha": "escribe aquí la categoría detectada",
    "confianza": 0.00,
    "descripcion": "Explicación técnica y breve de la física/evento observado."
}"""

# ── Carga del modelo (singleton) ──────────────────────────────────────────────

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
    _processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
    _model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="cuda",
        token=HF_TOKEN,
    )
    _model.eval()
    print("Modelo cargado ✓")


# ── Extracción de frames desde vídeo ─────────────────────────────────────────

def extraer_frames_video(video_path, n=9):
    """Extrae n frames con más peso en el último 50% (donde ocurren los accidentes)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el vídeo: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Vídeo sin frames detectables: {video_path}")

    n_ini   = max(1, int(n * 0.3))
    n_fin   = n - n_ini
    idx_ini = [int(total_frames * i / (n_ini * 3)) for i in range(n_ini)]
    idx_fin = [int(total_frames * (0.5 + 0.5 * i / n_fin)) for i in range(n_fin)]
    indices = sorted(set(idx_ini + idx_fin))
    indices = [min(idx, total_frames - 1) for idx in indices]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((224, 168))
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


def redimensionar_con_padding(img, size=672):
    """Escala manteniendo aspect ratio y rellena con negro hasta size x size."""
    img = img.copy()
    img.thumbnail((size, size), Image.LANCZOS)
    lienzo = Image.new("RGB", (size, size), color=(0, 0, 0))
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    lienzo.paste(img, (x, y))
    return lienzo


# ── Inferencia ────────────────────────────────────────────────────────────────

def analizar_grid_accidente(grid_image):
    cargar_modelo()

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT},
        ]
    }]

    grid_image = redimensionar_con_padding(grid_image.convert("RGB"), size=672)

    text   = _processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = _processor(
        text=text,
        images=[grid_image],
        return_tensors="pt",
    ).to(_model.device, torch.float16)

    t0 = time.time()
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=1.0,
        )
    elapsed_ms = (time.time() - t0) * 1000

    new_ids       = output_ids[0][inputs["input_ids"].shape[-1]:]
    raw           = _processor.decode(new_ids, skip_special_tokens=True)
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


# ── Parser robusto ────────────────────────────────────────────────────────────

def extraer_json_balanceado(texto):
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
    return None


def _normalizar_categoria(texto: str) -> str:
    texto = texto.lower()
    if "no_accidente" in texto or "no accidente" in texto or "sin accidente" in texto \
            or "no_crash" in texto or "no crash" in texto or "no accident" in texto:
        return "no_accidente"
    if "accidente" in texto or "colisión" in texto or "colision" in texto \
            or "choque" in texto or "crash" in texto or "accident" in texto \
            or "collision" in texto:
        return "accidente"
    return "unknown"


def _inferir_desde_texto(texto: str) -> str | None:
    t = texto.lower()
    sin_accidente = any(w in t for w in [
        "no accidente", "no hay accidente", "no se detecta", "tráfico normal",
        "no crash", "no accident", "normal traffic", "no collision",
    ])
    tiene_accidente = any(w in t for w in [
        "accidente", "colisión", "choque", "impacto", "volcamiento",
        "crash", "collision", "accident", "impact",
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
    print(f"  Respuesta cruda del modelo:\n  {raw}\n")

    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<thinking>.*?</thinking>", "", raw, flags=re.DOTALL)
    if "</think>" in raw:
        raw = raw.split("</think>")[-1]
    raw = raw.strip()

    if "```" in raw:
        for parte in raw.split("```"):
            if "categoria_predicha" in parte:
                raw = parte.removeprefix("json").strip()
                break

    json_str = extraer_json_balanceado(raw)
    if json_str:
        try:
            data      = json.loads(json_str)
            categoria = str(data.get("categoria_predicha", "unknown")).lower().strip()
            if categoria not in VALID_CATEGORIES:
                categoria = _normalizar_categoria(categoria)
            return (
                categoria,
                _safe_float(data.get("confianza", 0.0)),
                str(data.get("descripcion", "")).replace("\n", " ").strip(),
            )
        except json.JSONDecodeError:
            pass

    cat_match  = re.search(r'"categoria_predicha"\s*:\s*"([^"]+)"', raw, re.IGNORECASE)
    conf_match = re.search(r'"confianza"\s*:\s*([0-9.]+)', raw, re.IGNORECASE)
    desc_match = re.search(r'"descripcion"\s*:\s*"([^"]*)', raw, re.IGNORECASE)

    if cat_match:
        return (
            _normalizar_categoria(cat_match.group(1).lower().strip()),
            _safe_float(conf_match.group(1)) if conf_match else 0.0,
            desc_match.group(1).strip() if desc_match else "(truncado)",
        )

    inferida = _inferir_desde_texto(raw)
    if inferida:
        return inferida, 0.0, raw.replace("\n", " ")[:300]

    return "ERROR", 0.0, raw.replace("\n", " ")[:300]


# ── Recolección de vídeos con tope ────────────────────────────────────────────

def recoger_videos(carpeta_normal, carpeta_crash, max_por_cat):
    def listar_mp4(carpeta):
        return sorted(Path(carpeta).rglob("*.mp4"))

    videos_normal = listar_mp4(carpeta_normal)
    videos_crash  = listar_mp4(carpeta_crash)

    random.seed(42)
    random.shuffle(videos_normal)
    random.shuffle(videos_crash)

    videos_normal = videos_normal[:max_por_cat]
    videos_crash  = videos_crash[:max_por_cat]

    print(f"Vídeos normal   : {len(videos_normal)} (de {len(listar_mp4(carpeta_normal))} disponibles)")
    print(f"Vídeos accidente: {len(videos_crash)}  (de {len(listar_mp4(carpeta_crash))} disponibles)")

    dataset = [(p, "no_accidente") for p in videos_normal] + [(p, "accidente") for p in videos_crash]
    random.shuffle(dataset)
    return dataset


# ── Pipeline principal ────────────────────────────────────────────────────────

def analizar_ccd_videos(
    carpeta_normal=CARPETA_NORMAL,
    carpeta_crash=CARPETA_CRASH,
    max_por_cat=MAX_POR_CATEGORIA,
    output_csv=OUTPUT_CSV,
    n_frames=9,
):
    cargar_modelo()  # una sola carga al inicio

    dataset = recoger_videos(carpeta_normal, carpeta_crash, max_por_cat)
    total   = len(dataset)
    print(f"\nTotal vídeos a analizar: {total}  |  Modelo: {MODEL_ID}\n")

    campos = [
        "indice", "video_id", "label_real", "prediccion", "confianza", "descripcion",
        "correcto", "total_ms", "prompt_ms", "eval_ms",
        "prompt_tokens", "generated_tokens", "tokens_per_sec",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()

        stats = {"ok": 0, "unknown": 0, "error": 0}

        for i, (video_path, label_real) in enumerate(dataset):
            video_id = Path(video_path).stem
            print(f"[{i+1}/{total}] {video_id} | Label: {label_real}")

            prediccion, confianza, descripcion, correcto = "ERROR", 0.0, "", 0
            metricas = {k: 0 for k in [
                "total_ms", "prompt_ms", "eval_ms",
                "prompt_tokens", "generated_tokens", "tokens_per_sec",
            ]}

            try:
                frames                             = extraer_frames_video(video_path, n=n_frames)
                grid                               = frames_a_grid(frames, cols=3)
                raw, metricas                      = analizar_grid_accidente(grid)
                prediccion, confianza, descripcion = parsear_respuesta(raw)
                correcto                           = 1 if prediccion == label_real else 0

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

    total_stats = sum(stats.values())
    print(f"\n{'='*50}")
    print(f"  RESUMEN FINAL")
    print(f"{'='*50}")
    print(f"  Total procesados : {total_stats}")
    print(f"  Correctos        : {stats['ok']}  ({stats['ok']/max(total_stats,1)*100:.1f}%)")
    print(f"  Unknown/Error    : {stats['unknown'] + stats['error']}  "
          f"({(stats['unknown']+stats['error'])/max(total_stats,1)*100:.1f}%)")
    print(f"\n  Resultados guardados en: {output_csv}")

    import pandas as pd
    from sklearn.metrics import classification_report

    df       = pd.read_csv(output_csv)
    df_valid = df[df["prediccion"] != "ERROR"]
    print(f"\nAccuracy: {df_valid['correcto'].mean():.2%}")
    print(classification_report(df_valid["label_real"], df_valid["prediccion"]))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    analizar_ccd_videos()