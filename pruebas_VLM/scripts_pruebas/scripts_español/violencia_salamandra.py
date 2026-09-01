import csv
import json
import math
import os
import re
import time
from pathlib import Path

import av
import torch
from PIL import Image, ImageDraw
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# ── Configuración ─────────────────────────────────────────────────────────────

MODEL_ID = "BSC-LT/Salamandra-VL-7B-2512"

# El token de HF NO se hardcodea: exporta la variable de entorno HF_TOKEN
# (set HF_TOKEN=... en PowerShell, o $env:HF_TOKEN="...")
HF_TOKEN = os.getenv("HF_TOKEN")

PROMPT = """Eres un analista de seguridad revisando imágenes de videovigilancia.

Estás observando una cuadrícula de fotogramas de un vídeo en orden cronológico (t=1 es el más antiguo). Analiza la PROGRESIÓN de los eventos a través de los fotogramas, no solo un momento aislado.

Señales de violencia:
1. Personas golpeándose, empujándose o agarrándose entre sí.
2. Movimientos agresivos o erráticos entre fotogramas.
3. Personas cayendo o siendo derribadas.
4. Multitud reuniéndose repentinamente alrededor de un conflicto.
Señales de no violencia:
1. Personas caminando, sentadas o interactuando de forma pacífica.
2. Movimientos suaves y naturales entre fotogramas.
3. Ausencia de contacto físico agresivo.

Reglas:
1. Escribe ÚNICAMENTE una palabra de la lista: violencia, no_violencia
2. No se aceptan otros valores

Responde ÚNICAMENTE con JSON válido, sin explicaciones adicionales, sin markdown:
{
    "categoria_predicha": "escribe aquí la categoría detectada",
    "confianza": 0.00,
    "descripcion": "una frase que describa lo que sucede a través de los fotogramas"
}"""

CATEGORIAS_VALIDAS = {"violencia", "no_violencia"}

# ── Carga del modelo (singleton, igual que en el script de Salamandra) ────────

_processor = None
_model = None


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


# ── Extracción de frames y composición de grid (igual que en el script RWF) ──

def extraer_frames(video_path, n_frames=9):
    container = av.open(video_path)
    stream = container.streams.video[0]
    total = stream.frames

    if total == 0:
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

    if len(frames_raw) > n_frames:
        step = len(frames_raw) / n_frames
        frames_raw = [frames_raw[int(i * step)] for i in range(n_frames)]

    return [f.resize((320, 240)) for f in frames_raw]


def redimensionar_con_padding(img, size=672):
    """Escala manteniendo el aspect ratio (sin deformar) y rellena con negro
    hasta dejar un cuadrado size x size, en vez de forzar un resize que
    aplaste la imagen."""
    img = img.copy()
    img.thumbnail((size, size), Image.LANCZOS)
    lienzo = Image.new("RGB", (size, size), color=(0, 0, 0))
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    lienzo.paste(img, (x, y))
    return lienzo


def frames_a_grid(frames, cols=6):
    n = len(frames)
    rows = math.ceil(n / cols)
    w, h = frames[0].size
    grid = Image.new("RGB", (w * cols, h * rows), color=(0, 0, 0))

    for i, frame in enumerate(frames):
        frame = frame.copy()
        draw = ImageDraw.Draw(frame)
        draw.rectangle([(0, 0), (55, 20)], fill=(0, 0, 0))
        draw.text((4, 4), f"t={i+1}/{n}", fill=(255, 255, 0))
        row, col = divmod(i, cols)
        grid.paste(frame, (col * w, row * h))

    return grid


# ── Inferencia con Salamandra (igual patrón que analizar_imagen del script base) ─

def analizar_grid(grid_image):
    cargar_modelo()

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT},
        ]
    }]

    # 672x672 es el tamaño que ya funciona de forma estable con Salamandra.
    # En vez de un resize directo (que aplastaría el grid si no es cuadrado),
    # se escala manteniendo el aspect ratio y se rellena con negro.
    grid_image = redimensionar_con_padding(grid_image.convert("RGB"), size=672)

    text = _processor.apply_chat_template(messages, add_generation_prompt=True)
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

    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    raw = _processor.decode(new_ids, skip_special_tokens=True)

    prompt_tokens = int(inputs["input_ids"].shape[-1])
    gen_tokens = int(new_ids.shape[0])

    metricas = {
        "total_ms": round(elapsed_ms, 1),
        "prompt_ms": 0,
        "eval_ms": round(elapsed_ms, 1),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": gen_tokens,
        "tokens_per_sec": round(gen_tokens / max(elapsed_ms / 1000, 1e-9), 1),
    }
    return raw.strip(), metricas


# ── Parser (robusto a truncado/think, + normalización binaria) ───────────────

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


def _normalizar_categoria(valor):
    v = str(valor).lower().strip()
    v = re.sub(r"\s+", "_", v)
    if "no" in v and "violen" in v:
        return "no_violencia"
    if "violen" in v or "pelea" in v or "fight" in v:
        return "violencia"
    return v if v in CATEGORIAS_VALIDAS else "ERROR"


def parsear_respuesta(raw):
    print(f"Respuesta cruda del modelo:\n{raw}\n")

    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
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
            data = json.loads(json_str)
            return (
                _normalizar_categoria(data.get("categoria_predicha", "ERROR")),
                data.get("confianza", 0),
                data.get("descripcion", ""),
            )
        except json.JSONDecodeError:
            pass

    cat_match = re.search(r'"categoria_predicha"\s*:\s*"([^"]+)"', raw, re.IGNORECASE)
    conf_match = re.search(r'"confianza"\s*:\s*([0-9.]+)', raw, re.IGNORECASE)
    desc_match = re.search(r'"descripcion"\s*:\s*"([^"]*)', raw, re.IGNORECASE)

    if cat_match:
        categoria = _normalizar_categoria(cat_match.group(1))
        confianza = float(conf_match.group(1)) if conf_match else 0
        descripcion = desc_match.group(1).strip() if desc_match else "(truncado)"
        return categoria, confianza, descripcion

    return "ERROR", 0, raw[:300]


# ── Dataset RWF-2000 ───────────────────────────────────────────────────────────

def descargar_rwf2000(destino="C:/TFG_Ada/dataset/rwf2000"):
    snapshot_download(
        repo_id="DanJoshua/RWF-2000",
        repo_type="dataset",
        local_dir=destino,
        token=HF_TOKEN,
    )
    print(f"✅ Dataset descargado en {destino}")


def analizar_rwf2000_local(
    carpeta="C:/Users/adaxi/OneDrive/Escritorio\\dataset/rwf2000/RWF-2000/RWF-2000/train",
    n_frames=30,
    output_csv="resultados_rwf_salamandra_es.csv",
):
    cargar_modelo()  # se carga una sola vez, no hay bucle de modelos como con Ollama

    # OJO: las etiquetas se mapean directamente a violencia/no_violencia para que
    # coincidan con las categorías que el prompt le pide al modelo (en el script
    # original se comparaba "fight"/"non-fight" contra "violencia"/"no_violencia",
    # lo que hacía que 'correcto' fuera siempre 0).
    videos_validos = []
    for label, nombre_carpeta in [("violencia", "Fight"), ("no_violencia", "NonFight")]:
        for video in Path(carpeta).rglob(f"{nombre_carpeta}/*.avi"):
            videos_validos.append((video, label))
        for video in Path(carpeta).rglob(f"{nombre_carpeta}/*.mp4"):
            videos_validos.append((video, label))

    print(f"Total vídeos encontrados: {len(videos_validos)}")

    campos = [
        "indice", "archivo", "label_real", "prediccion", "confianza", "descripcion",
        "correcto", "total_ms", "prompt_ms", "eval_ms", "prompt_tokens",
        "generated_tokens", "tokens_per_sec",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()

        for i, (video_path, label_real) in enumerate(videos_validos):
            print(f"\n[{i+1}/{len(videos_validos)}] {video_path.name} | Label: {label_real}")

            try:
                frames = extraer_frames(str(video_path), n_frames=n_frames)
                if not frames:
                    raise ValueError("No se pudieron extraer frames")

                grid = frames_a_grid(frames)
                raw, metricas = analizar_grid(grid)
                prediccion, confianza, descripcion = parsear_respuesta(raw)
                correcto = 1 if prediccion == label_real else 0

            except Exception as e:
                print(f"  ⚠️  Error: {e}")
                prediccion, confianza, descripcion, correcto = "ERROR", 0, str(e), 0
                metricas = {k: 0 for k in [
                    "total_ms", "prompt_ms", "eval_ms",
                    "prompt_tokens", "generated_tokens", "tokens_per_sec",
                ]}

            print(f"  Predicción: {prediccion} | Correcto: {correcto}")

            writer.writerow({
                "indice": i + 1,
                "archivo": video_path.name,
                "label_real": label_real,
                "prediccion": prediccion,
                "confianza": confianza,
                "descripcion": descripcion,
                "correcto": correcto,
                "total_ms": metricas.get("total_ms", 0),
                "prompt_ms": metricas.get("prompt_ms", 0),
                "eval_ms": metricas.get("eval_ms", 0),
                "prompt_tokens": metricas.get("prompt_tokens", 0),
                "generated_tokens": metricas.get("generated_tokens", 0),
                "tokens_per_sec": metricas.get("tokens_per_sec", 0),
            })
            f.flush()

    import pandas as pd
    from sklearn.metrics import classification_report

    df = pd.read_csv(output_csv)
    df_valid = df[df["prediccion"] != "ERROR"]
    print(f"\n✅ Accuracy: {df_valid['correcto'].mean():.2%}")
    print(classification_report(df_valid["label_real"], df_valid["prediccion"]))


if __name__ == "__main__":
    analizar_rwf2000_local(n_frames=30, output_csv="resultados_rwf_salamandra_es_v2.csv")