import os
import csv
import json
import re
import time
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# ── Configuración ─────────────────────────────────────────────────────────────

MODEL_ID   = "BSC-LT/Salamandra-VL-7B-2512"
OUTPUT_CSV = "resultados_fuego_salamandra_es.csv"

HF_TOKEN = os.getenv("HF_TOKEN")

DATASET_PATH = "C:/Users/adaxi/OneDrive/Escritorio/dataset/fuego_datatset/data/val"

VALID_CATEGORIES = {"humo", "fuego", "ambos", "none"}

LABEL_MAP = {
    "smoke": "humo",
    "fire":  "fuego",
    "both":  "ambos",
    "none":  "none",
}

PROMPT = """Eres un analista de incendios.

Analiza la imagen para identificar señales de fuego como columnas de humo, llamas o distorsión térmica. Debes ser capaz de reconocer estas señales incluso en fotos de baja calidad, borrosas, de visión nocturna o degradadas.

Criterios de detección:
1. Fuego en malas condiciones: Busca agrupaciones de píxeles de alta intensidad (punto caliente). En cámaras de baja calidad o nocturnas, el fuego aparece como manchas o puntos brillantes de color blanco, amarillo o naranja. Puede no parecer una llama tradicional, sino una mancha brillante sobreexpuesta rodeada de halos anómalos.
2. Fuego vs. Luces Artificiales: Las fuentes de fuego son irregulares, a diferencia de las luces de ciudad o farolas, que son estáticas y geométricas.
3. Humo vs. Nubes: El humo aparece como penachos grisáceo-blanquecinos que se originan en una fuente concreta en el suelo y obscurecen el horizonte de forma antinatural. Las nubes son más altas y estructuradas.
4. Si están presentes TANTO llamas como humo, DEBES devolver "ambos".
5. Nunca clasifiques como "humo" si hay focos incandescentes visibles; usa "ambos" o "fuego".

Reglas:
1. Escribe ÚNICAMENTE una de estas palabras exactas: humo, fuego, ambos, none
2. No se aceptan otros valores

Responde ÚNICAMENTE con JSON válido, sin explicaciones adicionales, sin markdown:
{
    "categoria_predicha": "escribe aquí la categoría detectada",
    "confianza": 0.00,
    "descripcion": "una frase describiendo lo que está ocurriendo"
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


# ── Lectura de etiquetas YOLO ─────────────────────────────────────────────────

def parse_label_from_txt(txt_path):
    """
    Lee un archivo de etiquetas YOLO y devuelve la etiqueta en español.
        0 = smoke → humo
        1 = fire  → fuego
    """
    if not txt_path.exists():
        return "none"
    content = txt_path.read_text().strip()
    if not content:
        return "none"

    classes   = set()
    for line in content.splitlines():
        parts = line.strip().split()
        if parts:
            classes.add(parts[0])

    has_smoke = "0" in classes
    has_fire  = "1" in classes

    if has_fire and has_smoke:
        return "ambos"
    elif has_fire:
        return "fuego"
    elif has_smoke:
        return "humo"
    return "none"


# ── Utilidades de imagen ──────────────────────────────────────────────────────

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

def analizar_frame(image_path):
    cargar_modelo()

    pil_image = Image.open(image_path).convert("RGB")
    pil_image = redimensionar_con_padding(pil_image, size=672)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT},
        ]
    }]

    text   = _processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = _processor(
        text=text,
        images=[pil_image],
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
    for en, es in LABEL_MAP.items():
        if en in texto:
            return es
    if "ambos" in texto or ("fuego" in texto and "humo" in texto):
        return "ambos"
    if "fuego" in texto or "llama" in texto or "incendio" in texto:
        return "fuego"
    if "humo" in texto or "columna" in texto or "penacho" in texto:
        return "humo"
    if any(w in texto for w in ["none", "nada", "normal", "sin", "no hay"]):
        return "none"
    return "unknown"


def _inferir_desde_texto(texto: str) -> str | None:
    t = texto.lower()
    tiene_fuego = any(w in t for w in ["fuego", "llama", "incendio", "ardiendo", "fire", "flame"])
    tiene_humo  = any(w in t for w in ["humo", "columna", "penacho", "smoke", "plume"])
    sin_señal   = any(w in t for w in [
        "no hay fuego", "no hay humo", "no se detecta", "sin fuego",
        "no fire", "no smoke", "no visible", "clear",
    ])
    if sin_señal and not tiene_fuego and not tiene_humo:
        return "none"
    if tiene_fuego and tiene_humo:
        return "ambos"
    if tiene_fuego:
        return "fuego"
    if tiene_humo:
        return "humo"
    return None


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parsear_respuesta(raw):
    print(f"Respuesta cruda del modelo:\n{raw}\n")

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


# ── Pipeline principal ────────────────────────────────────────────────────────

def analizar_directorio(directory_path=DATASET_PATH, output_csv=OUTPUT_CSV):
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    directory_path   = Path(directory_path)

    image_files = sorted([
        p for p in directory_path.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_files:
        print(f"No se encontraron imágenes en {directory_path}")
        return

    cargar_modelo()  # una sola carga al inicio
    print(f"Encontradas {len(image_files)} imágenes. Iniciando análisis...")
    print(f"Modelo: {MODEL_ID}\n")

    campos = [
        "ruta", "etiqueta", "categoria_predicha", "confianza", "descripcion",
        "total_ms", "prompt_ms", "eval_ms", "prompt_tokens", "generated_tokens", "tokens_per_sec",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=campos)
        writer.writeheader()

        stats = {"ok": 0, "unknown": 0, "error": 0}

        for idx, img_path in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] {img_path.name}", end=" ... ")

            # Localizar el .txt de etiqueta YOLO
            label_candidate = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
            if not label_candidate.exists():
                label_candidate = img_path.with_suffix(".txt")
            etiqueta = parse_label_from_txt(label_candidate)

            metricas = {k: 0 for k in [
                "total_ms", "prompt_ms", "eval_ms",
                "prompt_tokens", "generated_tokens", "tokens_per_sec",
            ]}

            try:
                raw, metricas                    = analizar_frame(str(img_path))
                categoria, confianza, descripcion = parsear_respuesta(raw)
            except Exception as e:
                print(f"ERROR: {e}")
                categoria, confianza, descripcion = "ERROR", 0.0, str(e)
                stats["error"] += 1
            else:
                if categoria in ("unknown", "ERROR"):
                    stats["unknown"] += 1
                    print(f"UNKNOWN  (etiqueta={etiqueta})")
                else:
                    stats["ok"] += 1
                    print(f"{categoria} ({confianza:.2f})  etiqueta={etiqueta}")

            writer.writerow({
                "ruta":               str(img_path),
                "etiqueta":           etiqueta,
                "categoria_predicha": categoria,
                "confianza":          confianza,
                "descripcion":        descripcion,
                "total_ms":           metricas.get("total_ms", 0),
                "prompt_ms":          metricas.get("prompt_ms", 0),
                "eval_ms":            metricas.get("eval_ms", 0),
                "prompt_tokens":      metricas.get("prompt_tokens", 0),
                "generated_tokens":   metricas.get("generated_tokens", 0),
                "tokens_per_sec":     metricas.get("tokens_per_sec", 0),
            })
            csv_file.flush()

    total = sum(stats.values())
    print(f"\n{'='*50}")
    print(f"  RESUMEN FINAL")
    print(f"{'='*50}")
    print(f"  Total procesadas : {total}")
    print(f"  Correctas        : {stats['ok']}  ({stats['ok']/max(total,1)*100:.1f}%)")
    print(f"  Unknown/Error    : {stats['unknown'] + stats['error']}  "
          f"({(stats['unknown']+stats['error'])/max(total,1)*100:.1f}%)")
    print(f"\n  Resultados guardados en: {output_csv}")

    import pandas as pd
    from sklearn.metrics import classification_report

    df       = pd.read_csv(output_csv)
    df_valid = df[df["categoria_predicha"] != "ERROR"]
    print(f"\nAccuracy: {df_valid.apply(lambda r: r['etiqueta'] == r['categoria_predicha'], axis=1).mean():.2%}")
    print(classification_report(df_valid["etiqueta"], df_valid["categoria_predicha"]))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    analizar_directorio()