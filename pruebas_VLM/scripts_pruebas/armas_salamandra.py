import csv
import io
import json
import os
import re
import time
from pathlib import Path

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# ── Configuración ─────────────────────────────────────────────────────────────

MODEL_ID   = "BSC-LT/Salamandra-VL-7B-2512"
OUTPUT_CSV = "resultados_weapons_salamandra_es.csv"

# El token de HF como variable de entorno, no en el código:
# $env:HF_TOKEN = "hf_..."
HF_TOKEN = os.getenv("HF_TOKEN")

REPO       = "ari-dasci/OD-WeaponDetection"
CARPETA    = "Weapons and similar handled objects/Sohas_weapon-Detection/images_test"
GITHUB_API = f"https://api.github.com/repos/{REPO}/contents/{CARPETA}"

PALABRAS_WEAPON    = ["knife", "pistol", "frame", "dsc", "defense", "hb", "KravMaga", "ruso"]
PALABRAS_NO_WEAPON = ["smartphone", "billete", "monedero", "tarjeta", "img"]
PALABRAS_VALIDAS   = PALABRAS_WEAPON + PALABRAS_NO_WEAPON

CATEGORIAS_VALIDAS = {"arma", "no_arma"}

PROMPT = """Eres un analista de seguridad revisando imágenes.

Determina si la imagen contiene un arma peligrosa (cuchillo o pistola) o no.

Directrices:
1. Responde "arma" si ves un cuchillo, una pistola o cualquier objeto que parezca un arma (navaja, escopeta, etc.)
2. Responde "no_arma" si no ves ningún objeto que parezca un arma
3. Escribe ÚNICAMENTE una de estas palabras exactas: arma, no_arma

Responde ÚNICAMENTE con JSON válido, sin explicaciones, sin markdown:
{
    "categoria_predicha": "escribe aquí la categoría detectada",
    "confianza": 0.00,
    "descripcion": "una frase describiendo lo que ves"
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


# ── Utilidades ────────────────────────────────────────────────────────────────

def obtener_label(nombre_archivo):
    """Devuelve 'arma', 'no_arma' o None si no debe procesarse."""
    nombre = nombre_archivo.lower()
    if not any(p.lower() in nombre for p in PALABRAS_VALIDAS):
        return None
    if any(p.lower() in nombre for p in PALABRAS_WEAPON):
        return "arma"
    return "no_arma"


def listar_imagenes_github():
    print("Obteniendo lista de imágenes desde GitHub...")
    headers  = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(GITHUB_API, headers=headers)
    response.raise_for_status()
    archivos = response.json()
    return [
        {"nombre": a["name"], "url": a["download_url"]}
        for a in archivos
        if a["name"].lower().endswith((".jpg", ".jpeg", ".png"))
    ]


def descargar_imagen(url):
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def redimensionar_con_padding(img, size=672):
    """Escala manteniendo aspect ratio y rellena con negro."""
    img = img.copy()
    img.thumbnail((size, size), Image.LANCZOS)
    lienzo = Image.new("RGB", (size, size), color=(0, 0, 0))
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    lienzo.paste(img, (x, y))
    return lienzo


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

    pil_image = redimensionar_con_padding(pil_image.convert("RGB"), size=672)

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


# ── Parser ────────────────────────────────────────────────────────────────────

def _normalizar_categoria(valor):
    v = str(valor).lower().strip()
    if "no" in v and "arma" in v:
        return "no_arma"
    if "arma" in v or "pistol" in v or "knife" in v or "gun" in v or "weapon" in v:
        return "arma"
    return v if v in CATEGORIAS_VALIDAS else "ERROR"


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

    cat_match  = re.search(r'"categoria_predicha"\s*:\s*"([^"]+)"', raw, re.IGNORECASE)
    conf_match = re.search(r'"confianza"\s*:\s*([0-9.]+)', raw, re.IGNORECASE)
    desc_match = re.search(r'"descripcion"\s*:\s*"([^"]*)', raw, re.IGNORECASE)

    if cat_match:
        return (
            _normalizar_categoria(cat_match.group(1)),
            float(conf_match.group(1)) if conf_match else 0,
            desc_match.group(1).strip() if desc_match else "(truncado)",
        )

    return "ERROR", 0, raw[:300]


# ── Pipeline principal ────────────────────────────────────────────────────────

def analizar_dataset():
    imagenes = listar_imagenes_github()

    imagenes_validas = [
        img for img in imagenes
        if obtener_label(img["nombre"]) is not None
    ]

    total    = len(imagenes)
    validas  = len(imagenes_validas)
    ignoradas = total - validas
    print(f"Total imágenes : {total}")
    print(f"A procesar     : {validas}")
    print(f"Ignoradas      : {ignoradas}")
    print(f"Modelo         : {MODEL_ID}\n")

    campos = [
        "indice", "archivo", "label_real", "prediccion", "confianza", "descripcion",
        "correcto", "total_ms", "prompt_ms", "eval_ms", "prompt_tokens",
        "generated_tokens", "tokens_per_sec",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()

        for i, img in enumerate(imagenes_validas):
            nombre     = img["nombre"]
            label_real = obtener_label(nombre)
            print(f"\n[{i+1}/{validas}] {nombre} | Label: {label_real}")

            try:
                pil_image                          = descargar_imagen(img["url"])
                raw, metricas                      = analizar_imagen(pil_image)
                prediccion, confianza, descripcion = parsear_respuesta(raw)
                correcto                           = 1 if prediccion == label_real else 0
            except Exception as e:
                print(f"  ⚠️  Error: {e}")
                prediccion, confianza, descripcion, correcto = "ERROR", 0, str(e), 0
                metricas = {k: 0 for k in [
                    "total_ms", "prompt_ms", "eval_ms",
                    "prompt_tokens", "generated_tokens", "tokens_per_sec",
                ]}

            print(f"  Predicción: {prediccion} | Correcto: {correcto}")

            writer.writerow({
                "indice":           i + 1,
                "archivo":          nombre,
                "label_real":       label_real,
                "prediccion":       prediccion,
                "confianza":        confianza,
                "descripcion":      descripcion,
                "correcto":         correcto,
                "total_ms":         metricas.get("total_ms", 0),
                "prompt_ms":        metricas.get("prompt_ms", 0),
                "eval_ms":          metricas.get("eval_ms", 0),
                "prompt_tokens":    metricas.get("prompt_tokens", 0),
                "generated_tokens": metricas.get("generated_tokens", 0),
                "tokens_per_sec":   metricas.get("tokens_per_sec", 0),
            })
            f.flush()

    import pandas as pd
    from sklearn.metrics import classification_report

    df       = pd.read_csv(OUTPUT_CSV)
    df_valid = df[df["prediccion"] != "ERROR"]

    print(f"\n{'='*55}")
    print(f"Modelo    : {MODEL_ID}")
    print(f"Total     : {validas} | Errores: {len(df) - len(df_valid)} | Ignoradas: {ignoradas}")
    print(f"Accuracy  : {df_valid['correcto'].mean():.2%}")
    print(f"\n{classification_report(df_valid['label_real'], df_valid['prediccion'])}")
    print(f"\nCSV guardado en: {OUTPUT_CSV}")


if __name__ == "__main__":
    analizar_dataset()