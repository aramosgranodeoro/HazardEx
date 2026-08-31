import csv
import json
import os
import re
import time

import torch
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# ── Configuración ─────────────────────────────────────────────────────────────

MODEL_ID   = "BSC-LT/Salamandra-VL-7B-2512"
OUTPUT_CSV = "resultados_chartqa_salamandra_es.csv"
MAX_EJEMPLOS = 2500  # cap, igual que en el resto del TFG

# $env:HF_TOKEN = "hf_..."
HF_TOKEN = os.getenv("HF_TOKEN")

PROMPT_TEMPLATE = """Eres un analista de gráficos experto.

Responde la siguiente pregunta sobre la imagen del gráfico. La pregunta puede estar en inglés; respóndela igualmente, dando la respuesta en el mismo idioma/formato en que aparece en el gráfico (números, nombres de categorías, etc. tal y como se ven en la imagen).

PREGUNTA: {query}

Directrices:
1. Da una respuesta breve y exacta (un número, una palabra o una frase corta), tal como aparece en el gráfico.
2. No inventes datos que no estén en la imagen.

Responde ÚNICAMENTE con JSON válido, sin explicaciones, sin markdown:
{{
    "respuesta": "escribe aquí solo la respuesta, nada más",
    "confianza": 0.00,
    "razonamiento": "una frase explicando tu razonamiento"
}}"""

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


# ── Utilidades de imagen ──────────────────────────────────────────────────────

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

def analizar_chart(pil_image, query):
    cargar_modelo()

    prompt = PROMPT_TEMPLATE.format(query=query)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
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


# ── Parser (7 pasos, igual patrón que weapons/violence/fire) ─────────────────

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

    # 1. Quita <think>...</think> completos
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    # 2. Quita </think> huérfano (modelos que se comen la etiqueta de apertura)
    if "</think>" in raw:
        raw = raw.split("</think>")[-1]
    raw = raw.strip()

    # 3. Quita bloques ```json ... ```
    if "```" in raw:
        for parte in raw.split("```"):
            if "respuesta" in parte:
                raw = parte.removeprefix("json").strip()
                break

    # 4. Extracción JSON balanceada (tolera texto extra antes/después)
    json_str = extraer_json_balanceado(raw)
    if json_str:
        try:
            data = json.loads(json_str)
            return (
                str(data.get("respuesta", "ERROR")).strip(),
                data.get("confianza", 0),
                data.get("razonamiento", ""),
            )
        except json.JSONDecodeError:
            pass

    # 5. Regex directa por campo (para JSON truncado)
    resp_match = re.search(r'"respuesta"\s*:\s*"([^"]*)"', raw, re.IGNORECASE)
    conf_match = re.search(r'"confianza"\s*:\s*([0-9.]+)', raw, re.IGNORECASE)
    raz_match  = re.search(r'"razonamiento"\s*:\s*"([^"]*)', raw, re.IGNORECASE)

    if resp_match:
        return (
            resp_match.group(1).strip(),
            float(conf_match.group(1)) if conf_match else 0,
            raz_match.group(1).strip() if raz_match else "(truncado)",
        )

    # 6. Fallback texto libre: "respuesta: X" / "the answer is X"
    patrones = [
        r'respuesta[:\s]+([^\.\n]+)',
        r'answer is[:\s]+([^\.\n]+)',
    ]
    for patron in patrones:
        match = re.search(patron, raw, re.IGNORECASE)
        if match:
            return match.group(1).strip(), 0, "(fallback texto libre)"

    # 7. Último recurso
    return "ERROR", 0, raw[:300]


# ── Pipeline principal ────────────────────────────────────────────────────────

def analizar_dataset():
    print("Cargando dataset ChartQA (streaming)...")
    dataset = load_dataset("HuggingFaceM4/ChartQA", split="test", streaming=True)

    campos = [
        "indice", "query", "label", "respuesta_modelo", "confianza", "razonamiento",
        "total_ms", "prompt_ms", "eval_ms", "prompt_tokens",
        "generated_tokens", "tokens_per_sec",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()

        for i, ejemplo in enumerate(dataset):
            if i >= MAX_EJEMPLOS:
                break

            pil_image = ejemplo["image"]
            query     = ejemplo["query"]
            label     = ejemplo["label"][0]

            print(f"\n[{i+1}/{MAX_EJEMPLOS}] Pregunta: {query}")
            print(f"     Respuesta correcta: {label}")

            try:
                raw, metricas = analizar_chart(pil_image, query)
                resp_modelo, confianza, razonamiento = parsear_respuesta(raw)
            except Exception as e:
                print(f"     ⚠️  Error en ejemplo {i+1}: {e}")
                resp_modelo, confianza, razonamiento = "ERROR", 0, str(e)
                metricas = {k: 0 for k in [
                    "total_ms", "prompt_ms", "eval_ms",
                    "prompt_tokens", "generated_tokens", "tokens_per_sec",
                ]}

            print(f"     Respuesta modelo: {resp_modelo}")

            writer.writerow({
                "indice":           i + 1,
                "query":            query,
                "label":            label,
                "respuesta_modelo": resp_modelo,
                "confianza":        confianza,
                "razonamiento":     razonamiento,
                "total_ms":         metricas.get("total_ms", 0),
                "prompt_ms":        metricas.get("prompt_ms", 0),
                "eval_ms":          metricas.get("eval_ms", 0),
                "prompt_tokens":    metricas.get("prompt_tokens", 0),
                "generated_tokens": metricas.get("generated_tokens", 0),
                "tokens_per_sec":   metricas.get("tokens_per_sec", 0),
            })
            f.flush()

    print(f"\n✅ Análisis completado. Resultados guardados en '{OUTPUT_CSV}'")


if __name__ == "__main__":
    analizar_dataset()