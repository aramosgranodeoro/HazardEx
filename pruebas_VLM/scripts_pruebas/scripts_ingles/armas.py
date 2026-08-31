import ollama
import base64
import csv
import io
import json
import re
import requests
from pathlib import Path
from PIL import Image

# ── Configuración ─────────────────────────────────────────────────────────────

REPO        = "ari-dasci/OD-WeaponDetection"
CARPETA     = "Weapons and similar handled objects/Sohas_weapon-Detection/images_test"
GITHUB_API  = f"https://api.github.com/repos/{REPO}/contents/{CARPETA}"

PALABRAS_WEAPON    = ["knife", "pistol", "frame", "dsc", "defense", "hb", "KravMaga", "ruso"]
PALABRAS_NO_WEAPON = ["smartphone", "billete", "monedero", "tarjeta", "img"]
PALABRAS_VALIDAS   = PALABRAS_WEAPON + PALABRAS_NO_WEAPON


# ── Utilidades ────────────────────────────────────────────────────────────────

def pil_to_base64(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def obtener_label(nombre_archivo):
    """Devuelve 'weapon', 'no_weapon' o None si no debe procesarse."""
    nombre = nombre_archivo.lower()
    
    if not any(p in nombre for p in PALABRAS_VALIDAS):
        return None  # ignorar
    
    if any(p in nombre for p in PALABRAS_WEAPON):
        return "weapon"
    
    return "no_weapon"


def listar_imagenes_github():
    """Lista las imágenes del repo via GitHub API."""
    print("Obteniendo lista de imágenes desde GitHub...")
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(GITHUB_API, headers=headers)
    response.raise_for_status()
    
    archivos = response.json()
    return [
        {"nombre": a["name"], "url": a["download_url"]}
        for a in archivos
        if a["name"].lower().endswith((".jpg", ".jpeg", ".png"))
    ]


def descargar_imagen(url):
    """Descarga una imagen desde URL y la devuelve como PIL Image."""
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


# ── Modelo ────────────────────────────────────────────────────────────────────

def analizar_imagen(pil_image, model="llava:7b"):
    response = ollama.chat(
        model=model,
        options={"temperature": 0.1, "num_predict": 4096, "num_ctx": 8192 },
        messages=[{
            "role": "system",
            "content": "You are a security analyst. Do not think out loud. Respond ONLY with the requested JSON."
        }, {
            "role": "user",
            "content": """
            # ROLE
            You are a security analyst reviewing images.

            # OBJECTIVE
            Determine if the image contains a dangerous weapon (knife or pistol) or not.

            # GUIDELINES
            - Respond "weapon" if you see a knife, pistol, or anything that looks like a weapon.

            # OUTPUT FORMAT
            Respond ONLY with valid JSON:
            {
                "predicted_category": "weapon or no_weapon",
                "confidence": 0.0,
                "description": "one sentence describing what you see"
            }
            """,
            "images": [pil_to_base64(pil_image)]
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
    raw = re.sub(r"<tool_call>.*?</tool_call>", "", raw, flags=re.DOTALL)
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


# ── Pipeline principal ────────────────────────────────────────────────────────

def analizar_dataset(output_csv="resultados_weapons_llava.csv", model="llava:7b"):
    imagenes = listar_imagenes_github()
    
    # Filtra las que no deben procesarse
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

    campos = ["indice", "archivo", "label_real", "prediccion", "confianza", "descripcion", "correcto", "total_ms", "prompt_ms", "eval_ms", "prompt_tokens", "generated_tokens", "tokens_per_sec"]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()

        for i, img in enumerate(imagenes_validas):
            nombre     = img["nombre"]
            label_real = obtener_label(nombre)

            print(f"\n[{i+1}/{validas}] {nombre} | Label: {label_real}")

            try:
                pil_image               = descargar_imagen(img["url"])
                raw, metricas          = analizar_imagen(pil_image, model=model)
                prediccion, confianza, descripcion = parsear_respuesta(raw)
                correcto                = 1 if prediccion == label_real else 0
            except Exception as e:
                print(f"  ⚠️  Error: {e}")
                prediccion, confianza, descripcion, correcto = "ERROR", 0, "", 0

            print(f"  Predicción : {prediccion} | Correcto: {correcto}")

            writer.writerow({
                "indice":      i + 1,
                "archivo":     nombre,
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

    # Métricas finales
    import pandas as pd
    from sklearn.metrics import classification_report
    df = pd.read_csv(output_csv)
    print(f"\n✅ Accuracy global: {df['correcto'].mean():.2%}")
    print(f"   Procesadas: {len(df)} | Ignoradas: {ignoradas}")
    print(classification_report(df["label_real"], df["prediccion"]))


if __name__ == "__main__":
    modelos = [
        "llava:7b",
        "internlm/interns1:mini-q8_0",
        "qwen3.5:latest"
    ]
    for model in modelos:
        model_name_safe = model.replace(":", "_").replace("/", "_").replace("\\", "_")
        analizar_dataset(output_csv=f"resultados_weapons_{model_name_safe}.csv", model=model)