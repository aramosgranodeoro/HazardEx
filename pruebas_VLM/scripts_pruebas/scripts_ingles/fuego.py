import ollama, base64, csv, os, json, re
from pathlib import Path

VALID_CATEGORIES = {"smoke", "fire", "both", "none"}


def analyze_frame(image_path, model="llava:7b"):
    with open(image_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()
    print(f"Analyzing image: {image_path}")
    response = ollama.chat(
        model=model,
        options={
            "temperature": 0.1,
            "num_predict": 4096,  
        },
        messages=[{
            'role': 'user',
            'content': """
            # ROLE
            You are a fire analyst.

            # OBJECTIVE
            Analyze the image to identify fire signals such as smoke columns, flames, or heat haze. You must be able to recognize these signals even in low-quality, blurry, night-vision, or degraded photos.

            # DETECTION CRITERIA
            - **Low-Quality & Night-Time Analysis:** 
                - **Fire in poor conditions:** Look for clusters of high-intensity pixels that represent a **hotspot**. In low-quality, grainy, or night cameras, fire often overexposes the sensor. At a distance or in poor resolution, it appears as **bright white, yellow, or orange glowing spots/points**. It may not look like a traditional flame, but rather a bright, glowing, overexposed white blob, often surrounded by unnatural halos.
                - **Fire vs. Artificial Lights:** Differentiate irregular fire sources from structured artificial lighting. Unlike city lights, streetlamps, or spotlights—which are usually static, geometric, or part of a regular grid—fire spots are often more irregular and may show unnatural glow patterns in the surrounding environment.
                - **Smoke vs. Clouds:** Differentiate **smoke at a distance** by looking for grayish-white plumes or **blurred (difuminada) and hazy textures**. You must distinguish these from clouds; smoke typically originates from a specific ground source or obscures the horizon and vegetation unnaturally, whereas clouds are generally higher, more widespread, and have different structural patterns.
                - **Smoke in poor conditions:** Look for faint grayish areas or textures that unnaturally obscure the background/horizon or reflect the light of the fire to identify **smoke**, even if the image is highly noisy, grainy, or pixelated.
            - **Classification Priority:** 
                - If BOTH flames (or overexposed fire blobs) and smoke are present, you MUST return "both". 
                - Never categorize as "smoke" if there are visible incandescent or bright fire spots; in that case, use "both" or "fire".

            # OUTPUT FORMAT (MANDATORY)
            You MUST respond ONLY with a valid JSON object. No extra text, no markdown, no explanation before or after.
            Start your response directly with { and end with }.
            {
              "predicted_category": "<smoke|fire|both|none>",
              "confidence": <number between 0.0 and 1.0>,
              "description": "<one sentence describing what is happening>"
            }
            """,
            'images': [img_b64]
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

    return response, metricas


def parse_label_from_txt(txt_path):
    """
    Reads a YOLO label .txt file and returns the label string.
    YOLO class mapping for this dataset:
        0 = smoke
        1 = fire
    The file can contain 0, 1, both, or be empty (none).
    """
    if not txt_path.exists():
        return "none"

    content = txt_path.read_text().strip()
    if not content:
        return "none"

    classes = set()
    for line in content.splitlines():
        parts = line.strip().split()
        if parts:
            classes.add(parts[0])

    has_smoke = "0" in classes
    has_fire  = "1" in classes

    if has_fire and has_smoke:
        return "both"
    elif has_fire:
        return "fire"
    elif has_smoke:
        return "smoke"
    else:
        return "none"


def parse_model_response(response):
    """
    Extrae predicted_category, confidence y description de la respuesta del modelo.
    Maneja los casos habituales de salida malformada:
      1. JSON válido directo
      2. JSON dentro de fences ```json ... ```
      3. JSON con texto libre antes/después → busca el primer { ... }
      4. Texto libre con claves reconocibles (fallback por regex)
      5. Cualquier otra cosa → unknown
    """
    raw_text = response['message']['content'].strip()

    # ── Intento 1: JSON directo o dentro de fences ──────────────────────────
    clean = raw_text

    # Quitar fences ```json ... ``` o ``` ... ```
    if "```" in clean:
        for part in clean.split("```"):
            part = part.strip()
            if part.lower().startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                clean = part
                break

    # ── Intento 2: Extraer el primer bloque { ... } del texto ───────────────
    if not clean.startswith("{"):
        match = re.search(r'\{[^{}]*\}', raw_text, re.DOTALL)
        if match:
            clean = match.group(0)

    # ── Intento 3: Parsear el JSON encontrado ───────────────────────────────
    if clean.startswith("{"):
        try:
            data = json.loads(clean)
            predicted_category = str(data.get("predicted_category", "unknown")).lower().strip()
            # Normalizar si el modelo devuelve algo como "fire detected" en vez de "fire"
            if predicted_category not in VALID_CATEGORIES:
                predicted_category = _normalize_category(predicted_category)
            confidence  = _safe_float(data.get("confidence", 0.0))
            description = str(data.get("description", "")).replace("\n", " ").strip()
            return predicted_category, confidence, description
        except json.JSONDecodeError:
            pass

    # ── Intento 4: Regex sobre texto libre ──────────────────────────────────
    # Cubre respuestas tipo "Category: smoke\nConfidence: 0.9\nDescription: ..."
    cat_match  = re.search(
        r'(?:predicted[_\s]?category|category)["\s:]+([a-z]+)', raw_text, re.IGNORECASE)
    conf_match = re.search(
        r'confidence["\s:]+([0-9]*\.?[0-9]+)', raw_text, re.IGNORECASE)
    desc_match = re.search(
        r'description["\s:]+(.+?)(?:\n|$)', raw_text, re.IGNORECASE)

    if cat_match:
        predicted_category = _normalize_category(cat_match.group(1).lower().strip())
        confidence  = _safe_float(conf_match.group(1)) if conf_match else 0.0
        description = desc_match.group(1).strip() if desc_match else raw_text[:200]
        return predicted_category, confidence, description

    # ── Intento 5: Inferir categoría del texto libre ─────────────────────────
    # Si el modelo describió algo sin estructura, intentamos deducir la clase
    inferred = _infer_from_text(raw_text)
    if inferred:
        return inferred, 0.0, raw_text.replace("\n", " ")[:300]

    # ── Fallback total ───────────────────────────────────────────────────────
    return "unknown", 0.0, raw_text.replace("\n", " ")[:300]


def _safe_float(value):
    """Convierte a float de forma segura, devuelve 0.0 si falla."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalize_category(text: str) -> str:
    """
    Mapea variaciones comunes a una categoría válida.
    Ejemplos: 'fire detected' → 'fire', 'no fire' → 'none', 'smoke and fire' → 'both'
    """
    text = text.lower()
    if "both" in text or ("fire" in text and "smoke" in text):
        return "both"
    if "fire" in text:
        return "fire"
    if "smoke" in text:
        return "smoke"
    if any(w in text for w in ["none", "no fire", "no smoke", "clear", "normal"]):
        return "none"
    return "unknown"


def _infer_from_text(text: str) -> str | None:
    """
    Intenta inferir la categoría leyendo el texto libre del modelo.
    Devuelve None si no hay suficiente señal.
    """
    text_lower = text.lower()
    has_fire  = any(w in text_lower for w in ["flame", "fire", "blaze", "burning", "ignit"])
    has_smoke = any(w in text_lower for w in ["smoke", "plume", "haze", "fume"])
    no_signal = any(w in text_lower for w in [
        "no fire", "no smoke", "no flames", "no visible", "not detect",
        "cannot detect", "clear sky", "no sign"
    ])

    if no_signal and not has_fire and not has_smoke:
        return "none"
    if has_fire and has_smoke:
        return "both"
    if has_fire:
        return "fire"
    if has_smoke:
        return "smoke"
    return None


def analyze_directory(directory_path, output_csv="results.csv", model="llava:7b"):
    """
    Walks directory_path recursively, finds every image file, pairs it with
    its YOLO label .txt, runs analyze_frame on each image, and writes one CSV row per image.

    Expected dataset layout (standard YOLO structure):
        <directory_path>/
            images/   *.jpg / *.png  ...
            labels/   *.txt          ...
    """
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    directory_path = Path(directory_path)

    image_files = [
        p for p in directory_path.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        print(f"No images found in {directory_path}")
        return

    print(f"Found {len(image_files)} image(s). Starting analysis...")

    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["route", "label", "predicted_category", "confidence", "description", "total_ms", "prompt_ms", "eval_ms", "prompt_tokens", "generated_tokens", "tokens_per_sec"]
        )
        writer.writeheader()

        stats = {"ok": 0, "unknown": 0, "error": 0}

        for idx, img_path in enumerate(sorted(image_files), 1):
            print(f"\n[{idx}/{len(image_files)}] {img_path.name}", end=" ... ")

            # Localizar el .txt de label
            label_candidate = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
            if not label_candidate.exists():
                label_candidate = img_path.with_suffix(".txt")
            label = parse_label_from_txt(label_candidate)

            # Ejecutar el modelo
            try:
                response, metricas = analyze_frame(str(img_path), model=model)
                predicted_category, confidence, description = parse_model_response(response)
            except Exception as e:
                print(f"ERROR: {e}")
                predicted_category, confidence, description = "error", 0.0, str(e)
                stats["error"] += 1
            else:
                if predicted_category == "unknown":
                    stats["unknown"] += 1
                    print(f"UNKNOWN  (label={label})")
                else:
                    stats["ok"] += 1
                    print(f"{predicted_category} ({confidence:.2f})  label={label}")

            writer.writerow({
                "route":              str(img_path),
                "label":              label,
                "predicted_category": predicted_category,
                "confidence":         confidence,
                "description":        description,
                "total_ms":           metricas.get("total_ms", 0),
                "prompt_ms":          metricas.get("prompt_ms", 0),
                "eval_ms":            metricas.get("eval_ms", 0),
                "prompt_tokens":      metricas.get("prompt_tokens", 0),
                "generated_tokens":   metricas.get("generated_tokens", 0),
                "tokens_per_sec":     metricas.get("tokens_per_sec", 0)
            })
            csv_file.flush()

    # Resumen final
    total = sum(stats.values())
    print(f"\n{'='*50}")
    print(f"  RESUMEN FINAL")
    print(f"{'='*50}")
    print(f"  Total procesadas : {total}")
    print(f"  Correctas        : {stats['ok']}  ({stats['ok']/total*100:.1f}%)")
    print(f"  Unknown          : {stats['unknown']}  ({stats['unknown']/total*100:.1f}%)")
    print(f"  Errores          : {stats['error']}  ({stats['error']/total*100:.1f}%)")
    print(f"\n  Resultados guardados en: {output_csv}")


def main():
    url = "C:/Users/adaxi/OneDrive/Escritorio/dataset/fuego_datatset/data/val"
    modelos = [
        "internlm/interns1:mini-q8_0",
        "qwen3.5:latest"
    ]
    for model in modelos:
        model_name_safe = model.replace(":", "_").replace("/", "_").replace("\\", "_")
        analyze_directory(url, output_csv=f"resultados_fuego_{model_name_safe}.csv", model=model)


if __name__ == "__main__":
    main()