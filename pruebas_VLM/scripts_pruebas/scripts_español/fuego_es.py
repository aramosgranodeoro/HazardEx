import ollama, base64, csv, os, json, re
from pathlib import Path

VALID_CATEGORIES = {"humo", "fuego", "ambos", "none"}

# Mapeo para normalizar etiquetas del dataset YOLO (inglés → español)
LABEL_MAP = {
    "smoke": "humo",
    "fire":  "fuego",
    "both":  "ambos",
    "none":  "none",
}


# ── Utilidades ────────────────────────────────────────────────────────────────

def analizar_frame(image_path, model="llava:7b"):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    print(f"Analizando imagen: {image_path}")
    response = ollama.chat(
        model=model,
        options={"temperature": 0.1, "num_predict": 4096, "think": False},
        messages=[{
            "role": "user",
            "content": """
            # ROL
            Eres un analista de incendios.

            # OBJETIVO
            Analiza la imagen para identificar señales de fuego como columnas de humo, llamas o distorsión térmica. Debes ser capaz de reconocer estas señales incluso en fotos de baja calidad, borrosas, de visión nocturna o degradadas.

            # CRITERIOS DE DETECCIÓN
            - **Análisis en Baja Calidad y Nocturno:**
                - **Fuego en malas condiciones:** Busca agrupaciones de píxeles de alta intensidad que representen un **punto caliente**. En cámaras de baja calidad, granulosas o nocturnas, el fuego suele sobreexponer el sensor. A distancia o en baja resolución, aparece como **manchas/puntos brillantes de color blanco, amarillo o naranja resplandeciente**. Puede que no parezca una llama tradicional, sino más bien una mancha brillante, luminosa y sobreexpuesta, a menudo rodeada de halos anormales.
                - **Fuego vs. Luces Artificiales:** Diferencia las fuentes de fuego irregulares de la iluminación artificial estructurada. A diferencia de las luces de ciudad, farolas o focos —que suelen ser estáticos, geométricos o forman parte de una cuadrícula regular— los focos de fuego son generalmente más irregulares y pueden mostrar patrones de brillo anómalos en el entorno.
                - **Humo vs. Nubes:** Diferencia el **humo a distancia** buscando penachos grisáceo-blanquecinos o **texturas difuminadas y neblinosas**. Debes distinguirlos de las nubes; el humo típicamente se origina en una fuente concreta en el suelo u obscurece el horizonte y la vegetación de forma antinatural, mientras que las nubes son generalmente más altas, más extendidas y tienen patrones estructurales distintos.
                - **Humo en malas condiciones:** Busca áreas o texturas grisáceas tenues que obscurezcan de forma antinatural el fondo/horizonte o reflejen la luz del fuego para identificar el **humo**, incluso si la imagen es muy ruidosa, granulosa o pixelada.
            - **Prioridad de Clasificación:**
                - Si están presentes TANTO llamas (o manchas de fuego sobreexpuestas) como humo, DEBES devolver "ambos".
                - Nunca clasifiques como "humo" si hay focos incandescentes o puntos de fuego brillantes visibles; en ese caso, usa "ambos" o "fuego".

            # FORMATO DE SALIDA (OBLIGATORIO)
            Responde ÚNICAMENTE con un JSON válido, sin explicaciones adicionales, sin markdown, sin texto antes o después:
            {
              "categoria_predicha": "escribe aquí solo una palabra de la lista: humo, fuego, ambos, none",
              "confianza": 0.0,
              "descripcion": "una frase describiendo lo que está ocurriendo"
            }
            """,
            "images": [img_b64]
        }]
    )

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
    return response, metricas


# ── Lectura de etiquetas YOLO ─────────────────────────────────────────────────

def parse_label_from_txt(txt_path):
    """
    Lee un archivo de etiquetas YOLO y devuelve la etiqueta en español.
    Mapeo de clases:
        0 = smoke → humo
        1 = fire  → fuego
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
        return "ambos"
    elif has_fire:
        return "fuego"
    elif has_smoke:
        return "humo"
    else:
        return "none"


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
    """Mapea variaciones comunes a una categoría válida en español."""
    texto = texto.lower()
    # Primero intentar mapeo directo desde inglés
    for en, es in LABEL_MAP.items():
        if en in texto:
            return es
    # Luego buscar términos en español
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
    """Intenta inferir la categoría leyendo el texto libre del modelo. Devuelve None si no hay señal."""
    t = texto.lower()
    tiene_fuego = any(w in t for w in ["fuego", "llama", "incendio", "ardiendo", "fire", "flame"])
    tiene_humo  = any(w in t for w in ["humo", "columna", "penacho", "smoke", "plume"])
    sin_señal   = any(w in t for w in [
        "no hay fuego", "no hay humo", "no se detecta", "sin fuego",
        "no fire", "no smoke", "no visible", "clear"
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


def parsear_respuesta(response):
    """
    Extrae categoria_predicha, confianza y descripcion de la respuesta del modelo.
    Estrategias en orden:
      1. Eliminar bloques <think>...</think>
      2. Manejar </think> huérfano (modelos tipo Qwen/Intern)
      3. Extraer JSON de fences ```...```
      4. Extraer JSON con llaves balanceadas (soporta texto alrededor)
      5. Fallback campo a campo por regex (JSON truncado)
      6. Inferir categoría desde texto libre
      7. ERROR con preview del texto crudo
    """
    raw = response["message"]["content"].strip()
    print(f"Respuesta cruda del modelo:\n{raw}\n")

    # 1. Eliminar bloques <think> completos
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)

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


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# ── Pipeline principal ────────────────────────────────────────────────────────

def analizar_directorio(directory_path, output_csv="resultados.csv", model="llava:7b"):
    """
    Recorre directory_path recursivamente, localiza cada imagen, la empareja con
    su etiqueta YOLO (.txt), ejecuta el modelo y escribe una fila CSV por imagen.

    Estructura esperada (YOLO estándar):
        <directory_path>/
            images/   *.jpg / *.png  ...
            labels/   *.txt          ...
    """
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    directory_path = Path(directory_path)

    image_files = sorted([
        p for p in directory_path.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_files:
        print(f"No se encontraron imágenes en {directory_path}")
        return

    print(f"Encontradas {len(image_files)} imágenes. Iniciando análisis...")

    campos = [
        "ruta", "etiqueta", "categoria_predicha", "confianza", "descripcion",
        "total_ms", "prompt_ms", "eval_ms", "prompt_tokens", "generated_tokens", "tokens_per_sec"
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=campos)
        writer.writeheader()

        stats = {"ok": 0, "unknown": 0, "error": 0}

        for idx, img_path in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] {img_path.name}", end=" ... ")

            # Localizar el .txt de etiqueta
            label_candidate = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
            if not label_candidate.exists():
                label_candidate = img_path.with_suffix(".txt")
            etiqueta = parse_label_from_txt(label_candidate)

            # Ejecutar el modelo
            metricas = {k: 0 for k in ["total_ms", "prompt_ms", "eval_ms",
                                        "prompt_tokens", "generated_tokens", "tokens_per_sec"]}
            try:
                response, metricas = analizar_frame(str(img_path), model=model)
                categoria, confianza, descripcion = parsear_respuesta(response)
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

    # Resumen final
    total = sum(stats.values())
    print(f"\n{'='*50}")
    print(f"  RESUMEN FINAL")
    print(f"{'='*50}")
    print(f"  Total procesadas : {total}")
    print(f"  Correctas        : {stats['ok']}  ({stats['ok']/total*100:.1f}%)")
    print(f"  Unknown/Error    : {stats['unknown'] + stats['error']}  "
          f"({(stats['unknown'] + stats['error'])/total*100:.1f}%)")
    print(f"\n  Resultados guardados en: {output_csv}")


# ── Entrada ───────────────────────────────────────────────────────────────────

def main():
    url = "C:/Users/adaxi/OneDrive/Escritorio/dataset/fuego_datatset/data/val"
    modelos = [
        "llava:7b",
        "internlm/interns1:mini-q8_0",
        "qwen3.5:latest",
    ]
    for model in modelos:
        model_name_safe = model.replace(":", "_").replace("/", "_").replace("\\", "_")
        analizar_directorio(
            url,
            output_csv=f"resultados_fuego_{model_name_safe}_es.csv",
            model=model,
        )


if __name__ == "__main__":
    main()