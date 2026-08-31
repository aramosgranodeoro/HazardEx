import ollama
import base64
import io
import csv
import json
from datasets import load_dataset


def pil_to_base64(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def analyze_chart(pil_image, query, model):
    img_b64 = pil_to_base64(pil_image)
    response = ollama.chat(
        model=model,
        options={"temperature": 0.1, "num_predict": 4096, "think": False},
        messages=[{
            'role': 'user',
            'content': f"""
            # ROLE
            You are a chart analyst. Do not think out loud. Do not use <tool_call> tags. Respond ONLY with the requested JSON.

            # OBJECTIVE
            Answer the following question about the chart image.
            
            # QUESTION
            {query}

            # OUTPUT FORMAT
            Respond ONLY with valid JSON:
            {{
                "answer": "your answer here",
                "confidence": "a number between 0 and 1",
                "reasoning": "one sentence explaining your reasoning"
            }}
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


def parsear_respuesta(raw_content):
    try:
        # Limpia bloques ```json ... ```
        clean = raw_content.strip()
        if "```" in clean:
            clean = clean.split("```")[1].removeprefix("json").strip()
        
        data = json.loads(clean)

        # A veces el modelo usa claves distintas a "answer"
        for key in ["answer", "Answer", "ANSWER", "result", "value", "response"]:
            if key in data:
                return str(data[key]).strip()

        # Si no encuentra ninguna clave conocida, devuelve el primer valor del JSON
        first_value = next(iter(data.values()))
        return str(first_value).strip()

    except (json.JSONDecodeError, StopIteration):
        # Si no es JSON, intenta extraer la respuesta del texto libre
        # Busca patrones como "answer: X" o "the answer is X"
        import re
        patrones = [
            r'"answer"\s*:\s*"?([^",}\n]+)"?',
            r'answer is[:\s]+([^\.\n]+)',
            r'result[:\s]+([^\.\n]+)',
        ]
        for patron in patrones:
            match = re.search(patron, raw_content, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Último recurso: devuelve el texto completo
        return raw_content.strip()


def normalizar(texto):
    import re
    
    texto = texto.lower().strip()
    
    # Quita símbolos de % y $ y espacios extra
    texto = texto.replace("%", "").replace("$", "").replace(",", ".").strip()
    
    # Convierte fracción "1/2" -> float
    fraccion = re.match(r'^(-?\d+)\s*/\s*(-?\d+)$', texto)
    if fraccion:
        try:
            texto = str(round(int(fraccion.group(1)) / int(fraccion.group(2)), 6))
        except ZeroDivisionError:
            pass
    
    try:
        texto = str(float(texto))          
    except ValueError:
        pass
    
    return texto

def es_correcto(respuesta_modelo, label):
    norm_modelo = normalizar(respuesta_modelo)
    norm_label  = normalizar(label)
    
    if norm_modelo == norm_label:
        return 1
    
    # Compara como float con tolerancia por redondeo (87.5 == 87.50)
    try:
        if abs(float(norm_modelo) - float(norm_label)) < 1e-4:
            return 1
    except ValueError:
        pass
    
    return 0


def analizar_dataset(output_csv="resultados_llava.csv", model="llava:7b"):
    print("Loading dataset (streaming)...")
    dataset = load_dataset("HuggingFaceM4/ChartQA", split="test", streaming=True)

    campos = ["indice", "query", "label", "respuesta_modelo", "correcto", "total_ms", "prompt_ms", "eval_ms", "prompt_tokens", "generated_tokens", "tokens_per_sec"]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()

        for i, ejemplo in enumerate(dataset):
            pil_image = ejemplo["image"]
            query     = ejemplo["query"]
            label     = ejemplo["label"][0]

            print(f"\n[{i+1}] Pregunta: {query}")
            print(f"     Respuesta correcta: {label}")

            try:
                response, metricas      = analyze_chart(pil_image, query, model)
                raw_content   = response['message']['content']
                resp_modelo   = parsear_respuesta(raw_content)
                correcto      = es_correcto(resp_modelo, label)
            except Exception as e:
                print(f"     ⚠️  Error en ejemplo {i+1}: {e}")
                resp_modelo = "ERROR"
                correcto    = 0

            print(f"     Respuesta modelo: {resp_modelo} | Correcto: {correcto}")

            writer.writerow({
                "indice":           i + 1,
                "query":            query,
                "label":            label,
                "respuesta_modelo": resp_modelo,
                "correcto":         correcto,
                "total_ms":           metricas.get("total_ms", 0),
                "prompt_ms":          metricas.get("prompt_ms", 0),
                "eval_ms":            metricas.get("eval_ms", 0),
                "prompt_tokens":      metricas.get("prompt_tokens", 0),
                "generated_tokens":   metricas.get("generated_tokens", 0),
                "tokens_per_sec":     metricas.get("tokens_per_sec", 0)
            })
            f.flush()  # escribe en disco tras cada fila por si se interrumpe

    print(f"\n✅ Análisis completado. Resultados guardados en '{output_csv}'")


if __name__ == "__main__":
    modelos = [
        "llava:7b",
        "internlm/interns1:mini-q8_0",
        "qwen3.5:latest"
    ]
    for model in modelos:
        model_name_safe = model.replace(":", "_").replace("/", "_").replace("\\", "_")
        analizar_dataset(output_csv=f"resultados_chartsQA_{model_name_safe}.csv", model=model)