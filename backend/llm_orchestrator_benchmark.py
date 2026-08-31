"""
Benchmark de LLMs orquestadores para el agente HazardEx (LangGraph).

Compara distintos modelos servidos por Ollama (p. ej. qwen2.5:7b, llama3.1:8b,
mistral:7b, qwen2.5:14b si cabe en VRAM, etc.) sobre un conjunto de casos de
prueba fijo, midiendo:

  1. Precisión en la selección de la herramienta correcta (tool routing accuracy)
  2. Validez del formato de la tool call (JSON parseable / schema correcto)
  3. Calidad de la respuesta final (mediante LLM-juez + hueco para revisión humana)
  4. Latencia (tiempo hasta primer token de la tool call / respuesta completa)

Diseñado para funcionar 100% en local contra un servidor Ollama
(http://localhost:11434), evaluando un modelo cada vez para no saturar
una GPU de 12GB VRAM (ejecutar secuencialmente, no en paralelo).

USO:
    python llm_orchestrator_benchmark.py --models qwen2.5:7b llama3.1:8b mistral:7b
    python llm_orchestrator_benchmark.py --models qwen2.5:7b --judge qwen2.5:14b
    python llm_orchestrator_benchmark.py --list-cases

Salidas:
    results_raw.csv       -> una fila por (modelo, caso de prueba)
    results_summary.csv   -> una fila por modelo, métricas agregadas
    results_raw.jsonl     -> log completo (prompt, respuesta cruda, tool_calls, error)

Requisitos:
    pip install requests --break-system-packages
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
TIMEOUT_S = 120

# ---------------------------------------------------------------------------
# 1. Definición de herramientas (deben reflejar los tools reales del agente
#    LangGraph: vlm_tool, rag_tool, internet_tool). Ajusta los schemas si tu
#    implementación real difiere en nombres de parámetros.
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "vlm_tool",
            "description": (
                "Analiza una imagen o vídeo específico que ya haya sido subido en "
                "esta conversación para responder a preguntas de seguimiento sobre "
                "su contenido. Utiliza esta herramienta cuando el usuario haga una "
                "pregunta de seguimiento sobre una imagen o vídeo que ya haya sido "
                "analizado."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "media_id": {
                        "type": "string",
                        "description": (
                            "Identificador de la imagen o vídeo que se desea analizar, "
                            "tal como aparece en el historial de la conversación "
                            "(por ejemplo, [Image attached, media_id=...]). Si solo hay "
                            "una imagen en la conversación, utiliza esa."
                        ),
                    },
                    "question": {
                        "type": "string",
                        "description": (
                            "Pregunta específica del usuario sobre la imagen o vídeo. "
                            "Siempre debes incluir este argumento. Nunca lo dejes vacío."
                        ),
                    },
                },
                "required": ["media_id", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_tool",
            "description": (
                "Busca información en la base de conocimiento interna (normativa, "
                "glosario de categorías, papers)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Consulta en lenguaje natural sobre normativa, glosario o papers",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Número de fragmentos a recuperar (por defecto 4)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "internet_tool",
            "description": (
                "Busca información en Internet cuando el RAG no tiene cobertura suficiente."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Consulta de búsqueda web",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "Eres el agente conversacional de HazardEx, un sistema de detección de "
    "contenido potencialmente peligroso (violencia, armas, fuego/humo, "
    "accidentes de tráfico y desinformación). Tienes acceso a herramientas "
    "para analizar multimedia ya subida en la conversación, consultar tu base "
    "de conocimiento interna (normativa, glosario, papers) y buscar en "
    "internet cuando el RAG no tenga cobertura suficiente. Usa la herramienta "
    "adecuada cuando la pregunta lo requiera; si puedes responder directamente "
    "sin herramientas, hazlo."
)

# ---------------------------------------------------------------------------
# 2. Casos de prueba. `expected_tool` = None significa que lo correcto es
#    responder directamente sin invocar ninguna herramienta.
#    Añade/edita casos según los flujos reales que quieras validar.
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    id: str
    category: str  # "vlm" | "rag" | "internet" | "none" | "ambiguo"
    prompt: str
    expected_tool: Optional[str]
    notes: str = ""


TEST_CASES: list[TestCase] = [
    # --- vlm_tool: preguntas de SEGUIMIENTO sobre media ya subida en la conversación ---
    TestCase("vlm_1", "vlm", "¿Hay algún arma visible en el vídeo que acabo de subir?", "vlm_tool"),
    TestCase("vlm_2", "vlm", "Sobre la imagen que te mandé antes (media_id: img_042), ¿detectas fuego?", "vlm_tool"),
    TestCase("vlm_3", "vlm", "De ese accidente que aparece en el clip, ¿te parece grave?", "vlm_tool"),
    TestCase("vlm_4", "vlm", "¿Puedes contarme con más detalle qué se ve en el vídeo que adjunté?", "vlm_tool"),

    # --- rag_tool: normativa / glosario de categorías / papers internos indexados ---
    TestCase("rag_1", "rag", "¿Qué define exactamente el sistema como 'contenido violento' según el glosario de categorías?", "rag_tool"),
    TestCase("rag_2", "rag", "¿Qué normativa se usa para clasificar un cuchillo como arma blanca en el sistema?", "rag_tool"),
    TestCase("rag_3", "rag", "¿En qué paper se basa el criterio de detección de humo frente a niebla?", "rag_tool"),

    # --- internet_tool: información externa/actual que el RAG no cubre ---
    TestCase("net_1", "internet", "Busca noticias recientes sobre accidentes de tráfico graves en Extremadura.", "internet_tool"),
    TestCase("net_2", "internet", "¿Qué dice la última actualización de la ley de desinformación en la UE, publicada este mes?", "internet_tool"),
    TestCase("net_3", "internet", "¿Cuál es el precio actual de una cámara térmica para detección de incendios?", "internet_tool"),

    # --- none: no requiere herramienta ---
    TestCase("none_1", "none", "¿Qué categorías de contenido peligroso detecta HazardEx?", None),
    TestCase("none_2", "none", "Explícame brevemente cómo funciona el sistema de triage.", None),
    TestCase("none_3", "none", "Gracias, ¿puedes resumir en una frase lo que hemos hablado?", None),

    # --- ambiguo: casos límite pensados para estresar el routing ---
    TestCase(
        "amb_1", "ambiguo",
        "¿Este tipo de arma que sale en el vídeo es legal en España?",
        "vlm_tool",
        notes="Requiere primero identificar el arma en el vídeo (vlm_tool) antes de poder "
              "razonar sobre legalidad; un modelo débil podría saltar directo a internet_tool "
              "o rag_tool sin haber mirado el contenido real del vídeo.",
    ),
    TestCase(
        "amb_2", "ambiguo",
        "Según la normativa que consultaste antes, ¿este cuchillo del vídeo entraría en esa categoría?",
        "vlm_tool",
        notes="Menciona 'normativa' (podría tentar a rag_tool) pero el paso pendiente es "
              "identificar el objeto en el vídeo actual con vlm_tool antes de poder aplicar "
              "el criterio normativo.",
    ),
    TestCase(
        "amb_3", "ambiguo",
        "¿Qué dice el glosario del sistema sobre qué se considera un accidente 'grave'?",
        "rag_tool",
        notes="Suena a pregunta general pero pide explícitamente el glosario interno, no "
              "una búsqueda web ni un análisis de vídeo.",
    ),
]


# ---------------------------------------------------------------------------
# 3. Llamada al modelo vía Ollama
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    model: str
    case_id: str
    category: str
    expected_tool: Optional[str]
    called_tool: Optional[str]
    tool_call_raw: Optional[str]
    tool_json_valid: bool
    tool_correct: bool
    final_text: str
    latency_s: float
    error: Optional[str] = None
    quality_score: Optional[float] = None
    quality_reasoning: Optional[str] = None


def call_ollama(model: str, prompt: str) -> tuple[dict, float]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "tools": TOOLS,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    t0 = time.perf_counter()
    resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_S)
    latency = time.perf_counter() - t0
    resp.raise_for_status()
    return resp.json(), latency


def evaluate_case(model: str, case: TestCase) -> RunResult:
    try:
        data, latency = call_ollama(model, case.prompt)
    except Exception as e:  # noqa: BLE001 - queremos capturar cualquier fallo de red/modelo
        return RunResult(
            model=model, case_id=case.id, category=case.category,
            expected_tool=case.expected_tool, called_tool=None, tool_call_raw=None,
            tool_json_valid=False, tool_correct=False, final_text="",
            latency_s=-1.0, error=str(e),
        )

    message = data.get("message", {})
    tool_calls = message.get("tool_calls") or []
    final_text = message.get("content", "") or ""

    called_tool = None
    tool_call_raw = None
    tool_json_valid = False

    if tool_calls:
        first = tool_calls[0]
        fn = first.get("function", {})
        called_tool = fn.get("name")
        raw_args = fn.get("arguments")
        tool_call_raw = json.dumps(raw_args, ensure_ascii=False) if raw_args is not None else None
        # `arguments` puede venir ya como dict (Ollama lo parsea) o como string JSON.
        if isinstance(raw_args, dict):
            tool_json_valid = True
        elif isinstance(raw_args, str):
            try:
                json.loads(raw_args)
                tool_json_valid = True
            except json.JSONDecodeError:
                tool_json_valid = False

    tool_correct = called_tool == case.expected_tool

    return RunResult(
        model=model, case_id=case.id, category=case.category,
        expected_tool=case.expected_tool, called_tool=called_tool,
        tool_call_raw=tool_call_raw, tool_json_valid=tool_json_valid,
        tool_correct=tool_correct, final_text=final_text, latency_s=latency,
    )


# ---------------------------------------------------------------------------
# 4. Calidad de la respuesta final vía LLM-juez (opcional)
#    Limitación importante para la memoria del TFG: usar un modelo como juez
#    introduce sesgo (el juez puede favorecer estilos parecidos a sí mismo).
#    Se recomienda usar un modelo juez MÁS GRANDE que los evaluados y, sobre
#    una submuestra, validar manualmente el acuerdo juez-humano (Cohen's kappa
#    o simple % de acuerdo) para poder justificar la métrica en la memoria.
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """Vas a evaluar la respuesta de un asistente de un sistema de detección de contenido peligroso (HazardEx).

Pregunta del usuario:
{prompt}

¿Se esperaba que el asistente usara una herramienta? {expected_tool_desc}
Herramienta que efectivamente usó: {called_tool_desc}

Respuesta final del asistente:
{final_text}

Evalúa la respuesta SOLO en estos criterios, de 1 (muy mala) a 5 (excelente):
- Relevancia respecto a la pregunta
- Coherencia y claridad
- Concisión (sin relleno innecesario)

Responde EXCLUSIVAMENTE con un JSON válido, sin texto adicional, con este formato:
{{"score": <media de 1 a 5, puede tener decimales>, "reasoning": "<justificación breve en una frase>"}}
"""


def judge_case(judge_model: str, case: TestCase, result: RunResult) -> None:
    if result.error or not result.final_text.strip():
        return
    expected_desc = f"Sí, la herramienta '{case.expected_tool}'." if case.expected_tool else "No, se podía responder directamente."
    called_desc = result.called_tool if result.called_tool else "Ninguna."
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        prompt=case.prompt,
        expected_tool_desc=expected_desc,
        called_tool_desc=called_desc,
        final_text=result.final_text.strip(),
    )
    try:
        payload = {
            "model": judge_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_S)
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
        content = content.strip().strip("`").replace("json\n", "", 1) if content.startswith("```") else content
        parsed = json.loads(content)
        result.quality_score = float(parsed.get("score"))
        result.quality_reasoning = parsed.get("reasoning")
    except Exception as e:  # noqa: BLE001
        result.quality_reasoning = f"[error al juzgar: {e}]"


# ---------------------------------------------------------------------------
# 5. Orquestación del benchmark y agregación de métricas
# ---------------------------------------------------------------------------

def run_benchmark(models: list[str], cases: list[TestCase]) -> list[RunResult]:
    """Fase 1: recorre todos los modelos evaluados y todos los casos, SIN juzgar.

    Se hace en su propia fase (en vez de intercalar el juez caso a caso) para
    minimizar los cambios de modelo cargado en VRAM: si el juez y los modelos
    evaluados no caben juntos en memoria, Ollama tiene que descargar/recargar
    pesos en cada swap, lo que puede multiplicar el tiempo total del benchmark
    (decenas de swaps vs. 1-2 si el juicio se hace en bloque al final).
    """
    all_results: list[RunResult] = []
    for model in models:
        print(f"\n=== Evaluando modelo: {model} ===")
        for case in cases:
            print(f"  [{case.id}] {case.prompt[:60]}...", end=" ", flush=True)
            result = evaluate_case(model, case)
            all_results.append(result)
            status = "ERROR" if result.error else ("OK" if result.tool_correct else "FALLO ROUTING")
            print(f"-> {status} ({result.latency_s:.2f}s)")
    return all_results


def run_judging(judge_model: str, cases: list[TestCase], results: list[RunResult]) -> None:
    """Fase 2: carga el juez UNA sola vez y puntúa todos los resultados ya
    recogidos, en bloque. Reduce los swaps de modelo en VRAM a prácticamente 1.
    """
    cases_by_id = {c.id: c for c in cases}
    print(f"\n=== Juzgando respuestas con: {judge_model} ({len(results)} casos) ===")
    for i, result in enumerate(results, start=1):
        case = cases_by_id[result.case_id]
        print(f"  [{i}/{len(results)}] {result.model} / {case.id}...", end=" ", flush=True)
        judge_case(judge_model, case, result)
        score = result.quality_score if result.quality_score is not None else "N/A"
        print(f"-> score={score}")


def summarize(results: list[RunResult]) -> list[dict]:
    summary = []
    models = sorted(set(r.model for r in results))
    for model in models:
        subset = [r for r in results if r.model == model and not r.error]
        n = len(subset)
        errors = len([r for r in results if r.model == model and r.error])
        if n == 0:
            summary.append({"model": model, "n_ok": 0, "n_errors": errors})
            continue

        tool_cases = [r for r in subset if r.expected_tool is not None]
        tool_accuracy = (
            sum(1 for r in tool_cases if r.tool_correct) / len(tool_cases) * 100
            if tool_cases else float("nan")
        )
        called_any_tool = [r for r in subset if r.called_tool is not None]
        json_validity = (
            sum(1 for r in called_any_tool if r.tool_json_valid) / len(called_any_tool) * 100
            if called_any_tool else float("nan")
        )
        overall_routing_accuracy = sum(1 for r in subset if r.tool_correct) / n * 100
        latencies = [r.latency_s for r in subset if r.latency_s >= 0]
        quality_scores = [r.quality_score for r in subset if r.quality_score is not None]

        summary.append({
            "model": model,
            "n_ok": n,
            "n_errors": errors,
            "tool_selection_accuracy_%": round(tool_accuracy, 1) if tool_accuracy == tool_accuracy else None,
            "overall_routing_accuracy_%": round(overall_routing_accuracy, 1),
            "tool_json_validity_%": round(json_validity, 1) if json_validity == json_validity else None,
            "avg_latency_s": round(statistics.mean(latencies), 2) if latencies else None,
            "p95_latency_s": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 2) if len(latencies) >= 2 else None,
            "avg_quality_score_1_5": round(statistics.mean(quality_scores), 2) if quality_scores else None,
        })
    return summary


def write_outputs(results: list[RunResult], summary: list[dict]) -> None:
    with open("results_raw.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    with open("results_raw.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()) if results else [])
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    with open("results_summary.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary[0].keys()) if summary else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

    print("\n=== RESUMEN ===")
    for row in summary:
        print(row)
    print("\nArchivos generados: results_raw.jsonl, results_raw.csv, results_summary.csv")


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark de LLMs orquestadores para HazardEx (vía Ollama).")
    parser.add_argument("--models", nargs="+", default=["qwen2.5:7b"],
                         help="Modelos Ollama a evaluar, ej: qwen2.5:7b llama3.1:8b mistral:7b")
    parser.add_argument("--judge", default=None,
                         help="Modelo Ollama a usar como juez de calidad de respuesta (opcional). "
                              "Recomendado: un modelo más grande que los evaluados. Se ejecuta en "
                              "una segunda fase, después de evaluar todos los modelos, para cargarse "
                              "en VRAM una sola vez y no alternar con los modelos evaluados caso a caso.")
    parser.add_argument("--list-cases", action="store_true", help="Lista los casos de prueba y sale.")
    args = parser.parse_args()

    if args.list_cases:
        for c in TEST_CASES:
            print(f"[{c.category:8s}] {c.id:8s} expected={c.expected_tool} :: {c.prompt}")
        return

    results = run_benchmark(args.models, TEST_CASES)
    if args.judge:
        run_judging(args.judge, TEST_CASES, results)
    summary = summarize(results)
    write_outputs(results, summary)


if __name__ == "__main__":
    main()