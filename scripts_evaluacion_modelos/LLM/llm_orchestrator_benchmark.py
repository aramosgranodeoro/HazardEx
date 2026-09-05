"""
Benchmark de LLMs orquestadores para el agente HazardEx (LangGraph).

Compara distintos modelos servidos por Ollama sobre un conjunto de casos de
prueba fijo, midiendo:

  1. Precisión en la selección de la herramienta correcta (tool routing accuracy)
  2. Validez del formato de la tool call (JSON parseable / schema correcto)
  3. Latencia (tiempo hasta primer token de la tool call / respuesta completa)

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
TIMEOUT_S = 20

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
    """
    Eres HazardEx, un asistente especializado en moderacion de
contenido. Tu unico proposito es ayudar a analizar y debatir
sobre material multimedia (imagenes o videos) para identificar
contenido peligroso dentro de las siguientes categorias:
violencia, armas, incendios, accidentes de trafico y
desinformacion o noticias falsas.
# HERRAMIENTAS
Tienes acceso a las siguientes herramientas:
- vlm_tool: permite analizar contenido multimedia.
- rag_tool: permite consultar documentacion de referencia.
- internet_tool: permite realizar busquedas en Internet.
# AMBITO DE ACTUACION
Si el usuario pregunta acerca de un tema no relacionado con las
categorias de riesgo anteriores o con el contenido multimedia
analizado, indica que eres un asistente especializado en
moderacion y que no puedes proporcionar asistencia fuera de
dicho ambito.
# INSTRUCCIONES
- Utiliza una herramienta cuando resulte necesaria para
resolver la consulta.
- Cuando sea necesario buscar informacion, consulta primero
la base de conocimiento interna (rag_tool) y, si no hay
informacion suficiente, busca en Internet (internet_tool).
- Si la consulta requiere analizar nuevamente el contenido
multimedia, utiliza vlm_tool.
- En caso de incertidumbre, basa la respuesta en la informacion
disponible en el contexto o en las fuentes recuperadas.
spondiente a partir del contexto de la conversacion.
- Proporciona respuestas claras, profesionales y contextualizadas
sobre el contenido y sus posibles riesgos"""
)

# ---------------------------------------------------------------------------
# 2. Casos de prueba. `expected_tool` = None significa que lo correcto es
#    responder directamente sin invocar ninguna herramienta.
#    Añade/edita casos según los flujos reales que quieras validar.
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    id: str
    category: str  # "vlm" | "rag" | "internet" | "none"
    prompt: str
    expected_tool: Optional[str]
    notes: str = ""


TEST_CASES = [
    # --- vlm_tool: preguntas de SEGUIMIENTO sobre media ya subida en la conversación ---
    TestCase("vlm_1", "vlm", "¿Hay algún arma visible en el vídeo que acabo de subir?", "vlm_tool"),
    TestCase("vlm_2", "vlm", "Sobre la imagen que te mandé antes (media_id: img_042), ¿detectas fuego?", "vlm_tool"),
    TestCase("vlm_3", "vlm", "De ese accidente que aparece en el clip, ¿te parece grave?", "vlm_tool"),
    TestCase("vlm_4", "vlm", "¿Puedes contarme con más detalle qué se ve en el vídeo que adjunté?", "vlm_tool"),
    TestCase("vlm_5", "vlm", "¿Qué elementos se pueden ver en la imagen que subí hace un momento?", "vlm_tool"),
    TestCase("vlm_6", "vlm", "¿En la imagen que me enviaste, se ve algún objeto peligroso?", "vlm_tool"),
    TestCase("vlm_7", "vlm", "¿Puedes revisar otra vez el vídeo y decirme si aparece humo?", "vlm_tool"),
    TestCase("vlm_8", "vlm", "En la imagen anterior, ¿cuántas personas aparecen aproximadamente?", "vlm_tool"),
    TestCase("vlm_9", "vlm", "¿Se aprecia alguna colisión entre vehículos en el vídeo que subí?", "vlm_tool"),
    TestCase("vlm_10", "vlm", "Analiza con más detalle la zona de la imagen donde aparece el posible arma.", "vlm_tool"),
    TestCase("vlm_11", "vlm", "Vuelve a revisar la imagen y dime si ves llamas claramente.", "vlm_tool"),
    TestCase("vlm_12", "vlm", "¿Puedes comprobar si la persona del vídeo está sujetando algún cuchillo?", "vlm_tool"),
    TestCase("vlm_13", "vlm", "En el vídeo anterior, ¿hay algún momento en el que los vehículos lleguen a impactar?", "vlm_tool"),
    TestCase("vlm_14", "vlm", "¿La escena de la imagen parece violenta o simplemente una discusión?", "vlm_tool"),
    TestCase("vlm_15", "vlm", "Mira otra vez el contenido que subí y dime si aparece una pistola.", "vlm_tool"),
    TestCase("vlm_16", "vlm", "¿Puedes describir qué ocurre justo antes del posible accidente del vídeo?", "vlm_tool"),
    TestCase("vlm_17", "vlm", "En la imagen anterior, ¿el humo parece proceder de algún incendio?", "vlm_tool"),
    TestCase("vlm_18", "vlm", "¿Hay alguna persona herida visible en el vídeo que te envié?", "vlm_tool"),
    TestCase("vlm_19", "vlm", "¿Puedes identificar cuántos vehículos aparecen en el clip anterior?", "vlm_tool"),
    TestCase("vlm_20", "vlm", "Revisa la imagen anterior y dime si el objeto señalado podría ser un arma blanca.", "vlm_tool"),
    TestCase("vlm_21", "vlm", "¿Se observa fuego en alguna parte del fotograma que subí?", "vlm_tool"),
    TestCase("vlm_22", "vlm", "En el vídeo que analizaste antes, ¿se ve alguna pelea entre personas?", "vlm_tool"),
    TestCase("vlm_23", "vlm", "¿Puedes decirme si el coche de la izquierda llega a chocar con otro vehículo?", "vlm_tool"),
    TestCase("vlm_24", "vlm", "Analiza de nuevo la imagen anterior y dime qué objeto peligroso aparece, si hay alguno.", "vlm_tool"),
    TestCase("vlm_25", "vlm", "¿Qué sucede exactamente en los últimos segundos del vídeo que subí?", "vlm_tool"),
    TestCase("vlm_26", "vlm", "¿La imagen anterior muestra humo, niebla o no puedes distinguirlo?", "vlm_tool"),
    TestCase("vlm_27", "vlm", "¿Puedes comprobar si alguna persona aparece agrediendo físicamente a otra?", "vlm_tool"),
    TestCase("vlm_28", "vlm", "En el contenido multimedia anterior, ¿hay señales visibles de un accidente?", "vlm_tool"),
    TestCase("vlm_29", "vlm", "¿Puedes fijarte en el fondo de la imagen y decirme si aparece fuego?", "vlm_tool"),
    TestCase("vlm_30", "vlm", "Revisa otra vez el vídeo anterior y dime si detectas alguna situación de riesgo adicional.", "vlm_tool"),

    # --- rag_tool: normativa / glosario de categorías / papers internos indexados ---
    TestCase("rag_1", "rag", "¿Qué define exactamente el sistema como 'contenido violento' según el glosario de categorías?", "rag_tool"),
    TestCase("rag_2", "rag", "¿Qué normativa se usa para clasificar un cuchillo como arma blanca en el sistema?", "rag_tool"),
    TestCase("rag_3", "rag", "¿En qué paper se basa el criterio de detección de humo frente a niebla?", "rag_tool"),
    TestCase("rag_4", "rag", "Según la normativa interna, ¿qué se considera un accidente de tráfico 'grave'?", "rag_tool"),
    TestCase("rag_5", "rag", "¿Qué dice el glosario sobre la categoría de 'desinformación'?", "rag_tool"),
    TestCase("rag_6", "rag", "¿Qué criterios internos se usan para clasificar un vídeo como 'contenido violento'?", "rag_tool"),
    TestCase("rag_7", "rag", "Según la documentación del sistema, ¿qué diferencias hay entre fuego y humo como categorías de detección?", "rag_tool"),
    TestCase("rag_8", "rag", "¿Qué documentación interna explica cómo se clasifican las armas de fuego?", "rag_tool"),
    TestCase("rag_9", "rag", "¿Qué criterios recoge la base de conocimiento para identificar desinformación?", "rag_tool"),
    TestCase("rag_10", "rag", "¿Qué documento indexado describe el procedimiento de actuación ante un incendio?", "rag_tool"),
    TestCase("rag_11", "rag", "Según los documentos internos, ¿cómo se distingue una agresión física de una interacción no violenta?", "rag_tool"),
    TestCase("rag_12", "rag", "¿Qué fuente de la base de conocimiento define el concepto de arma blanca?", "rag_tool"),
    TestCase("rag_13", "rag", "¿Qué documentación recoge las recomendaciones de actuación ante humo en interiores?", "rag_tool"),
    TestCase("rag_14", "rag", "Según el material indexado, ¿qué factores permiten determinar la gravedad de un accidente?", "rag_tool"),
    TestCase("rag_15", "rag", "¿Cómo define la documentación interna una noticia potencialmente desinformativa?", "rag_tool"),
    TestCase("rag_16", "rag", "¿Qué criterios documentales utiliza HazardEx para identificar situaciones de violencia?", "rag_tool"),
    TestCase("rag_17", "rag", "¿Qué guía interna explica cómo actuar si se detecta un arma?", "rag_tool"),
    TestCase("rag_18", "rag", "Según la base de conocimiento, ¿qué recomendaciones deben seguirse ante un incendio doméstico?", "rag_tool"),
    TestCase("rag_19", "rag", "¿Qué información indexada hay sobre accidentes de tráfico con heridos?", "rag_tool"),
    TestCase("rag_20", "rag", "¿Qué documento del RAG explica los indicadores habituales de desinformación?", "rag_tool"),
    TestCase("rag_21", "rag", "Busca en la documentación interna la definición utilizada para clasificar una pelea como violencia.", "rag_tool"),
    TestCase("rag_22", "rag", "¿Qué normativa recogida en la base de conocimiento regula la posesión de armas blancas?", "rag_tool"),
    TestCase("rag_23", "rag", "Según los documentos indexados, ¿cómo debe actuar una persona si detecta humo en un edificio?", "rag_tool"),
    TestCase("rag_24", "rag", "¿Qué documentación del sistema establece las diferencias entre accidente leve, grave y muy grave?", "rag_tool"),
    TestCase("rag_25", "rag", "¿Qué fuentes internas utiliza el sistema para evaluar contenido posiblemente manipulado?", "rag_tool"),
    TestCase("rag_26", "rag", "¿Qué recomendaciones aparecen en la base de conocimiento ante una agresión física?", "rag_tool"),
    TestCase("rag_27", "rag", "¿Qué documento explica las distintas categorías de armas contempladas por HazardEx?", "rag_tool"),
    TestCase("rag_28", "rag", "¿Según el material indexado, cuáles son las principales señales visuales de un incendio?", "rag_tool"),
    TestCase("rag_29", "rag", "¿Qué criterios internos se emplean para considerar peligrosa una colisión de tráfico?", "rag_tool"),
    TestCase("rag_30", "rag", "¿Qué documentación interna recoge recomendaciones para verificar información antes de considerarla fiable?", "rag_tool"),

    # --- internet_tool: información externa/actual que el RAG no cubre ---
    TestCase("net_1", "internet", "Busca noticias recientes sobre accidentes de tráfico graves en Extremadura.", "internet_tool"),
    TestCase("net_2", "internet", "¿Qué dice la última actualización de la ley de desinformación en la UE, publicada este mes?", "internet_tool"),
    TestCase("net_3", "internet", "¿Cuál es el precio actual de una cámara térmica para detección de incendios?", "internet_tool"),
    TestCase("net_4", "internet", "¿Qué información disponible en línea hay sobre las últimas tendencias en seguridad vial?", "internet_tool"),
    TestCase("net_5", "internet", "Busca artículos recientes sobre la legalidad de armas de fuego en España.", "internet_tool"),
    TestCase("net_6", "internet", "¿Qué dice la prensa sobre el último accidente de tráfico grave en Madrid?", "internet_tool"),
    TestCase("net_7", "internet", "Busca información reciente sobre incendios forestales activos en España.", "internet_tool"),
    TestCase("net_8", "internet", "¿Cuáles son las últimas recomendaciones publicadas sobre prevención de incendios domésticos?", "internet_tool"),
    TestCase("net_9", "internet", "Busca noticias recientes relacionadas con desinformación generada mediante inteligencia artificial.", "internet_tool"),
    TestCase("net_10", "internet", "¿Qué novedades recientes hay sobre sistemas automáticos de detección de armas mediante visión artificial?", "internet_tool"),
    TestCase("net_11", "internet", "Busca los accidentes de tráfico más recientes ocurridos hoy en España.", "internet_tool"),
    TestCase("net_12", "internet", "¿Qué nuevas medidas ha anunciado recientemente la DGT para reducir accidentes?", "internet_tool"),
    TestCase("net_13", "internet", "Busca información actual sobre incendios forestales en Extremadura.", "internet_tool"),
    TestCase("net_14", "internet", "¿Qué incendios permanecen activos actualmente en territorio español?", "internet_tool"),
    TestCase("net_15", "internet", "Busca las últimas noticias sobre restricciones relacionadas con armas blancas en España.", "internet_tool"),
    TestCase("net_16", "internet", "¿Ha habido recientemente algún cambio legislativo sobre armas de fuego en España?", "internet_tool"),
    TestCase("net_17", "internet", "Busca publicaciones recientes sobre sistemas de detección automática de violencia en vídeo.", "internet_tool"),
    TestCase("net_18", "internet", "¿Qué avances recientes existen en detección de incendios mediante inteligencia artificial?", "internet_tool"),
    TestCase("net_19", "internet", "Busca investigaciones publicadas recientemente sobre detección de accidentes de tráfico con visión artificial.", "internet_tool"),
    TestCase("net_20", "internet", "¿Qué noticias recientes hay sobre campañas de desinformación en redes sociales?", "internet_tool"),
    TestCase("net_21", "internet", "Busca recomendaciones actuales de organismos oficiales sobre qué hacer ante un incendio forestal.", "internet_tool"),
    TestCase("net_22", "internet", "¿Cuáles son las estadísticas más recientes de accidentes de tráfico en España?", "internet_tool"),
    TestCase("net_23", "internet", "Busca información reciente sobre tecnologías utilizadas para detectar armas en espacios públicos.", "internet_tool"),
    TestCase("net_24", "internet", "¿Qué nuevos estudios se han publicado sobre detección de humo mediante cámaras?", "internet_tool"),
    TestCase("net_25", "internet", "Busca noticias de esta semana sobre accidentes múltiples en carreteras españolas.", "internet_tool"),
    TestCase("net_26", "internet", "¿Qué recomendaciones actuales existen para identificar contenido falso generado por IA?", "internet_tool"),
    TestCase("net_27", "internet", "Busca información publicada este año sobre modelos de visión para detectar peleas.", "internet_tool"),
    TestCase("net_28", "internet", "¿Qué sistemas comerciales recientes existen para detectar incendios mediante cámaras?", "internet_tool"),
    TestCase("net_29", "internet", "Busca las últimas novedades sobre regulación europea frente a la desinformación digital.", "internet_tool"),
    TestCase("net_30", "internet", "¿Qué avances recientes hay en modelos multimodales aplicados a la detección de situaciones de riesgo?", "internet_tool"),

    # --- none: no requiere herramienta ---
    TestCase("none_1", "none", "¿Qué categorías de contenido peligroso detecta HazardEx?", None),
    TestCase("none_2", "none", "Explícame brevemente cómo funciona el sistema de triage.", None),
    TestCase("none_3", "none", "Gracias, ¿puedes resumir en una frase lo que hemos hablado?", None),
    TestCase("none_4", "none", "¿Me puedes explicar otra vez lo que me has dicho?", None),
    TestCase("none_5", "none", "¿Me puedes explicar de otra forma lo que me has dicho antes?", None),
    TestCase("none_6", "none", "¿Qué son las chocolatinas?", None),
    TestCase("none_7", "none", "¿Para qué sirve el módulo de triaje de HazardEx?", None),
    TestCase("none_8", "none", "Resume la respuesta anterior de forma más sencilla.", None),
    TestCase("none_9", "none", "¿Qué diferencia hay entre analizar una imagen y un vídeo en HazardEx?", None),
    TestCase("none_10", "none", "¿Puedes responderme de forma más breve?", None),
    TestCase("none_11", "none", "¿Cuáles son los cinco tipos de riesgo que contempla el sistema?", None),
    TestCase("none_12", "none", "¿Qué hace HazardEx cuando el usuario sube una imagen?", None),
    TestCase("none_13", "none", "¿Qué hace HazardEx cuando se le proporciona un vídeo?", None),
    TestCase("none_14", "none", "¿Para qué se utilizan los modelos especializados dentro del sistema?", None),
    TestCase("none_15", "none", "¿Qué función tiene el agente dentro de HazardEx?", None),
    TestCase("none_16", "none", "Explícame qué significa triage en este sistema.", None),
    TestCase("none_17", "none", "¿El sistema puede analizar tanto imágenes como vídeos?", None),
    TestCase("none_18", "none", "¿Qué categorías están relacionadas con contenido visual estático?", None),
    TestCase("none_19", "none", "¿Qué categorías se analizan principalmente a partir de vídeos?", None),
    TestCase("none_20", "none", "¿Cuál es el objetivo principal de HazardEx?", None),
    TestCase("none_21", "none", "Dímelo de una forma más sencilla.", None),
    TestCase("none_22", "none", "Hazme un resumen breve de la explicación anterior.", None),
    TestCase("none_23", "none", "¿Puedes explicarlo con menos tecnicismos?", None),
    TestCase("none_24", "none", "¿Puedes ponerme un ejemplo sencillo de cómo funciona HazardEx?", None),
    TestCase("none_25", "none", "¿Qué significa que el sistema sea multimodal?", None),
    TestCase("none_26", "none", "¿Por qué HazardEx utiliza distintos modelos en lugar de uno solo?", None),
    TestCase("none_27", "none", "¿Qué ocurre después de que el módulo de triaje clasifica una entrada?", None),
    TestCase("none_28", "none", "¿Puede HazardEx mantener una conversación sobre un análisis previo?", None),
    TestCase("none_29", "none", "¿Qué papel tienen las herramientas dentro del agente?", None),
    TestCase("none_30", "none", "Resume en dos frases cómo está organizado el sistema.", None),
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
        "options": {"temperature": 0, "num_ctx": 4096},
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
    parser.add_argument("--list-cases", action="store_true", help="Lista los casos de prueba y sale.")
    args = parser.parse_args()

    if args.list_cases:
        for c in TEST_CASES:
            print(f"[{c.category:8s}] {c.id:8s} expected={c.expected_tool} :: {c.prompt}")
        return

    results = run_benchmark(args.models, TEST_CASES)
    summary = summarize(results)
    write_outputs(results, summary)


if __name__ == "__main__":
    main()