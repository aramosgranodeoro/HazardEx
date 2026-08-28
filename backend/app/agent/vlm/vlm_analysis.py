import ollama
import base64
import json
import re

# ---------------------------------------------------------------------------
# Prompts según el tipo de multimedia (foto o vídeo). 
# ---------------------------------------------------------------------------

PROMPT_PHOTO =  """
                # ROLE
                You are an analysis assistant for images.

                # CONTEXT
                {context}

                # OBJECTIVE
                Answer the user's question about the image below, using the context if relevant.
                The image is a single photograph.

                # QUESTION
                {question}

                # OUTPUT FORMAT
                Respond ONLY with valid JSON, no preamble, no markdown fences:
                {{
                "answer": "your answer here",
                "confidence": "high | medium | low"
                }}
                """

PROMPT_VIDEO = """
               # ROLE
                You are an analysis assistant for video content.

                # CONTEXT
                {context}

                # INPUT FORMAT
                The image you are given is NOT a single photo. It is a frame-grid: multiple
                frames extracted from a video clip, arranged in a grid, each with a timestamp
                overlay. Read the grid left-to-right, top-to-bottom as the chronological
                sequence of the video. Use the temporal order and timestamps to reason about
                what happens over time, not just what appears in a single frame.

                # OBJECTIVE
                Answer the user's question about the video below, using the context if
                relevant, and taking into account how the scene evolves across frames.

                # QUESTION
                {question}

                # OUTPUT FORMAT
                Respond ONLY with valid JSON, no preamble, no markdown fences:
                {{
                "answer": "your answer here",
                "confidence": "high | medium | low"
                }}
                """

def _parse_vlm_json(raw_text: str) -> dict:
    """
    Parser robusto para la salida del VLM, siguiendo el mismo patrón
    que el resto del pipeline (extracción por llaves balanceadas,
    limpieza de fences markdown, fallback por regex campo a campo).
    """
    text = raw_text.strip()

    # 1. Quitar posibles tags <think>...</think>
    text = re.sub(r"</?think>", "", text).strip()

    # 2. Quitar fences markdown ```json ... ```
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    # 3. Intento directo
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 4. Extracción por llaves balanceadas (primer { ... } completo)
    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    # 5. Fallback: regex campo a campo
    answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', text)
    confidence_match = re.search(r'"confidence"\s*:\s*"([^"]*)"', text)

    return {
        "answer": answer_match.group(1) if answer_match else text[:500],
        "confidence": confidence_match.group(1) if confidence_match else "unknown",
        "parse_warning": "Fallback parsing used, JSON was not well-formed",
    }

def _select_prompt(media_type: str, question: str, context: str) -> str:
    template = PROMPT_VIDEO if media_type == "video" else PROMPT_PHOTO
    return template.format(
        context=context if context else "No additional context provided.",
        question=question,
    )

def analyze_vlm_data(question: str, media_type: str, image_b64: str = "", context: str = "") -> dict:
    """
    Responde a una pregunta del usuario sobre una foto o vídeo ya
    procesado, usando el contexto generado por el agente.

    Args:
        question: Pregunta del usuario sobre el medio.
        image_b64: Imagen codificada en base64. Si media_type es
                    "video", debe ser el frame-grid con timestamps generado
                    a partir del clip.
        media_type: "photo" o "video". Determina el prompt usado: para
                    "video" se instruye al modelo a interpretar la imagen
                    como una secuencia temporal de frames, no como una
                    foto única.
        context: Contexto adicional (p.ej. salida JSON de un adaptador
                 especialista previo) para fundamentar la respuesta.

    Returns:
        dict con {"answer": str, "confidence": str, "raw": str}
    """
    
    prompt = _select_prompt(media_type, question, context)

    response = ollama.chat(
        model="llava:7b",
        options={"temperature": 0.1},
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [image_b64],
        }],
    )

    raw_content = response["message"]["content"]
    parsed = _parse_vlm_json(raw_content)

    return {
        "answer": parsed.get("answer", ""),
        "confidence": parsed.get("confidence", "unknown"),
        "raw": raw_content,
    }


