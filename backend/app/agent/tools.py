from langchain_core.tools import tool
from app.agent.state import MessagesState
from app.agent.vlm.vlm_analysis import analyze_vlm_data
from ddgs import DDGS
import base64
from app.storage.minio_client import download_media, get_media
from langchain_core.runnables import RunnableConfig
from app.agent.rag.vectorstore_instance import vectorstore
from app.storage.conversations import get_conversation_metadata
from typing import Annotated
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedState


@tool
def vlm_tool(
    media_id: str,
    question: str,
    state: Annotated[MessagesState, InjectedState],
    config: RunnableConfig,
) -> dict:
    """
    Analiza una imagen o vídeo específico que ya haya sido subido en esta conversación para responder a preguntas de seguimiento sobre su contenido.
    Utiliza esta herramienta cuando el usuario haga una pregunta de seguimiento sobre una imagen o vídeo que ya haya sido analizado.

    Argumentos:

    media_id: Identificador de la imagen o vídeo que se desea analizar, tal como aparece en el historial de la conversación 
              (por ejemplo, [Image attached, media_id=...]). Si solo hay una imagen en la conversación, utiliza esa.
    question: Pregunta específica del usuario sobre la imagen o vídeo. Siempre debes incluir este argumento. Nunca lo dejes vacío.

    Devuelve un diccionario con la respuesta del análisis realizado por el VLM.
    """
    available_media = state.get("available_media", {})

    if media_id not in available_media:
        return {
            "answer": f"No image or video found with media_id='{media_id}' in this conversation."
                      f"Available media_id values: {list(available_media.keys())}",
            "confidence": "unknown",
            "raw": "",
        }

    thread_id = config["configurable"]["thread_id"]
    media_type = available_media[media_id]
    
    storage_key = f"{media_id}_grid" if media_type == "video" else media_id
    media_bytes = get_media(thread_id, storage_key)

    if media_bytes is None:
        return {"answer": "No image or video associated with this media_id.", "confidence": "unknown", "raw": ""}
    image_b64 = base64.b64encode(media_bytes).decode("utf-8")

    return analyze_vlm_data(
        question=question,
        media_type=media_type,
        image_b64=image_b64,
        context="",
    )

@tool
def rag_tool(query: str, k: int = 4) -> dict:
    """Busca información en la base de conocimiento interna (normativa, glosario de categorías, papers)."""
    docs = vectorstore.similarity_search(query, k=k)
    return {
        "chunks": [d.page_content for d in docs],
        "sources": [d.metadata.get("source", "desconocida") for d in docs]
    }

@tool
def internet_tool(query: str) -> dict:
    """
    Busca información en Internet cuando el RAG no tiene cobertura suficiente.
    """

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))

    return {
        "results": [
            {
                "title": r["title"],
                "url": r["href"],
                "content": r["body"]
            }
            for r in results
        ]
    }

TOOLS = [vlm_tool, rag_tool, internet_tool]
tools_by_name = {tool.name: tool for tool in TOOLS}

