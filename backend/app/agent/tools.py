from langchain_core.tools import tool
from app.agent.vlm.vlm_analysis import analyze_vlm_data
from ddgs import DDGS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import base64
from app.storage.minio_client import download_media
from langchain_core.runnables import RunnableConfig
from app.agent.rag.vectorstore_instance import vectorstore
from app.storage.conversations import get_conversation_metadata

# INDEX_FOLDER = "./app/agent/rag/chroma_db"

# embeddings = HuggingFaceEmbeddings(
#     model_name="intfloat/multilingual-e5-small",
#     model_kwargs={"device": "cpu"}
# )

# vectorstore = Chroma(
#     persist_directory=INDEX_FOLDER,
#     embedding_function=embeddings
# )

@tool
def vlm_tool(question: str, config: RunnableConfig) -> dict:
    """
    Analyzes the image or video that was already uploaded and analyzed
    in this conversation, to answer follow-up questions about its content.

    Use this tool when the user asks a follow-up question about the
    image/video already analyzed.

    Args:
        question: The user's specific question about the image/video.
            You MUST always include this argument. Never leave it empty.

    Returns:
        A dictionary with the VLM analysis response.
    """
    thread_id = config["configurable"]["thread_id"]

    media_bytes = download_media(thread_id)
    if media_bytes is None:
        return {"answer": "No hay imagen o vídeo asociado a esta conversación.", "confidence": "unknown", "raw": ""}

    metadata = get_conversation_metadata(thread_id) or {"media_type": "photo"}
    image_b64 = base64.b64encode(media_bytes).decode("utf-8")

    return analyze_vlm_data(
        question=question,
        media_type=metadata["media_type"],
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
        results = list(ddgs.text(query, max_results=3))

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

