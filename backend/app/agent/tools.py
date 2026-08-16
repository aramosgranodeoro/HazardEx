from langchain_core.tools import tool
from app.agent.vlm.vlm_analysis import analyze_vlm_data
from ddgs import DDGS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

INDEX_FOLDER = "./app/agent/rag/chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small",
    model_kwargs={"device": "cpu"}
)

vectorstore = Chroma(
    persist_directory=INDEX_FOLDER,
    embedding_function=embeddings
)

@tool
def vlm_tool(question: str, media_type: str, image_b64: str = "", context: str = "") -> dict:
    """
    Analyzes an image or video using a Vision-Language Model (VLM)
    to answer questions about its visual content.

    Use this tool when the user asks a follow-up question about the
    image/video that was already analyzed (e.g., "what color are the
    clothes?", "how many people are there?", "describe the scene in
    more detail").

    Args:
        question: The user's specific question about the image/video.
            You MUST always include this argument with the user's exact
            or rephrased question. Never leave it empty.
        media_type: Type of media to analyze. Must be "image" or "video".
        image_b64: Base64-encoded image, if available in the conversation
            context. Leave empty if you don't have it.
        context: Additional relevant context from the conversation (for
            example, the result of the initial analysis already performed).
            Leave empty if not applicable.

    Returns:
        A dictionary with the VLM analysis response.
    """
    if not question:
        question = "Describe in detail what is observed in the image/video."
    return analyze_vlm_data(question, media_type, image_b64, context)

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

