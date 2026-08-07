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
    """Analiza una imagen o vídeo ."""
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

