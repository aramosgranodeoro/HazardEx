from langchain_core.tools import tool
from triaje_multimodelo import ejecutar_triaje
from rag_index import vectorstore
import requests

@tool
def vlm_tool(image_path: str, categoria: str = None) -> dict:
    """Analiza una imagen con el modelo VLM especialista (violencia, armas, fuego, accidente, gráfico)."""
    resultado = ejecutar_triaje(image_path, categoria_forzada=categoria)
    return {
        "predicted_category": resultado["predicted_category"],
        "confidence": resultado["confidence"],
        "description": resultado["description"]
    }

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
    """Busca en internet cuando el RAG no tiene cobertura suficiente."""
    resp = requests.get("https://api.tavily.com/search", params={"query": query, "api_key": "..."})
    data = resp.json()
    return {"results": [r["content"] for r in data.get("results", [])[:3]]}

TOOLS = [vlm_tool, rag_tool, internet_tool]
tools_by_name = {tool.name: tool for tool in TOOLS}

