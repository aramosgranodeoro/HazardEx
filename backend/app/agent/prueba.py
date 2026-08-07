# Step 3: Define model node (VERSIÓN CON TOOLS MOCKEADAS PARA VALIDAR EL FLUJO)
import operator
import os
import time
from typing import Literal

from ddgs import DDGS
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, ToolMessage, AnyMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, START, StateGraph
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from typing_extensions import TypedDict, Annotated
from IPython.display import Image, display

model = ChatOllama(
    model="qwen2.5:7b",
    temperature=0
)

# ---------- TOOLS MOCKEADAS ----------
INDEX_FOLDER = "./rag/chroma_db"
 
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small",
    model_kwargs={"device": "cpu"}
)

vectorstore = Chroma(
    persist_directory=INDEX_FOLDER,
    embedding_function=embeddings
)
coleccion = vectorstore.get()
print("Número de chunks indexados:", len(coleccion["ids"]))
print("Fuentes únicas indexadas:", set(m.get("source", "?") for m in coleccion["metadatas"]))
@tool
def vlm_tool(image_path: str, categoria: str = None) -> dict:
    """Analiza una imagen con el modelo VLM especialista (violencia, armas, fuego, accidente, gráfico)."""
    print(f"[MOCK] vlm_tool llamada con image_path={image_path}, categoria={categoria}")
    time.sleep(0.3)  # simula latencia de inferencia
    return {
        "predicted_category": "violencia",
        "confidence": 0.87,
        "description": "[MOCK] Se detecta una escena con posible agresión física entre dos personas."
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
tools_by_name = {t.name: t for t in TOOLS}

model_with_tools = model.bind_tools(TOOLS)

# ---------- ESTADO ----------

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# ---------- NODOS ----------

def llm_call(state: AgentState):
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="Eres un agente que ayuda a moderar contenido buscando información "
                        "sobre distintos tipos de riesgos (violencia, armas, fuego, accidentes). NO respondas a preguntas que no estén relacionadas con la moderación de contenido."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def tool_node(state: AgentState):
    """Performs the tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        selected_tool = tools_by_name[tool_call["name"]]
        observation = selected_tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: AgentState) -> Literal["tool_node", "__end__"]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
    last_message = state["messages"][-1] # Get the last message from the state
    if last_message.tool_calls:
        return "tool_node"
    return END

# ---------- GRAFO ----------

agent_builder = StateGraph(AgentState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()

# Mostrar el grafo
png = agent.get_graph(xray=True).draw_mermaid_png()
display(Image(png))

with open("graph.png", "wb") as f:
    f.write(png)

# ---------- CASOS DE PRUEBA PARA VALIDAR ROUTING ----------

casos_prueba = [
    # {
    #     "pregunta": "Cuál es la definición de violencia de género?",
    #     "tool_esperada": "rag_tool"
    # },
    # {
    #     "pregunta": "Qué categoría tiene esta imagen? image1.png",
    #     "tool_esperada": "vlm_tool"
    # },
    {
        "pregunta": "Zonas afectadas por el fuego en españa el mes pasado",
        "tool_esperada": "internet_tool"
    },
]

for caso in casos_prueba:
    print(f"\n{'='*60}")
    print(f"PREGUNTA: {caso['pregunta']}")
    print(f"TOOL ESPERADA: {caso['tool_esperada']}")
    print('='*60)

    mensajes = [HumanMessage(content=caso["pregunta"])]
    resultado = agent.invoke({"messages": mensajes})

    for m in resultado["messages"]:
        m.pretty_print()