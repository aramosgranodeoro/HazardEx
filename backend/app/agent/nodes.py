# Step 3: Define model node
from langchain.messages import SystemMessage
from app.agent.state import MessagesState
from langchain.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool
from langchain_ollama import ChatOllama

from app.agent.tools import TOOLS, tools_by_name  # vlm_tool, rag_tool, internet_tool

model = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
)

model_with_tools = model.bind_tools(TOOLS)

# Define model node
def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="""Eres HazardEx, un asistente especializado en moderación de contenido. Tu único propósito es
                        ayudar a analizar y comentar contenido multimedia (imágenes/vídeos) en busca de contenido peligroso en estas
                        categorías: violencia, armas, incendios, accidentes de tráfico y desinformación/noticias falsas.

                        Tienes acceso a herramientas para analizar imágenes concretas (vlm_tool), buscar en documentos de referencia
                        (rag_tool) y buscar en internet para obtener contexto (internet_tool). Úsalas cuando sean relevantes para la
                        pregunta del usuario.

                        Si el usuario pregunta sobre algo no relacionado con estas categorías de peligro o con el contenido multimedia
                        analizado, responde amablemente que eres un asistente especializado en moderación de contenido y que no puedes
                        ayudar con temas ajenos a la detección de violencia, armas, incendios, accidentes de tráfico o desinformación.

                        **Instrucciones:**
                        - Responde únicamente en español.
                        - Si necesitas llamar a una herramienta, usa el formato de llamada a herramienta correspondiente.
                        - Cuando sea necesario buscar información, busca primero en rag y, si no está ahí, busca en internet.
                        - Si el usuario pregunta sobre el contenido del material multimedia, ofrece una descripción clara y profesional
                        y usa vlm_tool.
                        - No hagas referencias a la confianza numérica de tus respuestas.
                        - Si tienes dudas, responde basándote en la información disponible en el contexto o en rag.
                        - Esta conversación puede contener más de una imagen o vídeo, cada uno marcado en el historial como
                        "[Image attached, media_id=...]". Cuando el usuario pregunte sobre "la imagen", "el vídeo", o use una
                        referencia ambigua, infiere a qué media_id se refiere a partir del contexto de la conversación (por ejemplo,
                        el adjuntado más recientemente, o uno mencionado explícitamente antes). Si solo hay un media_id en la
                        conversación, usa ese. Pasa siempre el media_id correcto al llamar a vlm_tool.
                        - No respondas con el porcentaje de confianza de la categoría. En su lugar, ofrece una descripción clara y
                        profesional del contenido y sus posibles peligros.
                        """
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

# Define tool node
def tool_node(state: MessagesState, config: RunnableConfig):
    """Performs the tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        args = dict(tool_call["args"])
        if "state" in tool.args_schema.model_fields:
            args["state"] = state
        observation = tool.invoke(args, config=config)
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}