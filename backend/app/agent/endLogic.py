from typing import Literal
from langgraph.graph import StateGraph, START, END
from app.agent.state import MessagesState

def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    """Decide si continuar con la ejecución del grafo o detenerse. Si el LLM hace un tool call, se continúa; de lo contrario, se detiene."""

    messages = state["messages"]
    last_message = messages[-1]

    # Si el LLM hace un tool call, entonces se realiza una acción
    if last_message.tool_calls:
        return "tool_node"

    # Si no hay tool calls, se detiene la ejecución del grafo
    return END