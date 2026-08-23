# Build workflow
from langgraph.graph import StateGraph
from app.agent.nodes import llm_call, tool_node
from app.agent.state import MessagesState
from app.agent.endLogic import should_continue, END, START

def build_agent_graph() -> StateGraph:
    """Construye el grafo del agente sin compilar (el checkpointer se añade al compilar)."""
    agent_builder = StateGraph(MessagesState)

    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)

    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        ["tool_node", END]
    )
    agent_builder.add_edge("tool_node", "llm_call")

    return agent_builder