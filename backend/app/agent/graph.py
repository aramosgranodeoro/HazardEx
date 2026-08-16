# Build workflow
from langgraph.graph import StateGraph
from app.agent.nodes import llm_call, tool_node
from app.agent.state import MessagesState
from app.agent.endLogic import should_continue, END, START

# agent_builder = StateGraph(MessagesState)

# # Add nodes
# agent_builder.add_node("llm_call", llm_call)
# agent_builder.add_node("tool_node", tool_node)

# # Add edges to connect nodes
# agent_builder.add_edge(START, "llm_call")
# agent_builder.add_conditional_edges(
#     "llm_call",
#     should_continue,
#     ["tool_node", END]
# )
# agent_builder.add_edge("tool_node", "llm_call")

# # Compile the agent with a checkpointer
# with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
#     agent = agent_builder.compile(checkpointer=checkpointer)

# # Show the agent
# display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# # Invoke
# messages = [HumanMessage(content="¿Donde se encuentra el fuego en la imagen?")]
# messages = agent.invoke({"messages": messages})
# for m in messages["messages"]:
#     m.pretty_print()


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