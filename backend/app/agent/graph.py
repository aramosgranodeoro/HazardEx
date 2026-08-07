# Build workflow
from langgraph.graph import StateGraph
from app.agent.nodes import llm_call, tool_node
from langgraph.graph import MessagesState
from langchain.messages import HumanMessage
from app.agent.endLogic import should_continue, END, START
from IPython.display import Image, display


agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# Show the agent
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# Invoke
messages = [HumanMessage(content="¿Donde se encuentra el fuego en la imagen?")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()