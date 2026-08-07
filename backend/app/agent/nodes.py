# Step 3: Define model node
from langchain.messages import SystemMessage
from langgraph.graph import MessagesState
from langchain.messages import ToolMessage

from langchain.tools import tool
from langchain_ollama import ChatOllama

from tools import TOOLS, tools_by_name  # vlm_tool, rag_tool, internet_tool

model = ChatOllama(
    model="qwen2.5:7b",
    temperature=0
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
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }


# Define tool node
def tool_node(state: MessagesState):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}