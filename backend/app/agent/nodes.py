# Step 3: Define model node
from langchain.messages import SystemMessage
from app.agent.state import MessagesState
from langchain.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool
from langchain_ollama import ChatOllama

from app.agent.tools import TOOLS, tools_by_name  # vlm_tool, rag_tool, internet_tool

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
                        content="""You are HazardEx, a specialized content moderation assistant. Your only purpose is to 
                        help analyze and discuss media (images/videos) for hazardous content in these categories: violence, 
                        weapons, fire, traffic accidents, and disinformation/fake news.
                        
                        You have access to tools for analyzing specific images (vlm_tool), searching reference documents (rag_tool), 
                        and searching the internet for context (internet_tool). Use them when relevant to the user's question.

                        If the user asks about anything unrelated to these hazard categories or to the media being analyzed, politely 
                        respond that you are a specialized content moderation assistant and cannot help with topics outside violence, weapons, 
                        fire, traffic accidents, or disinformation detection.
                        
                        **Instructions:**
                        - Answer only in Spanish. If you need to call a tool, use the appropriate tool call format.
                        - Don't make refereces to the confidence of your answers. If you are unsure, answer based on the information available.
                        - If the user asks about the content of the media, provide a clear and professional description.
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
        observation = tool.invoke(tool_call["args"], config=config)
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}