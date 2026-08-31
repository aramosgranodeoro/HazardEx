from langchain.messages import AnyMessage
from langchain_protocol import NotRequired
from typing_extensions import TypedDict, Annotated
import operator

"""
 Estado de la conversación, incluyendo los mensajes y el número de llamadas al LLM.
"""

def merge_dicts(left: dict, right: dict) -> dict:
    """Reducer para mergear available_media sin perder entradas anteriores."""
    return {**left, **right}


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    thread_id: NotRequired[str]
    available_media: Annotated[dict[str, str], merge_dicts]