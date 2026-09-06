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
    # lista de mensajes en la conversación. Se puede usar operator.add para añadir nuevos mensajes.
    messages: Annotated[list[AnyMessage], operator.add]
    # número de llamadas al LLM realizadas en esta conversación.
    llm_calls: int
    # identificador del hilo de la conversación. Puede ser None si no se ha iniciado una conversación.
    thread_id: NotRequired[str]
    # available_media es un diccionario que mapea media_id a media_type (ej. "image/png", "video/mp4").
    available_media: Annotated[dict[str, str], merge_dicts]