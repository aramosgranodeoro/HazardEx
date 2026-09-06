import logging

from fastapi import APIRouter, HTTPException, status

from app import state
from app.storage.conversations import list_conversations, delete_conversation_metadata
from app.triage.utils import parse_conversation_messages

logger = logging.getLogger(__name__)

router = APIRouter(tags=["conversations"])


@router.get("/conversations", status_code=status.HTTP_200_OK)
def get_conversations():
    try:
        conversations = list_conversations()
    except Exception as e:
        logger.exception("Error listando conversaciones")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al listar las conversaciones: {str(e)}",
        )

    return {"conversations": conversations}


@router.get("/conversation/{thread_id}", status_code=status.HTTP_200_OK)
async def get_conversation(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    try:
        conv_state = await state.agent.aget_state(config)
    except Exception as e:
        logger.exception("Error obteniendo el estado del agente para thread_id=%s", thread_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al recuperar la conversación: {str(e)}",
        )

    if not conv_state or not conv_state.values.get("messages"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversación no encontrada",
        )

    try:
        items = parse_conversation_messages(
            messages=conv_state.values["messages"],
            available_media=conv_state.values.get("available_media", {}),
        )
    except Exception as e:
        logger.exception("Error parseando mensajes para thread_id=%s", thread_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al procesar el historial de la conversación: {str(e)}",
        )

    return {"thread_id": thread_id, "items": items}


@router.delete("/conversation/{thread_id}", status_code=status.HTTP_200_OK)
async def delete_conversation(thread_id: str):
    try:
        await state.agent.checkpointer.adelete_thread(thread_id)
    except Exception as e:
        logger.exception("Error eliminando el estado del agente para thread_id=%s", thread_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al eliminar el estado del agente: {str(e)}",
        )

    try:
        delete_conversation_metadata(thread_id)
    except Exception as e:
        logger.exception("Error eliminando metadata para thread_id=%s", thread_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al eliminar la metadata de la conversación: {str(e)}",
        )

    return {"message": "Conversación eliminada correctamente", "thread_id": thread_id}