from fastapi import APIRouter, HTTPException

from app.agent.graph import agent
from app.storage.minio_client import delete_conversation_files


router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"]
)


@router.delete("/{thread_id}")
async def delete_conversation(thread_id: str):
    try:
        await agent.checkpointer.adelete_thread(thread_id)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al eliminar el estado del agente: {str(e)}"
        )

    try:
        delete_conversation_files(thread_id)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al eliminar los archivos: {str(e)}"
        )

    return {
        "message": "Conversación eliminada correctamente",
        "thread_id": thread_id
    }