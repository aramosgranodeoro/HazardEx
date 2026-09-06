import logging
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from langchain.messages import HumanMessage

from app import state
from app.triage.triage import classify_image, run_specialized_modules
from app.triage.utils import build_analysis_text, truncate_title
from app.storage.conversations import save_conversation_metadata
from app.services.media_service import process_and_upload_media
from app.services.analysis_service import register_media_in_agent_state, run_initial_analysis

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analysis"])


class QueryRequest(BaseModel):
    thread_id: Optional[str] = None
    question: str


@router.post("/analyze", status_code=status.HTTP_200_OK)
async def analyze(file: UploadFile = File(...), thread_id: Optional[str] = Form(None)):
    is_new_thread = thread_id is None
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    media_bytes = await file.read()
    if not media_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo recibido está vacío.",
        )

    content_type = file.content_type or "application/octet-stream"
    if not (content_type.startswith("image") or content_type.startswith("video")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tipo de contenido no soportado: {content_type}",
        )

    media_id = str(uuid.uuid4())

    try:
        categories, image = classify_image(media_bytes, file.filename)
        result = await run_specialized_modules(categories, image)
        analysis_text = build_analysis_text(result)

        uploaded = process_and_upload_media(
            thread_id, media_id, media_bytes, content_type, image, result
        )

        if is_new_thread:
            save_conversation_metadata(thread_id, file.filename)

        await register_media_in_agent_state(config, uploaded.media_id, uploaded.media_type)
        analysis = await run_initial_analysis(config, uploaded.media_id, analysis_text)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error procesando /analyze para thread_id=%s", thread_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al procesar el archivo multimedia: {str(e)}",
        )

    return {
        "thread_id": thread_id,
        "media_id": uploaded.media_id,
        "annotated_media_id": uploaded.annotated_media_id,
        "analysis": analysis,
    }


@router.post("/query", status_code=status.HTTP_200_OK)
async def query(payload: QueryRequest):
    if not payload.question or not payload.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La pregunta no puede estar vacía.",
        )

    is_new_conversation = payload.thread_id is None
    thread_id = payload.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        new_state = await state.agent.ainvoke(
            {"messages": [HumanMessage(content=payload.question)]}, config=config
        )
    except Exception as e:
        logger.exception("Error procesando /query para thread_id=%s", thread_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al consultar al agente: {str(e)}",
        )

    if is_new_conversation:
        save_conversation_metadata(thread_id, truncate_title(payload.question))

    return {"thread_id": thread_id, "response": new_state["messages"][-1].content}