from http.client import HTTPException
import os
import shutil
from typing import Optional
import uuid
from fastapi.concurrency import asynccontextmanager
from langchain.messages import AIMessage, HumanMessage
from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from app.triage.triage import classify_image, run_specialized_modules
from app.triage.utils import build_analysis_text, generic_media_title, truncate_title, frame_to_jpeg_bytes, extract_frames, transcode_to_h264
from app.storage.minio_client import upload_media, get_media, delete_media
from pydantic import BaseModel
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  
from app.agent.graph import build_agent_graph
from fastapi.middleware.cors import CORSMiddleware
from app.storage.conversations import list_conversations, delete_conversation_metadata, save_conversation_metadata
from app.agent.rag.embeddings import load_document, delete_document, get_vectorstore
from dotenv import load_dotenv
import os
import re


MEDIA_MARKER_RE = re.compile(r"\[Image attached, media_id=([a-f0-9\-]+)\]")

load_dotenv()

"""
AÑADIR CÓDIGOS DE ERROR HTTP PARA LOS ENDPOINTS DE RAG Y ANALYZE, POR EJEMPLO:
- 400 Bad Request: Para extensiones de archivo no soportadas.
- 409 Conflict: Cuando se intenta subir un archivo que ya existe.
- 500 Internal Server Error: Para errores inesperados durante el procesamiento.
"""
agent = None

vectorstore = None

EXTENSIONES_PERMITIDAS = {".pdf", ".docx", ".txt", ".md"}

DOCUMENTS_FOLDER = os.getenv("DOCUMENTS_FOLDER")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, vectorstore
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        agent_builder = build_agent_graph()
        agent = agent_builder.compile(checkpointer=checkpointer)
        vectorstore = get_vectorstore()
        yield

 
class QueryRequest(BaseModel):
    """
    Modelo de datos para la solicitud de consulta al agente.
    """
    thread_id: Optional[str] = None
    question: str

app = FastAPI(
    title="HazardEx",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
)

@app.get("/")
def root():
    return {"message": "Backend funcionando"}

@app.get("/health")
def health():
    return {"status": "ok"}

# Análisis inicial del medio (imagen o vídeo) y generación de contexto para el agente
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    thread_id: Optional[str] = Form(None)
):
    is_new_thread = thread_id is None
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    media_bytes = await file.read()
    categories, image = classify_image(media_bytes, file.filename)
    result = await run_specialized_modules(categories, image)
    analysis_text = build_analysis_text(result)
    content_type = file.content_type or "application/octet-stream"

    media_id = str(uuid.uuid4())
    media_type = "video" if content_type.startswith("video") else "photo"
    if media_type == "video":
        # Transcodifica para asegurar reproducción en navegador
        h264_bytes = transcode_to_h264(media_bytes)
        upload_media(thread_id, media_id, h264_bytes, "video/mp4")

        # Genera thumbnail a partir del vídeo ORIGINAL (OpenCV suele leer HEVC bien)
        frames = extract_frames(media_bytes, n=2)
        thumb_bytes = frame_to_jpeg_bytes(frames[0])
        upload_media(thread_id, f"{media_id}_thumb", thumb_bytes, "image/jpeg")
    else:
        upload_media(thread_id, media_id, media_bytes, content_type)

    if is_new_thread:
        save_conversation_metadata(thread_id, file.filename)

    await agent.aupdate_state(
        config,
        {"available_media": {media_id: media_type}}
    )

    prompt = (
        f"""
        [Image attached, media_id={media_id}]
        These are the results of the initial automatic analysis of the multimedia content:
        {analysis_text}
        Write a clear and professional initial description of the content based solely on these results, 
        without inventing or adding any additional information.
        Answer only in Spanish.
        """
    )

    response = await agent.ainvoke(
        {"messages": [HumanMessage(content=prompt)]},
        config
    )
    ai_message = response["messages"][-1]

    return {
        "thread_id": thread_id,
        "media_id": media_id,
        "analysis": ai_message.content
    }

@app.post("/query")
async def query(payload: QueryRequest):
    is_new_conversation = payload.thread_id is None
    thread_id = payload.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    new_state = await agent.ainvoke(
        {"messages": [HumanMessage(content=payload.question)]},
        config=config
    )

    if is_new_conversation:
        save_conversation_metadata(thread_id, truncate_title(payload.question))

    return {
        "thread_id": thread_id,
        "response": new_state['messages'][-1].content
    }

# ---------- Historial ----------

@app.get("/conversations")
def get_conversations():
    return {"conversations": list_conversations()}

@app.get("/conversation/{thread_id}")
async def get_conversation(thread_id: str):
    """Devuelve el historial de mensajes para poder reanudar la conversación."""
    config = {"configurable": {"thread_id": thread_id}}
    state = await agent.aget_state(config)
    if not state or not state.values.get("messages"):
        raise HTTPException(status_code=404, detail="Conversación no encontrada")

    available_media = state.values.get("available_media", {})

    items = []
    for msg in state.values["messages"]:
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else ""
            match = MEDIA_MARKER_RE.search(content)
            if match:
                media_id = match.group(1)
                items.append({
                    "type": "media",
                    "media_id": media_id,
                    "media_type": available_media.get(media_id, "photo"),
                })
                continue
            items.append({"type": "message", "role": "user", "text": content})
        elif isinstance(msg, AIMessage) and msg.content:
            items.append({"type": "message", "role": "assistant", "text": msg.content})

    return {
        "thread_id": thread_id,
        "items": items,
    }

@app.get("/media/{thread_id}/{media_id}")
async def get_media_endpoint(thread_id: str, media_id: str, request: Request):
    data = get_media(thread_id, media_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Media no encontrada")

    config = {"configurable": {"thread_id": thread_id}}
    state = await agent.aget_state(config)
    available_media = state.values.get("available_media", {}) if state else {}
    media_type = available_media.get(media_id)
    content_type = "video/mp4" if media_type == "video" else "image/jpeg"

    file_size = len(data)
    range_header = request.headers.get("range")

    if range_header is None:
        return Response(
            content=data,
            media_type=content_type,
            headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
        )

    # Parsear "bytes=start-end"
    range_value = range_header.replace("bytes=", "").split("-")
    start = int(range_value[0]) if range_value[0] else 0
    end = int(range_value[1]) if len(range_value) > 1 and range_value[1] else file_size - 1
    end = min(end, file_size - 1)
    chunk = data[start:end + 1]

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(len(chunk)),
    }

    return Response(content=chunk, status_code=206, media_type=content_type, headers=headers)


@app.delete("/conversation/{thread_id}")
async def delete_conversation(thread_id: str):
    try:
        await agent.checkpointer.adelete_thread(thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al eliminar el estado del agente: {str(e)}")

    delete_conversation_metadata(thread_id)
    delete_media(thread_id)

    return {"message": "Conversación eliminada correctamente", "thread_id": thread_id}


# ---------- RAG ----------

@app.get("/rag")
def list_rag_documents():
    if not os.path.isdir(DOCUMENTS_FOLDER):
        return {"documents": []}
    files = [
        f for f in os.listdir(DOCUMENTS_FOLDER)
        if os.path.splitext(f)[1].lower() in EXTENSIONES_PERMITIDAS
    ]
    return {"documents": files}


@app.post("/rag")
async def rag(file: UploadFile = File(...)):
    extension = os.path.splitext(file.filename)[1].lower()

    if extension not in EXTENSIONES_PERMITIDAS:
        raise HTTPException(
            status_code=400,
            detail=f"Extensión no soportada: {extension}. Permitidas: {EXTENSIONES_PERMITIDAS}"
        )

    os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
    ruta_destino = os.path.join(DOCUMENTS_FOLDER, file.filename)

    if os.path.exists(ruta_destino):
        raise HTTPException(
            status_code=409,
            detail=f"Ya existe un documento con el nombre '{file.filename}'. Elimínalo antes o renombra el archivo."
        )

    with open(ruta_destino, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        n_chunks = load_document(ruta_destino, vectorstore)
    except Exception as e:
        os.remove(ruta_destino)
        raise HTTPException(status_code=500, detail=f"Error al indexar el documento: {str(e)}")

    return {"message": "Documento indexado correctamente", "filename": file.filename, "chunks_indexados": n_chunks}


@app.delete("/rag")
async def delete_rag(filename: str):
    ruta_archivo = os.path.join(DOCUMENTS_FOLDER, filename)
    n_borrados = delete_document(filename, vectorstore)

    if n_borrados == 0:
        raise HTTPException(status_code=404, detail=f"No se encontraron chunks indexados para '{filename}'")

    if os.path.exists(ruta_archivo):
        os.remove(ruta_archivo)

    return {"message": "Documento eliminado correctamente", "filename": filename, "chunks_eliminados": n_borrados}