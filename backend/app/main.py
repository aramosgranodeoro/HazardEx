from http.client import HTTPException
import io
import os
import shutil
from typing import Optional
import uuid
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import StreamingResponse
from langchain.messages import AIMessage, HumanMessage
from fastapi import FastAPI, File, UploadFile
from app.triage.triage import classify_image, run_specialized_modules
from app.triage.utils import build_analysis_text, generic_media_title, truncate_title
from app.storage.minio_client import upload_media, download_media
from pydantic import BaseModel
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  
from app.agent.graph import build_agent_graph
from fastapi.middleware.cors import CORSMiddleware
from app.storage.state import media_metadata
from app.storage.conversations import list_conversations, delete_conversation_metadata, save_conversation_metadata
from app.agent.rag.vectorstore_instance import vectorstore
from app.agent.rag.embeddings import load_document, delete_document
from app.storage.minio_client import upload_media, download_media, delete_media

"""
AÑADIR CÓDIGOS DE ERROR HTTP PARA LOS ENDPOINTS DE RAG Y ANALYZE, POR EJEMPLO:
- 400 Bad Request: Para extensiones de archivo no soportadas.
- 409 Conflict: Cuando se intenta subir un archivo que ya existe.
- 500 Internal Server Error: Para errores inesperados durante el procesamiento.
"""
agent = None

EXTENSIONES_PERMITIDAS = {".pdf", ".docx", ".txt"}

DOCUMENTS_FOLDER = "C:/Users/adaxi/OneDrive/Escritorio/TFG - copia/backend/app/agent/rag/documents"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        agent_builder = build_agent_graph()
        agent = agent_builder.compile(checkpointer=checkpointer)
        yield

 
class QueryRequest(BaseModel):
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
)

@app.get("/")
def root():
    return {"message": "Backend funcionando"}

@app.get("/health")
def health():
    return {"status": "ok"}

# Análisis inicial del medio (imagen o vídeo) y generación de contexto para el agente
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Generar un thread_id único para la conversación
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Leer el archivo subido y analizarlo => primer análisis
    media_bytes = await file.read()
    categories, image = classify_image(media_bytes, file.filename)
    result = await run_specialized_modules(categories, image)
    analysis_text = build_analysis_text(result)

    content_type = file.content_type or "application/octet-stream"
    upload_media(thread_id, media_bytes, content_type)
    media_metadata[thread_id] = {
        "media_type": "video" if content_type.startswith("video") else "photo",
    }

    save_conversation_metadata(thread_id, file.filename, media_metadata[thread_id]["media_type"])

    prompt = (
        f"""
        These are the results of the initial automatic analysis of the multimedia content:
        {analysis_text}

        Write a clear and professional initial description of the content based solely on these results, 
        without inventing or adding any additional information.

        Anwer only in Spanish.
        """
    )

    response = await agent.ainvoke(
        {"messages": [HumanMessage(content=prompt)]},
        config
    )

    ai_message = response["messages"][-1]

    return {
        "thread_id": thread_id,
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
        save_conversation_metadata(thread_id, truncate_title(payload.question), "text")

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

    info = media_metadata.get(thread_id)
    has_media = info is not None
    skip_first_human = has_media  # el primer HumanMessage es el prompt interno de triage

    messages = []
    for msg in state.values["messages"]:
        if isinstance(msg, HumanMessage):
            if skip_first_human:
                skip_first_human = False
                continue
            messages.append({"role": "user", "text": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            messages.append({"role": "assistant", "text": msg.content})

    return {
        "thread_id": thread_id,
        "messages": messages,
        "has_media": has_media,
        "media_type": info["media_type"] if info else None,
    }


@app.get("/media/{thread_id}")
async def get_media(thread_id: str):
    data = download_media(thread_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Media no encontrada")
    info = media_metadata.get(thread_id, {})
    content_type = "video/mp4" if info.get("media_type") == "video" else "image/jpeg"
    return StreamingResponse(io.BytesIO(data), media_type=content_type)


@app.delete("/conversation/{thread_id}")
async def delete_conversation(thread_id: str):
    try:
        await agent.checkpointer.adelete_thread(thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al eliminar el estado del agente: {str(e)}")

    delete_conversation_metadata(thread_id)
    delete_media(thread_id)
    media_metadata.pop(thread_id, None)

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