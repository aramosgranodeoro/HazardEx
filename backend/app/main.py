import uuid
from fastapi.concurrency import asynccontextmanager
from langchain.messages import AIMessage, HumanMessage
from fastapi import FastAPI, File, UploadFile
from app.triage.triage import classify_image, run_specialized_modules
from app.triage.utils import build_analysis_text
from pydantic import BaseModel
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  
from app.agent.graph import build_agent_graph
from fastapi.middleware.cors import CORSMiddleware
# import os
# import shutil
# from fastapi import UploadFile, File, HTTPException
# from app.agent.rag.rag_query import vectorstore
# from app.agent.rag.rag_index import indexar_documento, eliminar_documento, DOCUMENTS_FOLDER

"""
AÑADIR CÓDIGOS DE ERROR HTTP PARA LOS ENDPOINTS DE RAG Y ANALYZE, POR EJEMPLO:
- 400 Bad Request: Para extensiones de archivo no soportadas.
- 409 Conflict: Cuando se intenta subir un archivo que ya existe.
- 500 Internal Server Error: Para errores inesperados durante el procesamiento.
"""
agent = None

EXTENSIONES_PERMITIDAS = {".pdf", ".docx", ".txt"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        agent_builder = build_agent_graph()
        agent = agent_builder.compile(checkpointer=checkpointer)
        yield

 
class QueryRequest(BaseModel):
    thread_id: str
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

    # Construir el texto de análisis 
    analysis_text = build_analysis_text(result)

    # Inyectar el análisis directamente en el estado del hilo
    await agent.aupdate_state(
        config,
        {"messages": [AIMessage(content=f"Resultado del análisis inicial: {analysis_text}")]}
    )

    # Devolver el thread_id y el resultado del análisis inicial
    return {
        "thread_id": thread_id,
        "analysis": analysis_text
    }

@app.post("/query")
async def query(payload: QueryRequest):
    config = {"configurable": {"thread_id": payload.thread_id}}
    new_state = await agent.ainvoke(
        {"messages": [HumanMessage(content=payload.question)]},
        config=config
    )
    return {"response": new_state['messages'][-1].content}

# @app.delete("/conversation/{thread_id}")
# async def delete_conversation(thread_id: str):
#     """Borra la conversación y el estado asociado a un thread_id del checkpointer de LangGraph."""
#     try:
#         # checkpointer es la instancia que ya usas al compilar el grafo
#         # (agent_builder.compile(checkpointer=checkpointer))
#         checkpointer.delete_thread(thread_id)
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error al eliminar la conversación: {str(e)}"
#         )

#     return {
#         "message": "Conversación eliminada correctamente",
#         "thread_id": thread_id
#     }

# @app.post("/rag")
# async def rag(file: UploadFile = File(...)):
#     """Sube un documento y lo indexa en el RAG."""
#     extension = os.path.splitext(file.filename)[1].lower()

#     if extension not in EXTENSIONES_PERMITIDAS:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Extensión no soportada: {extension}. Permitidas: {EXTENSIONES_PERMITIDAS}"
#         )

#     os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
#     ruta_destino = os.path.join(DOCUMENTS_FOLDER, file.filename)

#     if os.path.exists(ruta_destino):
#         raise HTTPException(
#             status_code=409,
#             detail=f"Ya existe un documento con el nombre '{file.filename}'. Elimínalo antes o renombra el archivo."
#         )

#     # Guardar el archivo subido en disco
#     with open(ruta_destino, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     try:
#         n_chunks = indexar_documento(ruta_destino, vectorstore)
#     except Exception as e:
#         os.remove(ruta_destino)  # rollback si falla el indexado
#         raise HTTPException(status_code=500, detail=f"Error al indexar el documento: {str(e)}")

#     return {
#         "message": "Documento indexado correctamente",
#         "filename": file.filename,
#         "chunks_indexados": n_chunks
#     }


# @app.delete("/rag")
# async def delete_rag(filename: str):
#     """Elimina un documento del RAG por su nombre de archivo."""
#     ruta_archivo = os.path.join(DOCUMENTS_FOLDER, filename)

#     n_borrados = eliminar_documento(filename, vectorstore)

#     if n_borrados == 0:
#         raise HTTPException(
#             status_code=404,
#             detail=f"No se encontraron chunks indexados para '{filename}'"
#         )

#     if os.path.exists(ruta_archivo):
#         os.remove(ruta_archivo)

#     return {
#         "message": "Documento eliminado correctamente",
#         "filename": filename,
#         "chunks_eliminados": n_borrados
#     }