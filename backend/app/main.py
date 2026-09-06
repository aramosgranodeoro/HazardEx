from dotenv import load_dotenv
load_dotenv()
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


from app import state
from app.agent.graph import build_agent_graph
from app.agent.rag.embeddings import get_vectorstore
from app.routers.analysis import router as analysis_router
from app.routers.conversations import router as conversations_router
from app.routers.media import router as media_router
from app.routers.rag import router as rag_router  

# Guarda el estado del agente y el vectorstore en la base de datos SQLite "checkpoints.db" al iniciar la aplicación.
# El agente y el vectorstore se inicializan y se guardan en el estado global de la aplicación para su uso en los endpoints.
# Cada vez que se reinicia la aplicación, se restauran desde la base de datos para mantener la continuidad del estado.
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        agent_builder = build_agent_graph()
        state.agent = agent_builder.compile(checkpointer=checkpointer)
        state.vectorstore = get_vectorstore()
        yield


app = FastAPI(title="HazardEx", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
)

app.include_router(analysis_router)
app.include_router(conversations_router)
app.include_router(media_router)
app.include_router(rag_router)


@app.get("/")
def root():
    return {"message": "Backend funcionando"}


@app.get("/health")
def health():
    return {"status": "ok"}