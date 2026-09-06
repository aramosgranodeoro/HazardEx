import logging
import os
import shutil
from dotenv import load_dotenv
load_dotenv()
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from app import state
from app.agent.rag.embeddings import load_document, delete_document

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rag"])

EXTENSIONES_PERMITIDAS = {".pdf", ".docx", ".txt", ".md"}
DOCUMENTS_FOLDER = os.getenv("DOCUMENTS_FOLDER")


@router.get("/rag", status_code=status.HTTP_200_OK)
def list_rag_documents():
    """Devuelve la lista de documentos indexados para RAG."""
    if not DOCUMENTS_FOLDER:
        logger.error("DOCUMENTS_FOLDER no está configurado en el entorno")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="La carpeta de documentos no está configurada en el servidor.",
        )

    if not os.path.isdir(DOCUMENTS_FOLDER):
        return {"documents": []}

    try:
        files = [
            f for f in os.listdir(DOCUMENTS_FOLDER)
            if os.path.splitext(f)[1].lower() in EXTENSIONES_PERMITIDAS
        ]
    except OSError as e:
        logger.exception("Error listando documentos en %s", DOCUMENTS_FOLDER)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al listar los documentos: {str(e)}",
        )

    return {"documents": files}


@router.post("/rag", status_code=status.HTTP_201_CREATED)
async def rag(file: UploadFile = File(...)):
    """Indexa un documento para su uso en RAG."""
    if not DOCUMENTS_FOLDER:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="La carpeta de documentos no está configurada en el servidor.",
        )

    extension = os.path.splitext(file.filename)[1].lower()
    if extension not in EXTENSIONES_PERMITIDAS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Extensión no soportada: {extension}. Permitidas: {EXTENSIONES_PERMITIDAS}",
        )

    os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
    ruta_destino = os.path.join(DOCUMENTS_FOLDER, file.filename)

    if os.path.exists(ruta_destino):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Ya existe un documento con el nombre '{file.filename}'. Elimínalo antes o renombra el archivo.",
        )

    try:
        with open(ruta_destino, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except OSError as e:
        logger.exception("Error guardando el archivo %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al guardar el archivo: {str(e)}",
        )

    try:
        n_chunks = load_document(ruta_destino, state.vectorstore)
    except Exception as e:
        logger.exception("Error indexando el documento %s", file.filename)
        os.remove(ruta_destino)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al indexar el documento: {str(e)}",
        )

    return {
        "message": "Documento indexado correctamente",
        "filename": file.filename,
        "chunks_indexados": n_chunks,
    }


@router.delete("/rag", status_code=status.HTTP_200_OK)
async def delete_rag(filename: str):
    """Elimina un documento indexado para RAG y sus chunks asociados."""
    if not DOCUMENTS_FOLDER:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="La carpeta de documentos no está configurada en el servidor.",
        )

    try:
        n_borrados = delete_document(filename, state.vectorstore)
    except Exception as e:
        logger.exception("Error eliminando chunks para %s", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al eliminar los chunks del documento: {str(e)}",
        )

    if n_borrados == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No se encontraron chunks indexados para '{filename}'",
        )

    ruta_archivo = os.path.join(DOCUMENTS_FOLDER, filename)
    if os.path.exists(ruta_archivo):
        try:
            os.remove(ruta_archivo)
        except OSError as e:
            logger.exception("Error eliminando el archivo físico %s", filename)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Los chunks se eliminaron pero no se pudo borrar el archivo físico: {str(e)}",
            )

    return {
        "message": "Documento eliminado correctamente",
        "filename": filename,
        "chunks_eliminados": n_borrados,
    }