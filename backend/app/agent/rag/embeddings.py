import os
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, DirectoryLoader, Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_FOLDER = os.path.join(_BASE_DIR, "documents")
INDEX_FOLDER = os.path.join(_BASE_DIR, "chroma_db")
MODELO_EMBEDDINGS = "intfloat/multilingual-e5-small"


class EmbeddingsE5(HuggingFaceEmbeddings):
    """
    Wrapper que añade los prefijos requeridos por los modelos e5.
    Los modelos e5 requieren que los textos de los documentos tengan el prefijo "passage: ".
    Si no se añade este prefijo, los embeddings no serán compatibles con la consulta.
    """

    def embed_documents(self, texts):
        textos_con_prefijo = [f"passage: {t}" for t in texts]
        return super().embed_documents(textos_con_prefijo)

    def embed_query(self, text):
        return super().embed_query(f"query: {text}")


def _get_embeddings() -> EmbeddingsE5:
    return EmbeddingsE5(model_name=MODELO_EMBEDDINGS, model_kwargs={"device": "cpu"})


def load_document(ruta_archivo: str, vectorstore) -> int:
    """Carga, trocea e indexa un único documento nuevo en el vectorstore existente."""
    extension = os.path.splitext(ruta_archivo)[1].lower()

    if extension == ".pdf":
        loader = PyPDFLoader(ruta_archivo)
    elif extension == ".docx":
        loader = Docx2txtLoader(ruta_archivo)
    elif extension == ".txt":
        loader = TextLoader(ruta_archivo, encoding="utf-8")
    elif extension == ".md":
        loader = TextLoader(ruta_archivo, encoding="utf-8")
    else:
        raise ValueError(f"Extensión no soportada: {extension}")

    documentos = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documentos)

    # Metadato consistente para poder borrar después por nombre de archivo
    nombre_archivo = os.path.basename(ruta_archivo)
    for chunk in chunks:
        chunk.metadata["source"] = nombre_archivo

    vectorstore.add_documents(chunks)
    return len(chunks)


def delete_document(nombre_archivo: str, vectorstore) -> int:
    """Elimina todos los chunks de un documento del índice, dado su nombre de archivo."""
    resultado = vectorstore.get(where={"source": nombre_archivo})
    ids_a_borrar = resultado["ids"]

    if not ids_a_borrar:
        return 0

    vectorstore.delete(ids=ids_a_borrar)
    return len(ids_a_borrar)


def load_documents():
    """Carga PDFs, TXT y DOCX de la carpeta documents/ (para construir el índice inicial)."""
    docs = []
    print(f"Cargando documentos desde {DOCUMENTS_FOLDER}...")

    loader_pdf = DirectoryLoader(DOCUMENTS_FOLDER, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs.extend(loader_pdf.load())

    loader_txt = DirectoryLoader(
        DOCUMENTS_FOLDER, glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs.extend(loader_txt.load())

    loader_docx = DirectoryLoader(DOCUMENTS_FOLDER, glob="**/*.docx", loader_cls=Docx2txtLoader)
    docs.extend(loader_docx.load())

    print(f"Documentos cargados: {len(docs)}")
    return docs


def build_index():
    """
    Construye el índice vectorial a partir de los documentos en la carpeta documents/.
    Si el índice ya existe, lo sobrescribe.
    """
    documentos = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documentos)
    print(f"Chunks generados: {len(chunks)}")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=_get_embeddings(),
        persist_directory=INDEX_FOLDER
    )

    return vectorstore


def get_vectorstore():
    """Carga el vectorstore desde disco, o lo construye si no existe todavía."""
    if not os.path.exists(INDEX_FOLDER):
        print(f"Índice no encontrado en {INDEX_FOLDER}. Construyendo índice...")
        return build_index()

    print(f"Cargando índice desde {INDEX_FOLDER}...")
    return Chroma(persist_directory=INDEX_FOLDER, embedding_function=_get_embeddings())


if __name__ == "__main__":
    build_index()

