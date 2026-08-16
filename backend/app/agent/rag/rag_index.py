import os
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader


DOCUMENTS_FOLDER = "documents"
INDEX_FOLDER = "chroma_db"
MODELO_EMBEDDINGS = "intfloat/multilingual-e5-small" 

# Cargar un documento 
def load_document(ruta_archivo: str, vectorstore) -> int:
    """Carga, trocea e indexa un único documento nuevo en el vectorstore existente."""
    extension = os.path.splitext(ruta_archivo)[1].lower()

    if extension == ".pdf":
        loader = PyPDFLoader(ruta_archivo)
    elif extension == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(ruta_archivo)
    elif extension == ".txt":
        loader = TextLoader(ruta_archivo, encoding="utf-8")
    else:
        raise ValueError(f"Extensión no soportada: {extension}")

    documentos = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
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
    """Carga PDFs y archivos de texto de la carpeta documents/"""
    docs = []

    # PDFs
    loader_pdf = DirectoryLoader(
        DOCUMENTS_FOLDER, glob="**/*.pdf", loader_cls=PyPDFLoader
    )
    print(f"Cargando documentos desde {DOCUMENTS_FOLDER}...")  
    
    docs.extend(loader_pdf.load())

    # Texto plano / markdown
    loader_txt = DirectoryLoader(
        DOCUMENTS_FOLDER, glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs.extend(loader_txt.load())

    # docx
    loader_docx = DirectoryLoader(
    DOCUMENTS_FOLDER, glob="**/*.docx", loader_cls=Docx2txtLoader
    )
    docs.extend(loader_docx.load())
    print(f"Documentos cargados: {len(docs)}")
    return docs

class EmbeddingsE5(HuggingFaceEmbeddings):
    """Wrapper que añade los prefijos requeridos por los modelos e5."""
    def embed_documents(self, texts):
        textos_con_prefijo = [f"passage: {t}" for t in texts]
        return super().embed_documents(textos_con_prefijo)

    def embed_query(self, text):
        return super().embed_query(f"query: {text}")

def build_index():
    documentos = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documentos)
    print(f"Chunks generados: {len(chunks)}")

    embeddings = EmbeddingsE5(
        model_name=MODELO_EMBEDDINGS,
        model_kwargs={"device": "cpu"}
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=INDEX_FOLDER
    )

    print(f"Índice construido y guardado en {INDEX_FOLDER}")
    return vectorstore

def get_vectorstore():
    """Carga el vectorstore desde disco, o lo construye si no existe."""
    if not os.path.exists(INDEX_FOLDER):
        print(f"Índice no encontrado en {INDEX_FOLDER}. Construyendo índice...")
        return build_index()
    else:
        print(f"Cargando índice desde {INDEX_FOLDER}...")
        embeddings = EmbeddingsE5(
            model_name=MODELO_EMBEDDINGS,
            model_kwargs={"device": "cpu"}
        )
        return Chroma(
            persist_directory=INDEX_FOLDER,
            embedding_function=embeddings
        )

if __name__ == "__main__":
    build_index()