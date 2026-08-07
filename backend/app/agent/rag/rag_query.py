from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

INDEX_FOLDER = "chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small",
    model_kwargs={"device": "cpu"}
)

vectorstore = Chroma(
    persist_directory=INDEX_FOLDER,
    embedding_function=embeddings
)