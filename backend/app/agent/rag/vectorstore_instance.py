from app.agent.rag.embeddings import get_vectorstore

# Instancia única del vectorstore, compartida por toda la app.
vectorstore = get_vectorstore()