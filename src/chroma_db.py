# cargar los documentosa a la base de datos de chorma
import os
import shutil
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma"

def save_to_chroma_db(chunks: list[Document], embedding_model) -> Chroma:
    # remover las bases de datos existentes
    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
        except OSError as e:
            print(f"Error removing directory: {e}")
    # crear la base de datos
    chroma_db = Chroma.from_documents(chunks, embedding=embedding_model, persist_directory=CHROMA_PATH)

    print(f"Base de datos {len(chunks)} de documentos guardada en {CHROMA_PATH}")
    return chroma_db
