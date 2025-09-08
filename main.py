# integrar las funciones de chroma_db.py y text_processor.py en un solo documento main.py

from src.chroma_db import save_to_chroma_db
from src.text_processor import chunks_pdfs
from langchain_openai import OpenAIEmbeddings
import os

from dotenv import load_dotenv

load_dotenv()


# procesamos los documentos, que toma los documentos de la carpeta data/documents y los divide en chunks
processed_documents = chunks_pdfs()

# creamos el modelo de embedding
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# guardamos los documentos en la base de datos de chroma
db = save_to_chroma_db(processed_documents, embedding_model)

query = "¿cuales son los pasos recomendados para instaurar una demanda de divorcio de mutuo acuerdo?"
# obtenemos los 3 documentos mas relevantes
docs = db.similarity_search(query, k=3)
print(docs)
# unimos los documentos en un solo contexto
context = "\n\n---\n\n".join([doc.page_content for doc in docs])
print(context)

