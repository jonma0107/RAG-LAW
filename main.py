# integrar las funciones de chroma_db.py y text_processor.py en un solo documento main.py

from src.chroma_db import save_to_chroma_db
from src.text_processor import chunks_pdfs
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv

load_dotenv()


# procesamos los documentos, que toma los documentos de la carpeta data/documents y los divide en chunks
processed_documents = chunks_pdfs()

# creamos el modelo de embedding
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# guardamos los documentos en la base de datos de chroma
db = save_to_chroma_db(processed_documents, embedding_model)

query = "¿cuales son los planetas de nuestro sistema solar?"
# obtenemos los 3 documentos mas relevantes
docs = db.similarity_search(query, k=3)
print(docs)
# unimos los documentos en un solo contexto
context = "\n\n---\n\n".join([doc.page_content for doc in docs])
print(context)

# platilla que nos permitirá hacer la pregunta al modelo
PROMPT_TEMPLATE = """
You have to answer the following question based on the following context:
{context}
Answer the following question: {question}
Provide you detailed answer -
Don't include non-relevant information.
"""

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context, question=query)

# Import OpenAI model
model = ChatOpenAI()
response = model.predict(prompt)

print(response)

