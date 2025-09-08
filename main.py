# integrar las funciones de chroma_db.py y text_processor.py en un solo documento main.py

from src.chroma_db import save_to_chroma_db
from src.text_processor import chunks_pdfs
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
import sys

from dotenv import load_dotenv

load_dotenv()


# plantilla que nos permitirá hacer la pregunta al modelo
PROMPT_SYSTEM = (
    "Eres un asistente que responde estrictamente con base en el contexto "
    "proporcionado desde PDFs legales colombianos. Si la respuesta no está en el "
    "contexto, responde exactamente: 'No tengo suficiente información en los documentos "
    "para responder a esa pregunta.' Responde en español, de forma concisa y precisa. "
    "No uses conocimiento externo."
)
PROMPT_HUMAN = (
    "Contexto:\n{context}\n\n"
    "Pregunta: {question}"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", PROMPT_SYSTEM),
    ("human", PROMPT_HUMAN),
])


def run_question(question: str) -> None:
    # procesamos los documentos, que toma los documentos de la carpeta data/documents y los divide en chunks
    processed_documents = chunks_pdfs()

    # creamos el modelo de embedding
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # guardamos los documentos en la base de datos de chroma
    db = save_to_chroma_db(processed_documents, embedding_model)

    # obtenemos los 3 documentos mas relevantes
    docs = db.similarity_search(question, k=3)
    print(docs)
    # unimos los documentos en un solo contexto
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    print(context)

    messages = prompt_template.format_messages(context=context, question=question)

    # Import OpenAI model (temperature 0 para respuestas determinísticas)
    model = ChatOpenAI(temperature=0)
    response = model.invoke(messages)

    print(response.content)


if __name__ == "__main__":
    # Permitir pasar la pregunta por CLI: python main.py "tu pregunta"
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""

    # Si no viene por CLI, pedirla de forma interactiva
    if not question:
        try:
            question = input("Escribe tu pregunta: ").strip()
        except EOFError:
            question = ""

    if not question:
        print("No se proporcionó ninguna pregunta.")
        raise SystemExit(1)

    run_question(question)

