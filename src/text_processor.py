from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCUMENTS_PATH = "data/documents"
def chunks_pdfs() -> list[Document]:
    # inicializar el cargador de pdfs
    document_loader = PyPDFDirectoryLoader(DOCUMENTS_PATH)
    # objeto que contiene los documentos cargados
    documents = document_loader.load()
    # dividir los documentos en chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len, add_start_index=True)
    # crear los chunks
    chunks = text_splitter.split_documents(documents)
    return chunks