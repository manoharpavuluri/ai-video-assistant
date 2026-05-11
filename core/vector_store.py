import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db")
COLLECTION_NAME = "meeting_transcripts"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
            )

def build_vector_store(transcript: str) -> Chroma:
    print("Building vector store from transcript...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200 
    )
    chunks = text_splitter.split_text(transcript)
    documents = [Document(page_content=chunk, metadata={"source": f"chunk_{i}"}) for i, chunk in enumerate(chunks)] 

    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents,
        embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )   

    print(f"Vector store built and persisted at: {CHROMA_DIR}")
    return vector_store 

def load_vector_store() -> Chroma:
    print("Loading vector store...")
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    print("Vector store loaded.")
    return vector_store

def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
            )



