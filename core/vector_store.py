import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db")
COLLECTION_NAME = "meeting_transcripts"
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mistral-embed")


def get_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL):
    if model_name == "mistral-embed":
        return MistralAIEmbeddings(model=model_name, api_key=os.getenv("MISTRAL_API_KEY"))

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def split_transcript_for_rag(transcript: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=180,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
    )
    return splitter.split_text(transcript)


def build_vector_store(
    transcript: str,
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Chroma:
    print("Building vector store from transcript...")
    chunks = split_transcript_for_rag(transcript)
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "source": f"transcript_chunk_{i + 1}",
                "chunk": i + 1,
            },
        )
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings(embedding_model)

    vector_store = Chroma.from_documents(
        documents,
        embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_DIR,
    )

    print(f"Vector store built and persisted at: {CHROMA_DIR}")
    return vector_store


def load_vector_store(
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Chroma:
    print("Loading vector store...")
    embeddings = get_embeddings(embedding_model)
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print("Vector store loaded.")
    return vector_store


def get_retriever(vector_store: Chroma, k: int = 5):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
