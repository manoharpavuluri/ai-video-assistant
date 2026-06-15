import os
from dataclasses import dataclass
from uuid import uuid4

import requests
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from core.vector_store import build_vector_store, get_retriever, load_vector_store

DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
DEFAULT_MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")


def get_mistral_llm(model_name: str = DEFAULT_MISTRAL_MODEL):
    return ChatMistralAI(
        model=model_name,
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.2,
    )


def format_docs(docs: list) -> str:
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', f'chunk_{i + 1}')}\nContent: {doc.page_content}"
        for i, doc in enumerate(docs)
    ).strip()


def source_list(docs: list) -> str:
    sources = []
    for doc in docs:
        source = doc.metadata.get("source", "transcript_chunk")
        snippet = " ".join(doc.page_content.split())[:220]
        sources.append(f"- {source}: {snippet}")
    return "\n".join(sources)


RAG_SYSTEM_PROMPT = """You are an expert video transcript assistant. Answer the user's question based only on the transcript context below.

If the answer is not found in the context, say: "I could not find this information in the video transcript."

Be concise, precise, and mention relevant transcript source chunk ids when useful.

Transcript context:
{context}"""


@dataclass
class OllamaRagChain:
    retriever: object
    model_name: str = DEFAULT_OLLAMA_MODEL
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def invoke(self, question: str) -> str:
        docs = self.retriever.invoke(question)
        context = format_docs(docs)
        prompt = (
            RAG_SYSTEM_PROMPT.format(context=context)
            + f"\n\nUser question:\n{question}\n\nAnswer:"
        )
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_ctx": 8192},
            },
            timeout=300,
        )
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
        sources = source_list(docs)
        if sources:
            return f"{answer}\n\nSources:\n{sources}"
        return answer


def build_mistral_rag_chain(retriever, model_name: str):
    llm = get_mistral_llm(model_name)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def build_rag_chain(
    transcript: str,
    llm_provider: str = DEFAULT_LLM_PROVIDER,
    llm_model: str | None = None,
    embedding_model: str | None = None,
    retriever_k: int = 5,
):
    collection_name = f"meeting_transcript_{uuid4().hex}"
    vector_store = build_vector_store(
        transcript,
        collection_name=collection_name,
        embedding_model=embedding_model or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    )
    retriever = get_retriever(vector_store, k=retriever_k)

    if llm_provider == "ollama":
        return OllamaRagChain(retriever=retriever, model_name=llm_model or DEFAULT_OLLAMA_MODEL)
    if llm_provider == "mistral":
        return build_mistral_rag_chain(retriever, llm_model or DEFAULT_MISTRAL_MODEL)

    raise ValueError(f"Unsupported LLM provider: {llm_provider}")


def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store)
    return OllamaRagChain(retriever=retriever, model_name=DEFAULT_OLLAMA_MODEL)


def ask_question(rag_chain, question: str) -> str:
    print(f"Question: {question}")
    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}")
    return answer
