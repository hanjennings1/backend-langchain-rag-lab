"""Vector store and retrieval helpers."""

from lib.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
)


def build_embeddings():
    """Build the local Ollama embeddings object used by Chroma."""
    # Import OllamaEmbeddings from langchain_ollama.
    from langchain_ollama import OllamaEmbeddings

    # Return OllamaEmbeddings(model=EMBEDDING_MODEL).
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def build_vector_store():
    """Build the Chroma vector store used by the RAG pipeline."""
    # Import Chroma from langchain_chroma.
    from langchain_chroma import Chroma

    # Return a Chroma vector store configured with:
    # - collection_name=COLLECTION_NAME
    # - persist_directory=CHROMA_PATH
    # - embedding_function=build_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=build_embeddings(),
    )


def retrieve_context(question, *, vector_store=None, top_k=DEFAULT_TOP_K):
    """Retrieve scored documents for a question."""
    # Reject blank questions.
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-blank string.")
    
    # Use the provided vector_store or build_vector_store().
    store = vector_store if vector_store is not None else build_vector_store()

    # Call similarity_search_with_score(question, k=top_k).
    return store.similarity_search_with_score(question, k=top_k)