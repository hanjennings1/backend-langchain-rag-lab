"""Response formatting helpers for the LangChain RAG API."""

CHAIN_EXPRESSION = "ChatPromptTemplate | ChatOllama | StrOutputParser"

COMPONENTS = {
    "vector_store": "Chroma",
    "retrieval_method": "similarity_search_with_score",
    "prompt_template": "ChatPromptTemplate",
    "chat_model": "ChatOllama",
    "output_parser": "StrOutputParser",
}

SCORE_TYPE = "Chroma distance; lower usually means closer in this lesson setup"

FALLBACK_ANSWER = (
    "I do not have enough approved runbook context to answer that reliably."
)


def format_sources(scored_documents):
    """Format retrieved documents as source metadata for the API response."""
    # Return a list of dictionaries: should include source_id, title, category, section, chunk_id, and distance.
    # Do not include full document text in the source list.
    sources = []
    for document, score in scored_documents:
        metadata = document.metadata or {}
        sources.append({
            "source_id": metadata.get("source_id", "unknown"),
            "title": metadata.get("title", "Untitled"),
            "category": metadata.get("category", "Uncategorized"),
            "section": metadata.get("section", "Unspecified"),
            "chunk_id": metadata.get("chunk_id", "unknown"),
            "distance": round(float(score), 4),
        })
    return sources


def format_langchain_debug(scored_documents, context, top_k, fallback):
    """Return LangChain debug metadata for inspectability."""
    # Return chain expression, component names, retrieved count, retrieved chunk IDs, top_k, context length, fallback status, and score type.
    chunk_ids = [
        (document.metadata or {}).get("chunk_id", "unknown")
        for document, _score in scored_documents
    ]
    return {
        "chain_expression": CHAIN_EXPRESSION,
        "components": dict(COMPONENTS),
        "retrieved_count": len(scored_documents),
        "retrieved_chunk_ids": chunk_ids,
        "top_k": top_k,
        "context_characters": len(context),
        "fallback": fallback,
        "score_type": SCORE_TYPE,
    }


def format_success_response(answer, sources, debug):
    """Format a successful RAG response."""
    # Return a dictionary with answer, sources, and langchain.
    return {
        "answer": answer.strip(),
        "sources": sources,
        "langchain": debug,
    }


def format_fallback_response(debug):
    """Format a safe response when no usable context is available."""
    # Return a response with FALLBACK_ANSWER, empty sources, and debug.
    return {
        "answer": FALLBACK_ANSWER,
        "sources": [],
        "langchain": debug,
    }


def format_error_response(error, message):
    """Format an API error response."""
    # Return a dictionary with error and message.
    return {"error": error, "message": message}
