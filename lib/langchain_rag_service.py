"""LangChain-supported RAG workflow service."""

from lib.config import CHAT_MODEL, DEFAULT_TOP_K
from lib.response_formatter import (
    format_fallback_response,
    format_langchain_debug,
    format_sources,
    format_success_response,
)
from lib.vector_store import retrieve_context


class LangChainServiceError(Exception):
    """Raised when the LangChain RAG service cannot complete a request."""


def build_chat_model():
    """Build the local chat model wrapper."""
    # Import ChatOllama from langchain_ollama.
    from langchain_ollama import ChatOllama

    # Return ChatOllama(model=CHAT_MODEL, temperature=0).
    return ChatOllama(model=CHAT_MODEL, temperature=0)


def build_chain():
    """Build the LangChain prompt to model to parser sequence."""
    # Import StrOutputParser from langchain_core.output_parsers.
    from langchain_core.output_parsers import StrOutputParser
    # Import build_rag_prompt from lib.prompt_templates.
    from lib.prompt_templates import build_rag_prompt

    # Build this sequence:
    prompt_template = build_rag_prompt()
    llm = build_chat_model()

    # return prompt_template | llm | StrOutputParser()
    return prompt_template | llm | StrOutputParser()


def has_usable_context(scored_documents):
    """Return True when retrieval produced at least one document with text."""
    # Return False for an empty list & Return True only if at least one document has non-blank page_content.
    return any(
        (document.page_content or "").strip()
        for document, _score in scored_documents
    )


def format_context(scored_documents):
    """Format retrieved LangChain documents into prompt-ready context text."""
    # Build a readable context block for each retrieved document.
    blocks = []
    # Include source_id, title, category, section, chunk_id, distance, and text.
    for index, (document, score) in enumerate(scored_documents, start=1):
        metadata = document.metadata or {}
        blocks.append(
            f"[Context {index}]\n"
            f"Source ID: {metadata.get('source_id', 'unknown')}\n"
            f"Title: {metadata.get('title', 'Untitled')}\n"
            f"Category: {metadata.get('category', 'Uncategorized')}\n"
            f"Section: {metadata.get('section', 'Unspecified')}\n"
            f"Chunk ID: {metadata.get('chunk_id', 'unknown')}\n"
            f"Distance: {float(score):.4f}\n"
            f"Text: {document.page_content.strip()}"
        )
    return "\n\n".join(blocks)


def answer_question(
    question,
    *,
    vector_store=None,
    chain=None,
    top_k=DEFAULT_TOP_K,
):
    """Run the LangChain-supported RAG workflow for one validated question."""
    try:
        # Strip the question.
        cleaned_question = question.strip()

        # Retrieve scored documents.
        scored_documents = retrieve_context(
            cleaned_question, vector_store=vector_store, top_k=top_k
        )
        
        # If no usable context exists, return a safe fallback without calling the chain.
        if not has_usable_context(scored_documents):
            debug = format_langchain_debug(scored_documents, "", top_k, True)
            return format_fallback_response(debug)

        # Format the context.
        context = format_context(scored_documents)

        # Build or use the provided chain.
        rag_chain = chain if chain is not None else build_chain()
        # Invoke the chain with {"context": context, "question": cleaned_question}.
        answer = rag_chain.invoke(
            {"context": context, "question": cleaned_question}
        )

        # Reject empty model output.
        if not isinstance(answer, str) or not answer.strip():
            raise LangChainServiceError("The model returned an empty answer.")

        # Format sources and LangChain debug metadata.
        sources = format_sources(scored_documents)
        debug = format_langchain_debug(scored_documents, context, top_k, False)
        # Return the success response.
        return format_success_response(answer, sources, debug)

    # Wrap unexpected exceptions in LangChainServiceError.
    except LangChainServiceError:
        raise
    except Exception as exc:
        raise LangChainServiceError(f"LangChain RAG workflow failed: {exc}") from exc
