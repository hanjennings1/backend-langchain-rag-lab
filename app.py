"""Flask API for the LangChain RAG lab."""

from flask import Flask, jsonify, request

from lib.langchain_rag_service import LangChainServiceError, answer_question
from lib.response_formatter import format_error_response
from lib.validation import validate_question_payload


def create_app():
    """Create and configure the Flask application."""

    app = Flask(__name__)

    @app.post("/api/ask")
    def ask():
        """Accept a question and return a source-backed LangChain RAG response."""

        # Get the JSON body with request.get_json(silent=True).
        payload = request.get_json(silent=True)

        # Validate the question with validate_question_payload().
        question, error = validate_question_payload(payload)

        # Return validation errors as JSON with HTTP 400.
        if error:
            return jsonify(error), 400
        
        # Call answer_question(question) for valid requests.
        try:
            response = answer_question(question)

        # Return successful responses as JSON with HTTP 200.
        # Convert LangChainServiceError into a structured HTTP 502 response.
        except LangChainServiceError as exc:
            return jsonify(
                format_error_response("langchain_service_error", str(exc))
            ), 502

        return jsonify(response), 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
