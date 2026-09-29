"""Request validation helpers for POST /api/ask."""

from lib.config import MAX_QUESTION_LENGTH, MIN_QUESTION_LENGTH


# Helper for Validate_Questions
def _invalid(code, message):
    return None, {"error": code, "message": message}


def validate_question_payload(payload):
    """Validate the JSON body for POST /api/ask.

    Return:
        (question, None) when valid
        (None, error_dict) when invalid
    """

    # Reject non-dictionary payloads.
    if not isinstance(payload, dict):
        return _invalid("invalid_request", "Request body must be a JSON object.")
    
    # Require a "question" field.
    if "question" not in payload:
        return _invalid("missing_question", "The 'question' field is required.")

    # Require the question to be a string.
    question = payload["question"]
    if not isinstance(question, str):
        return _invalid("invalid_question", "The 'question' field must be a string.")
    
    # Strip whitespace.
    question = question.strip()
   
    # Reject blank questions.
    if not question:
        return _invalid("empty_question", "The 'question' field cannot be blank.")
        
    # Reject questions shorter than MIN_QUESTION_LENGTH.
    if len(question) < MIN_QUESTION_LENGTH:
        return _invalid("short_question", f"Question must be at least {MIN_QUESTION_LENGTH} characters.")

    # Reject questions longer than MAX_QUESTION_LENGTH.
    if len(question) > MAX_QUESTION_LENGTH:
        return _invalid("long_question", f"Question must be at most {MAX_QUESTION_LENGTH} characters.")

    # Return (question, None) when valid.
    return question, None
