# Planning Notes: LangChain RAG API

Complete each section before submitting. Replace every TODO with your own notes.

## 1. System Goal

- User or role: Engineers handling incidents
- Business problem: Engineers need fast, trustworthy incident steps; general models give generic/made-up advice
- Approved knowledge source: eliability team runbooks in `data/runbook_chunks.json`, stored in Chroma
- Endpoint route: `POST /api/ask`

## 2. Manual RAG Workflow Map

Name the manual RAG steps this LangChain version is organizing.

- Receive question: `app.py` reads JSON; `lib/validation.py` validates it.
- Retrieve context: `lib/vector_store.py` finds top-k runbook chunks with distance scores.
- Build prompt: `lib/prompt_templates.py` fills `{context}` and `{question}`
- Call model: `lib/langchain_rag_service.py` calls `llama3.2` via Ollama
- Parse or format output: Convert reply to a string; `lib/response_formatter.py` builds the JSON
- Return sources: Chunk metadata and distance (no full text)
- Verify quality: Run pytest, test fallback, check sources support the answer

## 3. LangChain Component Mapping

Map the manual workflow to LangChain-supported pieces.

- Prompt string maps to: `ChatPromptTemplate`
- Chroma search logic maps to: `Chroma` + `similarity_search_with_score`
- Direct model call maps to: `ChatOllama`
- Function-to-function workflow maps to: `prompt | llm | StrOutputParser()`
- Manual output cleanup maps to: `StrOutputParser`
- Manual testing maps to: Pytest with fake vector stores and chains

## 4. Response Contract

List the fields a successful response should include.

- answer
- sources
- langchain

## 5. Fallback Behavior

Explain what the system should do when no approved context is available:

Skip the model call and return a safe "no approved guidance found" message with empty `sources` and `fallback: true`, so the model can't invent unsupported advice.

## 6. Verification Plan

List at least four checks you should run before trusting the refactor.

- all pytests pass
- invalid questions return 400 (with Error and Message)
- no-context questions return the fallback & do not call the model
- service failures return 502 code

## 7. Reflection

Explain what LangChain simplified and what may be harder to inspect:

LangChain makes the workflow easy to compose and swap for fakes in tests, but the chain hides intermediate steps (like the exact prompt sent), so debugging metadata is important.
