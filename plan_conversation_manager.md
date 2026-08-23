# Conversation Manager Implementation Plan

## 1. Architecture Overview
We will introduce a `ConversationManager` class that sits above the existing building blocks (`LLMGenerator`, `Retriever`, `ChromaDBManager`). This manager will act as a pure business-logic layer, decoupled from any specific user interface (e.g., Console, Streamlit, FastAPI). It will maintain chat history, and intelligently route queries. A separate interface layer (like `console_app.py` or `main.py`) will handle input/output.

### The Flow for Each Turn:
1. **User Input:** The UI layer (e.g., Console loop, Streamlit chat input) captures the user prompt and passes it to `ConversationManager.process_message(user_input)`.
2. **Contextual Evaluation & Rewriting (The Router):** Pass the user's prompt and recent chat history to the LLM. The LLM will output a JSON determining:
    - **Intent:** `RAG` (needs external knowledge) or `CHAT` (can answer based on history/general pleasantries).
    - **Search Query:** If `RAG`, a rewritten, self-contained version of the query that resolves pronouns/context (e.g., "what are its advantages?" -> "what are the advantages of Depth First Search?").
3. **Execution Path:**
    - **If `CHAT`:** Append prompt to history, send history to LLM to get a response.
    - **If `RAG`:** 
        - Pass the *Search Query* to `generate_alternative_queries`.
        - Retrieve chunks via RRF.
        - Pass chunks + conversation history + original prompt to the LLM to generate the final response.
4. **History Update:** Save both the user prompt and the final LLM response to the conversation history. The response string is returned to the UI layer for display.

## 2. File Changes

### `conversation_manager.py` (New File)
- **State:** Maintains a `self.history` list `[{"role": "user", "content": "..."}, {"role": "model", "content": "..."}]`.
- **Method `process_message(user_input: str) -> str`:** The core method that takes a user query, processes it through the routing logic, updates history, and returns the assistant's string response. This makes the manager UI-agnostic.
- **Method `evaluate_and_rewrite()`:** A quick LLM call to classify intent and rewrite queries.

### `llm_generator.py` (Updates)
- **Update `generate_response()`:** Modify it to accept `history` in addition to `query` and `chunks`. It will inject the history into the prompt so the LLM remembers the context of the conversation.
- **Add `evaluate_intent()`:** A new helper function to call the LLM with a strict JSON schema prompt to decide between `RAG` and `CHAT`, and provide the rewritten query.

### `main.py` (Updates)
- Remove the hardcoded query and RRF logic.
- Implement the "UI Layer" for the console: A simple `while True` loop that accepts user input via `input()`, calls `manager.process_message(prompt)`, and `print()`s the result. This ensures the I/O is separated from the manager logic.

## 3. Key Decisions & Trade-offs

1. **Decoupled Manager vs Embedded Loop:**
   - *Trade-off:* Having the `while True` loop inside the Manager is faster to write but locks the system into terminal-only usage. Decoupling requires the caller script (`main.py`) to handle the loop.
   - *Decision:* We strictly decouple the I/O. The Manager will expose a `process_message` method, making it instantly portable to Streamlit, WebSockets, or REST endpoints.

2. **Combined Routing and Rewriting:**
   - *Trade-off:* We could do this in two separate LLM calls (one to check if RAG is needed, another to rewrite). Doing it in one call is slightly more complex to prompt, but significantly reduces latency for the user.
   - *Decision:* Single LLM call returning a structured format (JSON) like `{"intent": "RAG", "search_query": "..."}`.

3. **History Management Limit:**
   - *Trade-off:* Passing the entire conversation history every turn ensures perfect memory but increases latency, token costs, and risks hitting the context window limit.
   - *Decision:* We will implement a sliding window (e.g., retaining the last 5 conversational turns) when passing history to the LLM and the Router.

4. **Prompt-based History vs SDK Chat Sessions:**
   - *Trade-off:* Some SDKs have native `.start_chat()` objects. However, since we are switching context (sometimes injecting RAG chunks, sometimes not), the system prompt dynamically changes.
   - *Decision:* We will manually stringify the `history` array into the prompt. This gives us precise control over how context and RAG chunks are presented to the LLM.
