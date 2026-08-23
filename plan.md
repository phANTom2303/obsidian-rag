# Obsidian RAG Chatbot Plan

This project aims to build a basic, entirely on-device (except for the LLM API call) RAG (Retrieval-Augmented Generation) chatbot on top of an Obsidian vault. The goal is to keep the architecture simple, utilizing Python scripts and avoiding any background server infrastructure other than the UI.

## Architecture Overview
- **UI Framework:** Streamlit (Provides a fast, all-Python web interface)
- **Vector Database:** ChromaDB (Runs locally on disk via Python)
- **Embeddings:** Local HuggingFace models (e.g., `sentence-transformers` for on-device privacy)
- **LLM Provider:** Google Gemini API
- **Data Source:** Local Obsidian Vault (`.md` files)

---

## Phase 1: Setup and Environment
- Initialize a Python virtual environment to keep dependencies contained.
- Install the required core libraries:
  - `streamlit` (UI framework)
  - `chromadb` (Vector storage)
  - `sentence-transformers` (Local embeddings generation)
  - `google-generativeai` (Gemini API SDK)
  - `python-dotenv` (For managing API keys)

## Phase 2: Document Ingestion (Obsidian Parsing)
- **Goal:** Extract and chunk text from the Obsidian vault.
- **Approach:** 
  - Create an ingestion script (e.g., `ingest.py`).
  - Recursively scan the specified Obsidian vault path for all `.md` files.
  - Perform "good enough" basic parsing: read the raw text, ignoring complex wikilink resolution or attachment parsing to save time.
  - Split the text into manageable chunks (e.g., 500-1000 characters with some overlap) to ensure context fits well into the LLM prompt.

## Phase 3: Embedding & Vector Storage
- **Goal:** Convert text chunks into vectors and store them for fast semantic search.
- **Approach:**
  - Initialize a local, persistent ChromaDB client (stores data in a local folder like `./chroma_db`).
  - Load a lightweight, fast local embedding model (e.g., `all-MiniLM-L6-v2`).
  - Iterate over the parsed text chunks, generate embeddings, and upsert them into a ChromaDB collection.
  - Store metadata (like the source file name and chunk index) alongside the vectors to provide citations later.

## Phase 4: Retrieval and Chat Logic
- **Goal:** Retrieve relevant notes based on a user's query and generate an informed response.
- **Approach:**
  - Create a core module (e.g., `chat_logic.py`).
  - Implement a retrieval function: Embed the user's query using the local model and query ChromaDB for the top *k* most similar chunks.
  - Implement a generation function: Construct a prompt that combines the user's query with the retrieved context chunks.
  - Send the prompt to the Gemini API using the `google-generativeai` SDK.
  - Return the Gemini response along with the source citations.

## Phase 5: Web UI (Streamlit)
- **Goal:** Provide a turn-based chat interface.
- **Approach:**
  - Create the main application file (e.g., `app.py`).
  - Use Streamlit's built-in chat elements (`st.chat_message` and `st.chat_input`).
  - Maintain the conversation history in `st.session_state` so the chat context persists across reruns.
  - Hook up the UI to the functions defined in Phase 4.
  - Display the generated response and optionally list the Obsidian notes that were referenced.

## Phase 6: Execution & Testing
- Run `ingest.py` once to build the vector database.
- Run `streamlit run app.py` to launch the chatbot interface.
- Test with various questions about the notes in the vault to ensure retrieval is accurate and the LLM response is helpful.
