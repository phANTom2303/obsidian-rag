# Obsidian RAG — Comprehensive Project Report Reference Document

> **Purpose:** This document is a complete technical reference covering every file, feature, algorithm, and design decision in the Obsidian RAG project. It is intended to be provided to an LLM to generate a formal project report.

---

## 1. Project Summary

**Obsidian RAG** is a local-first Retrieval-Augmented Generation (RAG) chatbot system that enables natural-language conversational querying over a user's personal [Obsidian](https://obsidian.md/) markdown knowledge vault. The system combines local on-device document processing, embedding, and vector storage with cloud-based LLM generation via Google's Gemini API to deliver accurate, grounded answers sourced exclusively from the user's own notes.

**Key Value Proposition:** Unlike generic AI chatbots that hallucinate or draw from training data, Obsidian RAG is constrained to answer only from the user's vault contents, making it a trustworthy personal knowledge assistant.

---

## 2. Technology Stack

| Component | Technology | Details |
| :--- | :--- | :--- |
| **Programming Language** | Python 3.10+ | Core application language |
| **LLM Provider** | Google Gemini | `gemini-3.7-flash` model via `google-genai` SDK (v2.19.0) |
| **Embedding Model** | Nomic Embed v1.5 | `nomic-ai/nomic-embed-text-v1.5` via `sentence-transformers` (v6.0.0); 768-dimensional vectors; asymmetric task prefixes |
| **Vector Database** | ChromaDB | `chromadb` v1.5.9; persistent client; HNSW index; cosine similarity metric |
| **State Tracking** | SQLite 3 | `tracker.db`; tracks file paths, modification times, SHA-256 hashes, chunk IDs, sync timestamps |
| **Text Splitting** | LangChain | `langchain-text-splitters` v1.1.2; `RecursiveCharacterTextSplitter` with markdown-specific separators |
| **ML Framework** | PyTorch | `torch` v2.13.0 with CUDA 13 support; `transformers` v5.15.1 |
| **Environment** | python-dotenv | Loads `GEMINI_API_KEY` from `.env` file |
| **CLI Interface** | Built-in `sys.argv` | No external CLI framework; supports `ingest` and `chat` subcommands |
| **Planned UI** | Streamlit | `streamlit` v1.62.0 listed in dependencies (not yet implemented) |

---

## 3. Complete File Inventory and Analysis

### 3.1. `main.py` — CLI Entry Point & Component Factory

**Purpose:** The application's command-line entry point. Parses CLI arguments to run either batch ingestion or interactive chat mode. Implements a factory pattern for shared component initialization.

**Functions:**

| Function | Signature | Description |
| :--- | :--- | :--- |
| `_build_components()` | `() -> Tuple[Vectorizer, ChromaDBManager, Retriever]` | Instantiates shared Vectorizer, ChromaDBManager, and Retriever. Prevents duplicate model loading or DB connections. |
| `run_ingest(path)` | `(path: str) -> None` | Initializes FileTracker and triggers incremental ingestion of a file or directory. Prints progress. |
| `run_chat()` | `() -> None` | Initializes LLMGenerator and ConversationManager. Runs a terminal REPL loop (`You: ` prompt) until `exit`/`quit`/EOF/Ctrl-C. |
| `_usage()` | `() -> None` | Prints usage instructions to stderr and exits with code 1. |

**Configuration:**
- `_DB_PATH`: Resolves to `tracker.db` in the project root directory.
- Hardcoded Chroma settings: `persist_directory="chroma_db"`, `collection_name="obsidian_notes"`.

**Design Pattern:** Thin UI Controller — the console loop contains zero business logic; all orchestration is delegated to `ConversationManager.process_message()`.

**CLI Interface:**
- `python main.py ingest <path>` — Recursively ingests markdown files from a vault directory or a single file.
- `python main.py chat` — Launches the interactive REPL.

---

### 3.2. `chunking.py` — Markdown-Aware Text Chunking

**Purpose:** Splits markdown files into semantically meaningful chunks while protecting code blocks from being fragmented across chunk boundaries.

**Functions:**

| Function | Signature | Description |
| :--- | :--- | :--- |
| `chunk_file(file_path, chunk_size, chunk_overlap)` | `(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]` | Reads a markdown file, protects code blocks, splits text using LangChain, restores code blocks, and returns chunk dicts. |

**Algorithm — Code Block Protection:**
1. Extract all fenced code blocks (triple-backtick) using regex: `` ```.*?``` `` (with `re.DOTALL`).
2. Replace each code block with a unique placeholder token: `__CODEBLOCK_{i}__`.
3. Split the sanitized text using `RecursiveCharacterTextSplitter`.
4. Restore code blocks by replacing placeholders back with original content.
5. Use a `used_placeholders: set` to deduplicate — when chunks overlap, a code block is only restored in the first chunk that contains its placeholder.

**Separator Hierarchy (in order of priority):**
```
\n# , \n## , \n### , \n#### , \n##### , \n###### , \n---, \n\n, \n- , \n* , \n> , \n1. , \n, " ", ""
```

**Output Format:**
```python
{
    "page_content": "chunk text content...",
    "metadata": {
        "file_path": "/absolute/path/to/note.md",
        "file_name": "note.md",
        "chunk_number": 0  # 0-indexed
    }
}
```

**Configuration:** `chunk_size=1000` characters, `chunk_overlap=200` characters.

---

### 3.3. `vectorize.py` — Local Embedding Generation

**Purpose:** Wraps HuggingFace's `SentenceTransformer` to generate 768-dimensional embeddings locally using the Nomic Embed model. Implements asymmetric prefix embedding for optimal retrieval performance.

**Classes & Methods:**

| Class/Method | Signature | Description |
| :--- | :--- | :--- |
| `Vectorizer.__init__` | `(model_name: str = "nomic-ai/nomic-embed-text-v1.5")` | Loads the SentenceTransformer model with `trust_remote_code=True`. |
| `Vectorizer.vectorize_chunks` | `(chunks: List[Dict]) -> List[Dict]` | Prepends `"search_document: "` prefix to each chunk's text, generates batch embeddings, and constructs ChromaDB payload dicts with deterministic IDs. |
| `Vectorizer.embed_query` | `(query: str) -> List[float]` | Prepends `"search_query: "` prefix and returns the embedding as a list of floats. |

**Asymmetric Prefix Embedding:**
- Documents are prefixed with `"search_document: "` during indexing.
- Queries are prefixed with `"search_query: "` during retrieval.
- This is a requirement of the Nomic Embed model architecture for optimal retrieval performance.

**Output Format (for `vectorize_chunks`):**
```python
{
    "id": "note_abc123def456_chunk_0",
    "document": "original chunk text",
    "embedding": [0.123, -0.456, ...],  # 768 floats
    "metadata": {"file_path": "...", "file_name": "...", "chunk_number": 0}
}
```

**Dependencies:** Imports `make_chunk_id` from `file_tracker.py` for deterministic ID generation.

---

### 3.4. `db.py` — ChromaDB Vector Database Manager

**Purpose:** Data access layer wrapping `chromadb.PersistentClient`. Manages vector collection lifecycle and provides CRUD operations for vector storage.

**Classes & Methods:**

| Class/Method | Signature | Description |
| :--- | :--- | :--- |
| `ChromaDBManager.__init__` | `(persist_directory: str = "chroma_db", collection_name: str = "obsidian_notes")` | Connects to persistent ChromaDB storage and initializes collection with cosine similarity metric. |
| `ChromaDBManager.upsert_vectors` | `(vectorized_data: List[Dict]) -> None` | Batch-upserts IDs, documents, embeddings, and metadata to ChromaDB. |
| `ChromaDBManager.delete_by_ids` | `(ids: List[str]) -> None` | Deletes vectors by ID. Safely handles empty lists. |
| `ChromaDBManager.query` | `(query_embedding: List[float], n_results: int = 5) -> Dict` | Executes k-nearest-neighbor search using cosine similarity. |

**Configuration:**
- Storage path: `./chroma_db` (relative to `db.py`'s directory).
- Collection: `"obsidian_notes"`.
- Distance metric: Cosine similarity (`{"hnsw:space": "cosine"}`).

**Design Pattern:** Data Access Object (DAO) / Repository Pattern.

---

### 3.5. `file_tracker.py` — Incremental Sync Engine

**Purpose:** An incremental file synchronization engine backed by SQLite. Implements intelligent change detection to prevent expensive re-embedding of unchanged files. Handles the full lifecycle of vault files: addition, modification, and deletion.

**Module-Level Functions:**

| Function | Signature | Description |
| :--- | :--- | :--- |
| `make_chunk_id` | `(file_path: str, chunk_number: int) -> str` | Generates deterministic, globally unique chunk IDs: `<basename>_<sha256(canonical_path)[:12]>_chunk_<N>`. |
| `_sha256_file` | `(file_path: str) -> str` | Computes SHA-256 digest by streaming file in 64 KB blocks. |

**Classes & Methods:**

| Class/Method | Signature | Description |
| :--- | :--- | :--- |
| `FileTracker.__init__` | `(db_path: str, chroma_manager, vectorizer, excluded_dirs: Optional[set], min_file_bytes: int)` | Initializes SQLite connection and creates `file_tracker` table if not exists. |
| `FileTracker._init_db` | `() -> None` | Creates SQLite table: `file_tracker(file_path TEXT PRIMARY KEY, mtime REAL, sha256 TEXT, chunk_ids TEXT, last_synced REAL)`. |
| `FileTracker._walk_md_files` | `(root: str) -> List[str]` | Recursively discovers `.md` files. Resolves symlinks, skips hidden directories (prefixed with `.`), skips excluded directories, enforces minimum file size. |
| `FileTracker._ingest_file` | `(file_path: str) -> List[str]` | Full ingestion pipeline for a single file: chunk → embed (with `search_document:` prefix) → upsert to ChromaDB → return chunk IDs. |
| `FileTracker.ingest_path` | `(path: str) -> None` | Manual ingestion command. For directories: walks all `.md` files. For each file: applies two-layer change detection before deciding to ingest. |
| `FileTracker.sync` | `(vault_path: str) -> None` | Full three-phase vault sync. |

**Two-Layer Change Detection Algorithm:**
1. **Layer 1 — Fast mtime check:** Compare file's current `os.path.getmtime()` against the stored mtime in SQLite. If unchanged, skip entirely.
2. **Layer 2 — SHA-256 content hash:** If mtime changed, compute file's SHA-256 hash and compare against stored hash. If hash matches (mtime lied — e.g., `touch` or cloud sync), update only the mtime record without re-embedding. If hash differs, trigger full re-ingestion.

**Three-Phase Sync Algorithm:**
1. **Phase 1 — Deletions:** Find files in SQLite that no longer exist on disk. Delete their vectors from ChromaDB. Remove their SQLite records.
2. **Phase 2 — Additions:** Find files on disk not present in SQLite. Run full ingestion pipeline.
3. **Phase 3 — Modifications:** For files present in both, apply two-layer change detection. Re-ingest only truly modified files.

**Transactional Safety:** Vectors are written to ChromaDB before committing metadata to SQLite. This ensures that if the process crashes mid-sync, the next sync will re-detect and re-process the file rather than leaving orphan state.

**Configuration:**
- `_MIN_FILE_BYTES = 10` — Ignores empty/stub markdown files.
- `_EXCLUDED_DIRS = {".obsidian", "_templates", ".git"}` — Skips Obsidian configuration, template, and git directories.

**Deterministic Chunk ID Format:**
- `<file_basename_without_ext>_<sha256(canonical_absolute_path)[:12]>_chunk_<chunk_number>`
- This prevents ID collisions when different vault directories contain files with identical names.

---

### 3.6. `llm_generator.py` — Gemini LLM Interface

**Purpose:** Encapsulates all interactions with the Google Gemini API. Provides four distinct AI capabilities: intent classification with query rewriting, conversational chat, RAG response generation, and multi-query expansion.

**Classes & Methods:**

| Class/Method | Signature | Description |
| :--- | :--- | :--- |
| `LLMGenerator.__init__` | `(model_name: str = "gemini-3.7-flash", temperature: float = 0.7, max_output_tokens: int = 1024, top_p: float = 0.95)` | Initializes the `google.genai.Client` (reads API key from environment). Stores generation config. |
| `LLMGenerator.evaluate_intent` | `(user_input: str, history: Optional[List[Dict]]) -> Tuple[str, str]` | Single LLM call with `temperature=0.0` that returns a tuple: `("RAG", rewritten_search_query)` or `("CHAT", "")`. |
| `LLMGenerator.generate_chat_response` | `(user_input: str, history: Optional[List[Dict]]) -> str` | Generates a conversational reply using only chat history (no vault retrieval). |
| `LLMGenerator.generate_response` | `(query: str, chunks: List[Dict], history: Optional[List[Dict]]) -> str` | Formats retrieved vault chunks into a structured context block, injects conversation history, and generates a grounded answer. |
| `LLMGenerator.generate_alternative_queries` | `(original_query: str, num_queries: int = 3) -> List[str]` | Generates N diverse reformulations of the original query for multi-query retrieval. |
| `LLMGenerator._format_chunks` | `(chunks: List[Dict]) -> str` | Converts chunk dicts into a numbered text block: `[Source: note_name | Chunk: N]\ncontent`. Strips distance scores and file paths. |
| `LLMGenerator._format_history` | `(history: Optional[List[Dict]]) -> str` | Converts conversation turn list into `User: ...\nAssistant: ...\n` format. |
| `LLMGenerator._parse_intent_response` | `(raw: str) -> Tuple[str, str]` | Robust JSON parser for intent classification output. Strips markdown code fences (`` ```json ... ``` ``). Falls back to `("RAG", raw_text)` on any parsing error. |

**Intent Classification — Combined Routing & Query Rewriting:**
- A single Gemini call (with `temperature=0.0` for determinism) evaluates whether the user's message requires vault retrieval.
- If RAG is needed, the same call rewrites the query to be self-contained by resolving pronouns and anaphoric references from conversation history.
- Example: If history contains discussion about "Depth First Search" and user says *"what are its pros?"*, the rewritten query becomes *"what are the pros of Depth First Search?"*.
- Expected JSON output: `{"intent": "RAG", "search_query": "rewritten query"}` or `{"intent": "CHAT", "search_query": ""}`.

**RAG Generation System Prompt (key constraint):**
> *"Answer exclusively from the context below. If there is nothing pertinent to the user's query in the context, simply say 'There is no information in the vault for this query.' Do NOT make up information."*

**Fail-Safe Intent Parsing:**
- Strips markdown code fences (`` ```json `` and `` ``` ``) that Gemini sometimes wraps around JSON output.
- On any `json.JSONDecodeError` or `KeyError`, defaults to `("RAG", original_user_input)` to prevent dropped queries.

**Configuration:**
- Model: `gemini-3.7-flash`
- Generation temperature: `0.7` (for creative responses), `0.0` (for deterministic intent routing)
- `max_output_tokens`: `1024`
- `top_p`: `0.95`

---

### 3.7. `retrieve_chunks.py` — Semantic Retrieval with RRF

**Purpose:** Implements semantic vector search and multi-query rank aggregation using Reciprocal Rank Fusion (RRF). This module bridges embedding queries and ChromaDB lookups.

**Classes & Methods:**

| Class/Method | Signature | Description |
| :--- | :--- | :--- |
| `Retriever.__init__` | `(vectorizer: Vectorizer = None, db_manager: ChromaDBManager = None, persist_directory: str, collection_name: str)` | Accepts injected dependencies or lazily initializes defaults. |
| `Retriever.retrieve_chunks` | `(query: str, k: int = 5) -> List[Dict]` | Embeds a single query with `search_query:` prefix, queries ChromaDB, and returns formatted result dicts. |
| `Retriever.retrieve_with_rrf` | `(queries: List[str], k: int = 5, rrf_k: int = 60) -> List[Dict]` | Runs `retrieve_chunks` for each query in the list, then fuses all result sets using RRF scoring. |

**Reciprocal Rank Fusion (RRF) Algorithm:**
```
For each document d appearing in any result list:
    RRF_Score(d) = Σ  1 / (rank_q(d) + 1 + rrf_k)
                  q∈Q

where:
  - Q = set of all alternative queries
  - rank_q(d) = 0-indexed position of document d in query q's result list
  - rrf_k = 60 (standard smoothing constant from IR literature)
```

**Why RRF:**
- Non-parametric: doesn't require calibrated or normalized distance scores across different query embeddings.
- Robust: documents consistently ranked highly across multiple query perspectives bubble to the top.
- Standard: `rrf_k=60` is the value established in the original RRF paper (Cormack, Clarke, & Büttcher, 2009).

**Output Format:**
```python
{
    "document": "chunk text...",
    "metadata": {"file_path": "...", "file_name": "...", "chunk_number": 0},
    "distance": 0.234,  # cosine distance (single query) or rrf_score (multi-query)
    "id": "note_abc123_chunk_0"
}
```

---

### 3.8. `conversation_manager.py` — Orchestration Layer

**Purpose:** A UI-agnostic orchestrator that manages the complete lifecycle of each conversational turn. Handles multi-turn context, sliding-window history, intent classification, and execution path routing (RAG vs. CHAT).

**Classes & Methods:**

| Class/Method | Signature | Description |
| :--- | :--- | :--- |
| `ConversationManager.__init__` | `(llm: LLMGenerator, retriever: Retriever, *, max_history_turns: int = 5, rrf_top_k: int = 5, num_alt_queries: int = 3)` | Initializes with injected LLM and Retriever dependencies. Sets up empty history list. |
| `ConversationManager.process_message` | `(user_input: str) -> str` | **The single public API.** Appends user message → evaluates intent → routes to RAG or CHAT path → stores response → returns output. |
| `ConversationManager.clear_history` | `() -> None` | Resets the conversation history. |
| `ConversationManager._rag_path` | `(original_query: str, search_query: str) -> str` | Generates alternative queries → retrieves with RRF → generates grounded response. |
| `ConversationManager._chat_path` | `(user_input: str) -> str` | Generates a direct conversational reply from chat history (no retrieval). |
| `ConversationManager._append` | `(role: str, content: str) -> None` | Adds a `{"role": role, "content": content}` dict to the history list. |
| `ConversationManager._recent_history` | `() -> List[Dict[str, str]]` | Returns the last `max_history_turns * 2` messages (each turn = 1 user + 1 assistant message). |

**Turn Lifecycle:**
1. User input received via `process_message()`.
2. Input appended to history as `{"role": "user", "content": "..."}`.
3. `LLMGenerator.evaluate_intent()` called with current input and recent history.
4. **If intent is RAG:**
   - `LLMGenerator.generate_alternative_queries()` creates N diverse query reformulations from the rewritten search query.
   - `Retriever.retrieve_with_rrf()` runs all queries and fuses results.
   - `LLMGenerator.generate_response()` produces a grounded answer from retrieved chunks + history.
5. **If intent is CHAT:**
   - `LLMGenerator.generate_chat_response()` produces a reply from history alone.
6. Response appended to history as `{"role": "assistant", "content": "..."}`.
7. Response string returned to caller.

**Sliding-Window History:**
- Default: 5 turns = 10 messages (5 user + 5 assistant).
- Older messages are silently dropped from the window passed to the LLM, though they remain in the full history list.
- Trade-off: Balances context retention against token consumption and cost.

**Design Rationale (from `plan_conversation_manager.md`):**
1. **Decoupled Manager vs. Embedded Loop:** The manager is fully decoupled from any UI. The same `process_message()` API works for CLI, Streamlit, or FastAPI.
2. **Combined Routing + Rewriting:** A single LLM call handles both intent classification and query rewriting, halving latency compared to two separate calls.
3. **Prompt-based History:** History is stringified into prompts rather than using SDK chat sessions, allowing dynamic prompt switching between RAG context injection and general conversation modes.

---

### 3.9. `requirements.txt` — Dependencies

**Key dependencies by category:**

| Category | Packages |
| :--- | :--- |
| **Generative AI** | `google-genai==2.19.0`, `google-auth==2.56.3` |
| **Vector Database** | `chromadb==1.5.9` |
| **Embeddings & ML** | `sentence-transformers==6.0.0`, `transformers==5.15.1`, `torch==2.13.0`, `tokenizers==0.22.2`, `einops==0.8.2`, `onnxruntime==1.29.0` |
| **GPU Support** | `nvidia-cuda-runtime-cu13`, `nvidia-cudnn-cu13`, `nvidia-cublas-cu13`, etc. |
| **Text Processing** | `langchain-text-splitters==1.1.2`, `langchain-core==1.6.0`, `regex==2026.7.19` |
| **Environment** | `python-dotenv==1.2.3` |
| **Data Validation** | `pydantic==2.13.4` |
| **UI (Planned)** | `streamlit==1.62.0`, `uvicorn==0.52.4`, `starlette==1.6.0` |
| **Utilities** | `rich==15.0.0`, `tqdm==4.70.0` |

---

### 3.10. Supporting Files

| File | Purpose |
| :--- | :--- |
| `.env.example` | Template containing `GEMINI_API_KEY=` for user configuration. |
| `.gitignore` | Excludes `venv`, `__pycache__`, `chroma_db`, `.env`, `tracker.db` from version control. |
| `plan.md` | Original 6-phase implementation roadmap (environment setup → ingestion → embedding → retrieval → UI → testing). |
| `plan_conversation_manager.md` | Design specification for the ConversationManager, including motivation, turn lifecycle, and documented trade-offs. |

---

## 4. Feature Summary

### 4.1. Ingestion Features
- **Recursive vault crawling** with configurable directory exclusions (`.obsidian`, `_templates`, `.git`).
- **Minimum file size filtering** (10 bytes) to skip empty stubs.
- **Symlink resolution** with broken-link error handling.
- **Hidden directory skipping** (any directory prefixed with `.`).
- **Markdown-aware chunking** with heading-hierarchy-based splitting.
- **Code block protection** — fenced code blocks are never split across chunks.
- **Deduplication on chunk overlap** — code blocks restored only once when chunks overlap.
- **Deterministic chunk IDs** — path-hashed IDs prevent collisions across identically named files in different directories.
- **Incremental sync** — only changed files are re-processed.

### 4.2. Retrieval Features
- **Asymmetric embedding** with Nomic Embed v1.5 (separate document/query prefixes).
- **Multi-query expansion** — generates 3 alternative queries for broader coverage.
- **Reciprocal Rank Fusion** — merges results from multiple queries using a mathematically principled ranking algorithm.
- **Cosine similarity search** via ChromaDB's HNSW index.

### 4.3. Conversation Features
- **Intent classification** — automatically routes to RAG or conversational mode.
- **Context-aware query rewriting** — resolves pronouns and references from dialogue history.
- **Sliding-window memory** — retains 5 turns of conversation context.
- **Grounded generation** — LLM constrained to answer only from retrieved vault content.
- **Honest uncertainty** — explicitly states when vault lacks relevant information.
- **Multi-turn dialogue** — supports follow-up questions and conversational continuity.

### 4.4. Robustness Features
- **Two-layer change detection** — mtime fast-path + SHA-256 fallback handles "mtime lies."
- **Fail-safe intent parsing** — malformed LLM JSON output gracefully falls back to RAG mode.
- **Transactional ordering** — ChromaDB writes committed before SQLite state updates.
- **Graceful shutdown** — handles `EOFError`, `KeyboardInterrupt` in the CLI REPL.

---

## 5. Data Flow Diagram

### 5.1. Ingestion Data Flow
```
User provides vault path
    → FileTracker._walk_md_files() discovers all .md files
    → For each file:
        → Check mtime against SQLite record (Layer 1)
        → If mtime changed: compute SHA-256 (Layer 2)
        → If content actually changed:
            → Delete old chunk vectors from ChromaDB
            → chunking.chunk_file() splits markdown into chunks
            → Vectorizer.vectorize_chunks() generates embeddings (with "search_document:" prefix)
            → ChromaDBManager.upsert_vectors() stores in ChromaDB
            → Update SQLite record (file_path, mtime, sha256, chunk_ids, timestamp)
```

### 5.2. Query Data Flow
```
User types a question
    → ConversationManager.process_message()
    → Append to history
    → LLMGenerator.evaluate_intent() → {intent: RAG|CHAT, search_query: "..."}
    
    If RAG:
        → LLMGenerator.generate_alternative_queries() → [query1, query2, query3]
        → Retriever.retrieve_with_rrf([original + alternatives])
            → For each query:
                → Vectorizer.embed_query() with "search_query:" prefix
                → ChromaDBManager.query() → ranked results
            → RRF score fusion across all result sets
            → Return top-K fused chunks
        → LLMGenerator.generate_response(query, chunks, history)
        → Return grounded answer
    
    If CHAT:
        → LLMGenerator.generate_chat_response(input, history)
        → Return conversational reply
    
    → Append response to history
    → Return to user
```

---

## 6. Configuration Parameters

| Parameter | Value | File | Description |
| :--- | :--- | :--- | :--- |
| `chunk_size` | 1000 | `chunking.py` | Maximum characters per text chunk |
| `chunk_overlap` | 200 | `chunking.py` | Character overlap between consecutive chunks |
| `max_history_turns` | 5 | `conversation_manager.py` | Number of conversation turns retained (= 10 messages) |
| `rrf_top_k` | 5 | `conversation_manager.py` | Top K chunks returned after RRF fusion |
| `num_alt_queries` | 3 | `conversation_manager.py` | Number of alternative queries for multi-query retrieval |
| `rrf_k` | 60 | `retrieve_chunks.py` | RRF smoothing constant (standard IR value) |
| `temperature` | 0.7 | `llm_generator.py` | LLM generation temperature |
| `temperature` (intent) | 0.0 | `llm_generator.py` | Deterministic intent classification |
| `max_output_tokens` | 1024 | `llm_generator.py` | Maximum tokens in LLM response |
| `top_p` | 0.95 | `llm_generator.py` | Nucleus sampling parameter |
| `model_name` | `gemini-3.7-flash` | `llm_generator.py` | Gemini model variant |
| `embedding_model` | `nomic-ai/nomic-embed-text-v1.5` | `vectorize.py` | Local embedding model |
| `_MIN_FILE_BYTES` | 10 | `file_tracker.py` | Minimum file size for ingestion |
| `_EXCLUDED_DIRS` | `{.obsidian, _templates, .git}` | `file_tracker.py` | Directories excluded from crawling |

---

## 7. Design Decisions & Trade-offs

### 7.1. Local Embeddings + Cloud LLM (Hybrid Architecture)
- **Decision:** Use local `sentence-transformers` for embedding but Google Gemini for generation.
- **Rationale:** Embeddings run on every document chunk during ingestion (high volume, batch processing) — local execution avoids API costs and latency. Generation is query-time only (low volume) and benefits from Gemini's reasoning capabilities.
- **Privacy benefit:** Document content never leaves the user's machine for embedding.

### 7.2. Combined Intent Classification + Query Rewriting
- **Decision:** A single LLM call handles both routing and query rewriting.
- **Rationale:** Halves the latency compared to two separate calls. The rewriting task requires understanding intent anyway, so combining them is natural.

### 7.3. Prompt-based History vs. SDK Chat Sessions
- **Decision:** History is stringified into prompt text rather than using the Gemini SDK's multi-turn chat session API.
- **Rationale:** Allows dynamic prompt switching between RAG mode (which injects retrieved context) and CHAT mode (which doesn't). SDK chat sessions would lock the system prompt.

### 7.4. Sliding Window vs. Full History
- **Decision:** Only the last 5 turns (10 messages) are passed to the LLM.
- **Rationale:** Bounds token consumption and API costs while retaining enough context for natural multi-turn dialogue. Very old context is rarely relevant.

### 7.5. Deterministic Chunk IDs with Path Hashing
- **Decision:** Chunk IDs include a hash of the canonical file path.
- **Rationale:** Prevents ID collisions when different vault subdirectories contain files with the same name (e.g., `README.md` in multiple folders).

### 7.6. Two-Layer Change Detection
- **Decision:** Check mtime first, then SHA-256 only if mtime changed.
- **Rationale:** mtime is a near-zero-cost check (filesystem metadata). SHA-256 is more expensive (reads entire file). The two-layer approach handles "mtime lies" (from `touch`, cloud sync, etc.) without unnecessary re-embedding.

### 7.7. Code Block Protection During Chunking
- **Decision:** Extract and replace code blocks with placeholders before splitting.
- **Rationale:** Code blocks are semantically indivisible — splitting them across chunks would destroy their meaning and usefulness in retrieval.

---

## 8. Current Project Status

### Implemented
- ✅ Full ingestion pipeline (crawling, chunking, embedding, vector storage)
- ✅ Incremental sync with two-layer change detection
- ✅ Three-phase vault sync (deletions, additions, modifications)
- ✅ Code-block-safe markdown chunking
- ✅ Local asymmetric embedding with Nomic Embed v1.5
- ✅ Persistent vector storage with ChromaDB
- ✅ Semantic retrieval with cosine similarity
- ✅ Multi-query expansion
- ✅ Reciprocal Rank Fusion for result merging
- ✅ Intent-aware routing (RAG vs. CHAT)
- ✅ Context-aware query rewriting
- ✅ Sliding-window conversational memory
- ✅ Grounded answer generation with hallucination prevention
- ✅ Interactive CLI REPL
- ✅ Decoupled architecture ready for multiple frontends

### Planned / In Progress
- ⬜ Streamlit web UI (dependency already included)
- ⬜ FastAPI server mode
- ⬜ Configurable chunking strategies
- ⬜ Support for additional file formats (PDF, images via OCR)

---

## 9. Module Dependency Graph

```
main.py
├── conversation_manager.py
│   ├── llm_generator.py
│   │   └── google-genai (Gemini API)
│   └── retrieve_chunks.py
│       ├── vectorize.py
│       │   ├── sentence-transformers (Nomic Embed)
│       │   └── file_tracker.py (make_chunk_id)
│       └── db.py
│           └── chromadb
├── file_tracker.py
│   ├── chunking.py
│   │   └── langchain-text-splitters
│   ├── vectorize.py
│   └── db.py
├── vectorize.py
├── db.py
└── retrieve_chunks.py
```

---

*This document provides complete coverage of the Obsidian RAG project's codebase, architecture, algorithms, design decisions, and current status for use in generating a formal project report.*
