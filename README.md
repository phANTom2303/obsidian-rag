# 🧠 Obsidian RAG — Chat With Your Vault

A local-first Retrieval-Augmented Generation (RAG) chatbot that lets you have natural, context-aware conversations with your [Obsidian](https://obsidian.md/) vault. Ask questions about your notes, and get answers grounded entirely in your own knowledge base — no hallucinations, no cloud document uploads.

> **How it works:** Your markdown notes are chunked, embedded locally using a state-of-the-art open-source model, and stored in a local vector database. When you ask a question, the system retrieves the most relevant chunks from your vault and uses Google Gemini to synthesize a grounded answer — citing only what's actually in your notes.

---

## ✨ Features

### 🔍 Intelligent Retrieval
- **Multi-Query Expansion** — Your question is rewritten into multiple diverse search queries to maximize recall across your vault.
- **Reciprocal Rank Fusion (RRF)** — Results from multiple queries are fused using a robust, non-parametric ranking algorithm for superior relevance.
- **Asymmetric Embedding** — Uses separate `search_document:` and `search_query:` prefixes with [Nomic Embed](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) for optimal retrieval accuracy.

### 💬 Conversational Intelligence
- **Intent Routing** — Automatically classifies each message as requiring vault retrieval (RAG) or simple conversation (CHAT), so casual follow-ups don't trigger unnecessary searches.
- **Context-Aware Query Rewriting** — Resolves pronouns and references from conversation history (e.g., *"what are its pros?"* → *"what are the pros of Depth First Search?"*).
- **Sliding-Window Memory** — Maintains the last 5 conversation turns for multi-turn dialogue without unbounded token growth.
- **Grounded Answers** — The LLM is strictly constrained to answer from retrieved vault context. If no relevant information exists, it says so honestly.

### ⚡ Incremental Sync
- **Two-Layer Change Detection** — Fast `mtime` check followed by SHA-256 content verification. Only re-embeds files that have actually changed.
- **Three-Phase Vault Sync** — Handles deletions → additions → modifications in the correct order.
- **Code Block Protection** — Fenced code blocks are preserved intact during chunking — never split across chunk boundaries.

### 🏗️ Architecture
- **Decoupled Design** — The `ConversationManager` separates all business logic from the UI layer, making it trivial to swap the CLI for Streamlit, FastAPI, or any other interface.
- **Local-First Privacy** — Embeddings are generated and stored entirely on your machine. Only your natural-language queries are sent to the Gemini API for answer generation.

---

## 🏛️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                       │
│                                                                 │
│  Obsidian Vault (.md)                                           │
│       │                                                         │
│       ▼                                                         │
│  FileTracker ──► mtime + SHA-256 change detection               │
│       │                                                         │
│       ▼                                                         │
│  chunking.py ──► Markdown-aware splitting (code block safe)     │
│       │                                                         │
│       ▼                                                         │
│  vectorize.py ──► Nomic Embed v1.5 (local, 768-dim)             │
│       │                                                         │
│       ▼                                                         │
│  ChromaDB (persistent, cosine similarity)                       │
│  SQLite tracker.db (sync state)                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      QUERY & CHAT PIPELINE                      │
│                                                                 │
│  User Question                                                  │
│       │                                                         │
│       ▼                                                         │
│  ConversationManager                                            │
│       │                                                         │
│       ▼                                                         │
│  LLMGenerator.evaluate_intent() ──► RAG or CHAT?                │
│       │                                    │                    │
│       ▼ (RAG)                              ▼ (CHAT)             │
│  Generate alternative queries         Direct LLM response       │
│       │                               from chat history         │
│       ▼                                                         │
│  Retriever.retrieve_with_rrf()                                  │
│       │                                                         │
│       ▼                                                         │
│  LLMGenerator.generate_response()                               │
│  (grounded on retrieved vault chunks)                           │
│       │                                                         │
│       ▼                                                         │
│  Assistant Reply                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Google Gemini API Key** — Get one from [Google AI Studio](https://aistudio.google.com/apikey)
- An **Obsidian vault** with markdown notes

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/phANTom2303/obsidian-rag.git
   cd obsidian-rag
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or: venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your API key:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

---

## 📖 Usage

### 1. Ingest Your Vault

Index your entire Obsidian vault (or a specific file):

```bash
# Ingest a full vault directory
python main.py ingest /path/to/your/obsidian/vault

# Ingest a single file
python main.py ingest /path/to/your/obsidian/vault/note.md
```

The ingestion pipeline will:
- Recursively discover all `.md` files (skipping `.obsidian`, `_templates`, `.git`)
- Skip files that haven't changed since the last sync
- Chunk, embed, and store each file incrementally

### 2. Chat With Your Notes

```bash
python main.py chat
```

Then just start asking questions:

```
You: What are the main concepts in my machine learning notes?
Assistant: Based on your vault, the main concepts covered include...

You: What did I write about transformers specifically?
Assistant: In your note "Transformer Architecture", you documented...

You: Thanks, that's helpful!
Assistant: You're welcome! Let me know if you have any other questions.
```

Type `exit` or `quit` to end the session.

---

## 📁 Project Structure

```
obsidian-rag/
├── main.py                  # CLI entry point (ingest / chat commands)
├── chunking.py              # Markdown-aware text chunking with code block protection
├── conversation_manager.py  # Orchestrator: intent routing, history, RAG/CHAT paths
├── llm_generator.py         # Gemini API: intent classification, query expansion, generation
├── retrieve_chunks.py       # Semantic search with multi-query RRF fusion
├── vectorize.py             # Local embedding via Nomic Embed v1.5
├── db.py                    # ChromaDB vector database wrapper
├── file_tracker.py          # Incremental sync engine with SQLite state tracking
├── requirements.txt         # Pinned Python dependencies
├── .env.example             # Environment variable template
├── .gitignore               # Git ignore rules
├── plan.md                  # Original implementation roadmap
└── plan_conversation_manager.md  # ConversationManager design spec
```

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **LLM** | Google Gemini (`gemini-3.7-flash`) via `google-genai` SDK |
| **Embeddings** | `nomic-ai/nomic-embed-text-v1.5` via `sentence-transformers` (local) |
| **Vector Database** | ChromaDB (persistent, cosine similarity, HNSW index) |
| **State Tracking** | SQLite 3 |
| **Text Splitting** | LangChain `RecursiveCharacterTextSplitter` with markdown separators |
| **Language** | Python 3.10+ |

---

## ⚙️ Configuration

Key parameters can be adjusted in the source code:

| Parameter | Default | Location | Description |
| :--- | :--- | :--- | :--- |
| `chunk_size` | `1000` | `chunking.py` | Max characters per chunk |
| `chunk_overlap` | `200` | `chunking.py` | Overlap between consecutive chunks |
| `max_history_turns` | `5` | `conversation_manager.py` | Conversation turns retained in memory |
| `rrf_top_k` | `5` | `conversation_manager.py` | Number of chunks returned after RRF |
| `num_alt_queries` | `3` | `conversation_manager.py` | Alternative queries for multi-query retrieval |
| `rrf_k` | `60` | `retrieve_chunks.py` | RRF smoothing constant |
| `temperature` | `0.7` | `llm_generator.py` | LLM generation temperature |
| `model_name` | `gemini-3.7-flash` | `llm_generator.py` | Gemini model variant |

---

## 🔒 Privacy

- **Embeddings stay local** — Generated on your machine using `sentence-transformers`. Your notes never leave your device for embedding.
- **Only queries go to Gemini** — The Gemini API receives your question and retrieved text chunks for answer synthesis. No full vault uploads.
- **API key stays local** — Stored in `.env`, which is gitignored.

---

## 🗺️ Roadmap

- [x] Core RAG pipeline (chunking → embedding → retrieval → generation)
- [x] Incremental vault sync with change detection
- [x] Multi-query retrieval with Reciprocal Rank Fusion
- [x] Intent-aware routing (RAG vs. conversational)
- [x] Context-aware query rewriting
- [ ] Streamlit web UI
- [ ] FastAPI server mode
- [ ] Configurable chunking strategies
- [ ] Support for additional file formats (PDF, images via OCR)

---

## 📄 License

This project is open source. See the repository for license details.

---

## 🙏 Acknowledgements

- [Obsidian](https://obsidian.md/) — The knowledge base this project is built to serve
- [Nomic AI](https://www.nomic.ai/) — For the excellent open-source embedding model
- [Google Gemini](https://deepmind.google/technologies/gemini/) — For the generative AI backbone
- [ChromaDB](https://www.trychroma.com/) — For the developer-friendly vector database
- [LangChain](https://www.langchain.com/) — For text splitting utilities