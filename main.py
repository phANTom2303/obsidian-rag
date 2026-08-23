"""
main.py — Obsidian RAG entry point

Modes
-----
ingest <path>   Ingest a file or directory into ChromaDB (manual mode).
chat            Interactive query/response loop against the indexed vault.
"""

import sys
import os

from db import ChromaDBManager
from file_tracker import FileTracker
from llm_generator import LLMGenerator
from retrieve_chunks import Retriever
from vectorize import Vectorizer


# ---------------------------------------------------------------------------
# Shared initialisation
# ---------------------------------------------------------------------------

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracker.db")


def _build_components():
    """Initialise and return the shared vectorizer, ChromaDB manager, and retriever."""
    print("Initializing vectorizer (may download model on first run)...")
    vectorizer = Vectorizer()

    print("Initializing ChromaDB...")
    db_manager = ChromaDBManager(persist_directory="chroma_db", collection_name="obsidian_notes")

    retriever = Retriever(vectorizer=vectorizer, db_manager=db_manager)
    return vectorizer, db_manager, retriever


# ---------------------------------------------------------------------------
# Ingest mode
# ---------------------------------------------------------------------------

def run_ingest(path: str):
    """
    Manual ingest mode (plan §10).

    If *path* is a file, ingest that single .md file.
    If *path* is a directory, recursively walk it and ingest all .md files,
    skipping unchanged files via two-layer change detection (mtime → SHA-256).
    """
    vectorizer, db_manager, _ = _build_components()

    tracker = FileTracker(
        db_path=_DB_PATH,
        chroma_manager=db_manager,
        vectorizer=vectorizer,
    )

    print(f"\nStarting ingest for: {path}\n")
    tracker.ingest_path(path)


# ---------------------------------------------------------------------------
# Chat mode
# ---------------------------------------------------------------------------

def run_chat():
    """Interactive query loop against the already-indexed vault."""
    _, _, retriever = _build_components()
    llm = LLMGenerator()

    print("\nObsidian RAG — Chat mode. Type 'exit' to quit.\n")
    while True:
        try:
            user_query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_query or user_query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        print("\nGenerating alternative queries...")
        queries = llm.generate_alternative_queries(user_query, num_queries=3)

        print("Retrieving relevant chunks (RRF)...")
        top_chunks = retriever.retrieve_with_rrf(queries, k=5)

        print("Generating response...\n")
        response = llm.generate_response(user_query, top_chunks)

        print("=" * 60)
        print(response)
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _usage():
    print(__doc__)
    print("Usage:")
    print("  python main.py ingest <path>")
    print("  python main.py chat")
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _usage()

    command = sys.argv[1].lower()

    if command == "ingest":
        if len(sys.argv) < 3:
            print("Error: 'ingest' requires a path argument.")
            _usage()
        ingest_path = sys.argv[2]
        run_ingest(ingest_path)

    elif command == "chat":
        run_chat()

    else:
        print(f"Unknown command: {command!r}")
        _usage()