"""
conversation_manager.py — Stateful conversation orchestrator for Obsidian RAG.

This module owns the chat-turn lifecycle:

1. Accept a user message via `process_message()`.
2. Route it (RAG vs CHAT) with a single LLM call that also rewrites
   context-dependent queries into self-contained search queries.
3. Execute the appropriate path and return the assistant's reply.
4. Maintain a sliding-window history (configurable, default 5 turns).

The manager is deliberately UI-agnostic: it never reads `input()` or
calls `print()`.  A thin UI layer (console, Streamlit, FastAPI …)
drives the loop and renders output.
"""

from typing import List, Dict

from llm_generator import LLMGenerator
from retrieve_chunks import Retriever


class ConversationManager:
    """High-level orchestrator that wires retrieval and generation together
    while maintaining multi-turn conversational context."""

    def __init__(
        self,
        llm: LLMGenerator,
        retriever: Retriever,
        *,
        max_history_turns: int = 5,
        rrf_top_k: int = 5,
        num_alt_queries: int = 3,
    ):
        """
        Parameters
        ----------
        llm : LLMGenerator
            Shared LLM wrapper (used for routing, generation, and alt-query
            expansion).
        retriever : Retriever
            Shared retriever backed by ChromaDB + Vectorizer.
        max_history_turns : int
            Number of *turns* (user + model pairs) to keep in the sliding
            window.  Each turn contributes two messages.
        rrf_top_k : int
            How many chunks to return from Reciprocal Rank Fusion.
        num_alt_queries : int
            Number of alternative queries to generate for multi-query
            retrieval.
        """
        self.llm = llm
        self.retriever = retriever
        self.max_history_turns = max_history_turns
        self.rrf_top_k = rrf_top_k
        self.num_alt_queries = num_alt_queries

        # Each element: {"role": "user" | "model", "content": "..."}
        self.history: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_message(self, user_input: str) -> str:
        """Process a single user message and return the assistant's reply.

        This is the **only** method that the UI layer needs to call.

        Steps
        -----
        1. Append the user message to history.
        2. Route via ``evaluate_and_rewrite`` (single LLM call).
        3. Execute the RAG or CHAT path.
        4. Append the assistant reply to history and return it.
        """
        # 1. Record the user message
        self._append("user", user_input)

        # 2. Route: decide intent and (optionally) get a rewritten query
        intent, search_query = self.llm.evaluate_intent(
            user_input, self._recent_history()
        )

        # 3. Execute the chosen path
        if intent == "RAG":
            response = self._rag_path(user_input, search_query)
        else:
            response = self._chat_path(user_input)

        # 4. Record the assistant reply
        self._append("model", response)

        return response

    def clear_history(self) -> None:
        """Reset the conversation history."""
        self.history.clear()

    # ------------------------------------------------------------------
    # Execution paths
    # ------------------------------------------------------------------

    def _rag_path(self, original_query: str, search_query: str) -> str:
        """Retrieve context from the vault and generate a grounded answer."""
        alt_queries = self.llm.generate_alternative_queries(
            search_query, num_queries=self.num_alt_queries
        )
        top_chunks = self.retriever.retrieve_with_rrf(
            alt_queries, k=self.rrf_top_k
        )
        return self.llm.generate_response(
            original_query, top_chunks, history=self._recent_history()
        )

    def _chat_path(self, user_input: str) -> str:
        """Answer directly from conversation history / general knowledge."""
        return self.llm.generate_chat_response(
            user_input, history=self._recent_history()
        )

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def _append(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def _recent_history(self) -> List[Dict[str, str]]:
        """Return the last *max_history_turns* full turns (user+model pairs).

        If the history length exceeds the window, we slice from the tail
        keeping ``max_history_turns * 2`` messages (each turn = 2 msgs).
        """
        max_messages = self.max_history_turns * 2
        return self.history[-max_messages:]
