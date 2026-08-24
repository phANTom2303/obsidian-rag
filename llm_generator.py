import json
import os
from dotenv import load_dotenv
from google import genai
from typing import List, Dict, Any, Optional, Tuple

# Load environment variables (like GEMINI_API_KEY) from .env file
load_dotenv()


class LLMGenerator:
    def __init__(
        self, 
        model_name: str = "gemini-3.7-flash",
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
        top_p: float = 0.95
    ):
        """
        Initializes the LLM Generator with the specified model and defaults for generation parameters.
        Assumes the GOOGLE_API_KEY (or similar) is handled by the genai Client environment.
        """
        self.client = genai.Client()
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p

    # ------------------------------------------------------------------
    # Intent evaluation & query rewriting (single LLM call)
    # ------------------------------------------------------------------

    def evaluate_intent(
        self,
        user_input: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[str, str]:
        """Classify the user message as RAG or CHAT and, when RAG, rewrite
        it into a self-contained search query.

        Returns
        -------
        (intent, search_query) : Tuple[str, str]
            *intent* is ``"RAG"`` or ``"CHAT"``.
            *search_query* is the rewritten, self-contained query when the
            intent is RAG, or an empty string for CHAT.
        """
        history_block = self._format_history(history)

        prompt = (
            "You are a routing assistant for a Retrieval-Augmented Generation system "
            "over an Obsidian knowledge vault.\n\n"
            "Your job is to look at the user's latest message and the recent "
            "conversation history and decide:\n"
            "1. **Intent** — Does the user need information retrieved from their "
            "vault (`RAG`), or is this a conversational/general message that can "
            "be answered from the conversation history alone (`CHAT`)?\n"
            "   • Greetings, thank-yous, follow-up opinions, clarifications that "
            "don't need new data → `CHAT`.\n"
            "   • Questions asking for facts, definitions, summaries, or anything "
            "that likely lives in the user's notes → `RAG`.\n"
            "2. **Search Query** — If RAG, rewrite the user's message into a "
            "single, self-contained search query that resolves all pronouns and "
            "references using the conversation history.  If CHAT, leave this "
            "empty.\n\n"
            "Respond with **only** a JSON object (no markdown fences, no extra "
            "text):\n"
            '{"intent": "RAG" or "CHAT", "search_query": "..."}\n\n'
            f"### Conversation History\n{history_block}\n\n"
            f"### Latest User Message\n{user_input}\n"
        )

        interaction = self.client.interactions.create(
            model=self.model_name,
            input=prompt,
            generation_config={
                "temperature": 0.0,  # deterministic routing
                "max_output_tokens": 256,
                "top_p": self.top_p,
            },
        )

        return self._parse_intent_response(interaction.output_text)

    # ------------------------------------------------------------------
    # Chat response (no retrieval)
    # ------------------------------------------------------------------

    def generate_chat_response(
        self,
        user_input: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate a reply that only uses conversation history (no chunks)."""
        history_block = self._format_history(history)

        prompt = (
            "You are a helpful assistant for an Obsidian vault. "
            "Answer the user based on the conversation so far.\n\n"
            f"### Conversation History\n{history_block}\n\n"
            f"### Latest User Message\n{user_input}\n"
        )

        interaction = self.client.interactions.create(
            model=self.model_name,
            input=prompt,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "top_p": self.top_p,
            },
        )

        return interaction.output_text

    # ------------------------------------------------------------------
    # RAG-grounded response
    # ------------------------------------------------------------------

    def generate_response(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Takes a user query and a list of retrieved chunks, reformats the chunks 
        to retain only useful context (like file_name and chunk_number), and 
        calls the LLM to generate a response.

        When *history* is provided it is injected into the prompt so the model
        can maintain conversational continuity.
        """
        formatted_context = self._format_chunks(chunks)
        history_block = self._format_history(history)
        
        prompt = (
            f"You are a helpful assistant for an Obsidian vault. Use the following "
            f"context to answer the user's query.\n\n"
            f"### Conversation History\n{history_block}\n\n"
            f"Context:\n{formatted_context}\n\n"
            f"Query: {query}\n"
            f"Answer exclusively from the context. If the context has partial information, inform that in the output and piece together a good response based on what's available. If there is nothing pertinent to the query in the context, simply say \"There is no information in the vault for this query\"\n\n"
        )
        
        interaction = self.client.interactions.create(
            model=self.model_name,
            input=prompt,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "top_p": self.top_p
            }
        )
        
        return interaction.output_text

    # ------------------------------------------------------------------
    # Alternative-query expansion (unchanged interface)
    # ------------------------------------------------------------------

    def generate_alternative_queries(self, original_query: str, num_queries: int = 3) -> List[str]:
        """
        Generates alternative queries from the original user query for multi-query retrieval.
        """
        prompt = (
            f"You are an AI language model assistant. Your task is to generate {num_queries} "
            f"different versions of the given user query to retrieve relevant documents from a vector database. "
            f"By generating multiple perspectives on the user query, your goal is to help the user overcome "
            f"some of the limitations of the distance-based similarity search. "
            f"Provide these alternative questions separated by newlines, with no extra formatting or numbered lists.\n"
            f"Original query: {original_query}"
        )
        
        interaction = self.client.interactions.create(
            model=self.model_name,
            input=prompt,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "top_p": self.top_p
            }
        )
        
        # Parse the output into a list of queries
        queries = interaction.output_text.strip().split('\n')
        # Clean up any potential empty strings or leading/trailing spaces
        queries = [q.strip() for q in queries if q.strip()]
        
        # Prepend original query to ensure it's always included in retrieval
        return [original_query] + queries[:num_queries]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Cleans up the chunk data for the LLM. 
        Drops file paths and distance scores.
        Keeps file name, chunk numbering, and page content.
        """
        formatted_parts = []
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            
            # Extract useful metadata, omit distance and file path
            file_name = metadata.get("file_name", "Unknown File")
            chunk_number = metadata.get("chunk_number", i)
            content = chunk.get("page_content", "").strip()
            
            chunk_str = f"--- Source: {file_name} (Chunk {chunk_number}) ---\n{content}\n"
            formatted_parts.append(chunk_str)
            
        return "\n".join(formatted_parts)

    @staticmethod
    def _format_history(
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Stringify the history list into a readable block for prompt injection."""
        if not history:
            return "(No prior conversation)"

        lines = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    @staticmethod
    def _parse_intent_response(raw: str) -> Tuple[str, str]:
        """Best-effort parse of the JSON returned by the routing prompt.

        Falls back to RAG with the raw text as the search query so that a
        malformed response never silently drops a user question.
        """
        text = raw.strip()

        # Strip markdown code fences if the model wrapped them anyway
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = "\n".join(text.split("\n")[:-1])
        text = text.strip()

        try:
            data = json.loads(text)
            intent = data.get("intent", "RAG").upper()
            search_query = data.get("search_query", "")

            if intent not in ("RAG", "CHAT"):
                intent = "RAG"

            return intent, search_query
        except (json.JSONDecodeError, AttributeError):
            # Fail-safe: treat as a RAG query so we don't lose information
            return "RAG", text
