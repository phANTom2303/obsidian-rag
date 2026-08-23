import os
from dotenv import load_dotenv
from google import genai
from typing import List, Dict, Any

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

    def generate_response(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Takes a user query and a list of retrieved chunks, reformats the chunks 
        to retain only useful context (like file_name and chunk_number), and 
        calls the LLM to generate a response.
        """
        formatted_context = self._format_chunks(chunks)
        
        prompt = (
            f"You are a helpful assistant for an Obsidian vault. Use the following "
            f"context to answer the user's query.\n\n"
            f"Context:\n{formatted_context}\n\n"
            f"Query: {query}"
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
