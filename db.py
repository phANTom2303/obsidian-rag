import os
import chromadb
from typing import List, Dict, Any

class ChromaDBManager:
    def __init__(self, persist_directory: str = "chroma_db", collection_name: str = "obsidian_notes"):
        """
        Initializes a persistent ChromaDB client in the specified directory.
        """
        # Ensure the path is relative to this file's location (the project root)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.persist_dir = os.path.join(base_dir, persist_directory)
        
        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        # Get or create the collection with cosine similarity
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def upsert_vectors(self, vectorized_data: List[Dict[str, Any]]):
        """
        Takes vectorized dataset and upserts into the ChromaDB collection.
        Expected format for vectorized_data:
        list of dicts with 'id', 'document', 'embedding', 'metadata'
        """
        if not vectorized_data:
            return

        ids = [item['id'] for item in vectorized_data]
        documents = [item['document'] for item in vectorized_data]
        embeddings = [item['embedding'] for item in vectorized_data]
        metadatas = [item['metadata'] for item in vectorized_data]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def query(self, query_embedding: List[float], n_results: int = 5):
        """
        Queries the ChromaDB collection using a query embedding.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results
