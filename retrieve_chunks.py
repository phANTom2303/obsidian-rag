import os
import pprint
from typing import List, Dict, Any
from vectorize import Vectorizer
from db import ChromaDBManager

class Retriever:
    def __init__(self, vectorizer: Vectorizer = None, db_manager: ChromaDBManager = None, persist_directory: str = "chroma_db", collection_name: str = "obsidian_notes"):
        """
        Initializes the retriever with the vectorizer and ChromaDB manager.
        This avoids reloading the embedding model on every query.
        """
        if vectorizer is not None:
            self.vectorizer = vectorizer
        else:
            print("Loading embedding model...")
            self.vectorizer = Vectorizer()
            
        if db_manager is not None:
            self.db_manager = db_manager
        else:
            print("Connecting to ChromaDB...")
            self.db_manager = ChromaDBManager(
                persist_directory=persist_directory, 
                collection_name=collection_name
            )
            
    def retrieve_chunks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Takes a string query, converts it into vector embeddings, 
        and returns the top-k chunks with the highest similarity score.
        """
        # 1. Convert user query to vector
        query_embedding = self.vectorizer.embed_query(query)
        
        # 2. Make vector search in ChromaDB
        results = self.db_manager.query(query_embedding, n_results=k)
        
        # 3. Format the returned chunks
        retrieved_chunks = []
        if results and results.get("documents") and results["documents"][0]:
            documents = results["documents"][0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            
            for i in range(len(documents)):
                meta = metadatas[i].copy() if i < len(metadatas) and metadatas[i] else {}
                
                if i < len(distances):
                    meta["distance"] = distances[i]
                    
                retrieved_chunks.append({
                    "page_content": documents[i],
                    "metadata": meta
                })
                
        return retrieved_chunks

# Example usage
if __name__ == "__main__":
    retriever = Retriever()
    
    query = "How much did Microsoft pay to acquire GitHub?"
    print(f"\nSearching for: '{query}'\n")
    
    top_chunks = retriever.retrieve_chunks(query, k=5)
    
    for i, chunk in enumerate(top_chunks, 1):
        print("=" * 80)
        print(f"Result {i} (Distance/Score: {chunk['metadata'].get('distance', 'N/A')})")
        print("-" * 80)
        print(chunk["page_content"])
        print()
