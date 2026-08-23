from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class Vectorizer:
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        # Load the model directly, trust_remote_code is required for nomic
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
    
    def vectorize_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes a list of chunks (dictionaries with 'page_content' and 'metadata')
        and returns a dataset ready to be stored in Chroma DB.
        """
        if not chunks:
            return []
            
        # 1. Add the mandatory prefix for documents
        prefixed_chunks = [f"search_document: {chunk['page_content']}" for chunk in chunks]
        
        # 2. Create the embeddings
        # convert_to_tensor=True is optional but useful for PyTorch operations, we will stick to default which returns numpy arrays
        document_embeddings = self.model.encode(prefixed_chunks)
        
        # 3. Prepare the final dataset
        vectorized_dataset = []
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            file_name = metadata.get("file_name", "unknown")
            chunk_number = metadata.get("chunk_number", i + 1)
            
            vectorized_dataset.append({
                "id": f"{file_name}_chunk_{chunk_number}",
                "document": chunk["page_content"],
                "embedding": document_embeddings[i].tolist(), # Convert numpy array to list
                "metadata": metadata
            })
            
        return vectorized_dataset

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a user query with the required prefix for search.
        """
        # 1. Add the query prefix
        prefixed_query = f"search_query: {query}"
        
        # 2. Embed the query
        query_embedding = self.model.encode(prefixed_query)
        
        return query_embedding.tolist()
