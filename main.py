from chunking import chunk_file
from vectorize import Vectorizer
from db import ChromaDBManager
from retrieve_chunks import Retriever
from llm_generator import LLMGenerator
import pprint


def main():
    path = "/home/anish-goenka/Documents/obsidian-vault-current/cp-algorithms.com/graph/depth-first-search.md"
    print("Initializing Vectorizer (this may download the model on first run)...")
    vectorizer = Vectorizer()
    print("\nInitializing ChromaDB Manager...")
    db_manager = ChromaDBManager(persist_directory="chroma_db", collection_name="obsidian_notes")
    retriever = Retriever(vectorizer=vectorizer, db_manager=db_manager)
    
    # print("Chunking document...")
    # try:
    #     chunks = chunk_file(path)
    #     pprint.pprint(chunks)
    # except FileNotFoundError:
    #     print(f"File not found: {path}")
    #     print("Please provide a valid markdown file path to test.")
    #     return
        
    # print(f"Generated {len(chunks)} chunks.")
    
    # print(f"Vectorizing chunks...")
    # vectorized_data = vectorizer.vectorize_chunks(chunks)
    
    # print("\nVectorization complete. Output (Sample first chunk):")
    # if vectorized_data:
    #     pprint.pprint([vectorized_data[0]])
        
    
    # print("Upserting vectors into ChromaDB...")
    # db_manager.upsert_vectors(vectorized_data)
    
    # print("Successfully upserted data to ChromaDB at ./chroma_db")

    userQuery = "how to find lowest common ancestor"
    
    print(f"\nQuerying for: '{userQuery}'")
    
    llm = LLMGenerator()

    # 1. Generate alternative queries
    print("\nGenerating alternative queries for multi-query retrieval...")
    queries = llm.generate_alternative_queries(userQuery, num_queries=3)
    for i, q in enumerate(queries):
        print(f"Query {i+1}: {q}")
        
    # 2. Retrieve chunks with RRF
    print("\nRetrieving chunks using Reciprocal Rank Fusion (RRF)...")
    top_chunks = retriever.retrieve_with_rrf(queries, k=5)
    
    # Optional: Pretty print the returned chunks
    # print("\n--- RRF Search Results ---")
    # pprint.pprint(top_chunks)
    
    # 3. Generate response using LLM
    print("\nGenerating final LLM Response...")
    response = llm.generate_response(userQuery, top_chunks)
    
    print("\n" + "="*50)
    print("LLM RESPONSE")
    print("="*50)
    print(response)
    print("="*50 + "\n")
    
if __name__ == "__main__":
    main()