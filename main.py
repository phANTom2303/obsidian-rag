from chunking import chunk_file
from vectorize import Vectorizer
from db import ChromaDBManager
from retrieve_chunks import Retriever
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

    
    userQuery = "What is time complexity of depth first searh"
    print(f"\nQuerying ChromaDB for: '{userQuery}'")
    
    # 1. Retrieve chunks
    top_chunks = retriever.retrieve_chunks(userQuery, k=5)
    
    # 2. Pretty print the returned chunks
    print("\n--- Search Results ---")
    pprint.pprint(top_chunks)
    
if __name__ == "__main__":
    main()