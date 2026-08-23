from chunking import chunk_file
from vectorize import Vectorizer
from db import ChromaDBManager
from retrieve_chunks import Retriever

def pretty_print_chunks(chunks):
    for chunk in chunks:
        page_content = chunk.get("page_content", "")
        metadata = chunk.get("metadata", {})

        print("=" * 80)
        print(f"Chunk #{metadata.get('chunk_number', 'N/A')} - {metadata.get('file_name', 'N/A')}")
        print(f"Path: {metadata.get('file_path', 'N/A')}")
        print("-" * 80)
        print(page_content)
        print()

def pretty_print_vectorized(vectorized_data):
    for item in vectorized_data:
        print("=" * 80)
        print(f"ID: {item.get('id')}")
        print(f"Metadata: {item.get('metadata')}")
        print(f"Embedding Dimension: {len(item.get('embedding', []))}")
        print(f"Embedding Prefix (first 5 floats): {item.get('embedding', [])[:5]}")
        print("-" * 80)
        # print(item.get("document", ""))
        print()

def print_search_results(results):
    search_chunks = []
    if results and results.get("documents") and results["documents"][0]:
        for i in range(len(results["documents"][0])):
            # Add distance/score to metadata for display if desired
            meta = results["metadatas"][0][i].copy()
            if results.get("distances") and results["distances"][0]:
                meta["distance"] = results["distances"][0][i]
                
            search_chunks.append({
                "page_content": results["documents"][0][i],
                "metadata": meta
            })
            
    print("\n--- Search Results ---")
    pretty_print_chunks(search_chunks)

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
    #     # pretty_print_chunks(chunks)
    # except FileNotFoundError:
    #     print(f"File not found: {path}")
    #     print("Please provide a valid markdown file path to test.")
    #     return
        
    # print(f"Generated {len(chunks)} chunks.")
    
    # print(f"Vectorizing chunks...")
    # vectorized_data = vectorizer.vectorize_chunks(chunks)
    
    # print("\nVectorization complete. Output (Sample first chunk):")
    # if vectorized_data:
    #     pretty_print_vectorized([vectorized_data[0]])
        
    
    # print("Upserting vectors into ChromaDB...")
    # db_manager.upsert_vectors(vectorized_data)
    
    # print("Successfully upserted data to ChromaDB at ./chroma_db")

    
    userQuery = "What is time complexity of depth first searh"
    print(f"\nQuerying ChromaDB for: '{userQuery}'")
    
    # 1. Retrieve chunks
    top_chunks = retriever.retrieve_chunks(userQuery, k=5)
    
    # 2. Pretty print the returned chunks
    print("\n--- Search Results ---")
    pretty_print_chunks(top_chunks)
    
if __name__ == "__main__":
    main()